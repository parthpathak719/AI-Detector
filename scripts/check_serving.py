"""Verify that serving embeddings/scores match the selected training pipeline."""
import json
from pathlib import Path
import numpy as np
from detector.service import Detector
from scripts.train import load_task, TASKS
from scripts.finetune import embeddings


def main():
    detector = Detector()
    report = {}
    for task in TASKS:
        data, _ = load_task(Path('data/clean_strict'), task)
        test = data['test']
        indices = np.array([np.flatnonzero(test['lang'] == lang)[0] for lang in np.unique(test['lang'])])
        texts = test['text'][indices].tolist()
        live = detector.encoder.encode(texts, task=task)
        if task in detector.encoder.task_layers:
            folder = Path('.cache/finetune') / task / 'test'
            states = (np.load(folder / 'hidden.npy', mmap_mode='r'), np.load(folder / 'mask.npy', mmap_mode='r'))
            expected = embeddings(detector.encoder.task_layers[task], states, indices)
        else:
            expected = test['emb'][indices]
        np.testing.assert_allclose(live, expected, atol=2e-6, rtol=2e-4)
        model = detector.artifacts[task]['model']
        actual_scores = model.predict_proba(texts, live)
        expected_scores = model.predict_proba(texts, expected)
        np.testing.assert_allclose(actual_scores, expected_scores, atol=2e-5, rtol=2e-4)
        report[task] = {'examples': len(indices), 'encoder_variant': 'fine_tuned_last_block' if task in detector.encoder.task_layers else 'frozen',
                        'max_embedding_absolute_error': float(np.abs(live - expected).max()),
                        'max_score_absolute_error': float(np.abs(actual_scores - expected_scores).max())}
    Path('reports/finetuning/serving_parity.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
