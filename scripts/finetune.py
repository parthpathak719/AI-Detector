"""CPU-friendly MuRIL final-block fine-tuning with train-only grouped CV.

The embeddings and first 11 transformer blocks stay frozen. Cache their outputs
once, then fine-tune the final transformer block and supervised head inside each
training fold. Select frozen vs fine-tuned hybrids on mean CV macro-F1, refit on
train, calibrate on val, and evaluate once on test. Never select on test results.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import warnings

import joblib
import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from detector.embedding import MuRIL, MODEL_ID, REVISION
from detector.model import HybridModel
from scripts.train import load_task, metrics

DATA = Path('data/clean_strict')
CACHE = Path('.cache/finetune')
REPORTS = Path('reports/finetuning')
ARTIFACTS = Path('artifacts')
PARAMS = [dict(C=10., weight=1., ngram_range=(2, 4), min_df=3),
          dict(C=10., weight=1., ngram_range=(3, 5), min_df=3),
          dict(C=20., weight=.5, ngram_range=(2, 4), min_df=3)]
EPOCHS = 2
BATCH = 16
LR = 1e-5
SEED = 42


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def cached_states(encoder, task, split, texts):
    folder = CACHE / task / split
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((REVISION + '\n' + '\n'.join(texts)).encode()).hexdigest()
    meta = folder / 'meta.json'
    shape = (len(texts), 128, 768)
    if not (meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get('sha256') == digest):
        hidden = np.lib.format.open_memmap(folder / 'hidden.npy', mode='w+', dtype=np.float32, shape=shape)
        masks = np.lib.format.open_memmap(folder / 'mask.npy', mode='w+', dtype=np.uint8, shape=(len(texts), 128))
        order = np.argsort([len(t) for t in texts])
        started = time.monotonic()
        for start in range(0, len(texts), BATCH):
            idx = order[start:start + BATCH]
            x = encoder.tokenizer(texts[idx].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
            with torch.inference_mode():
                h = encoder.model(**x, output_hidden_states=True).hidden_states[-2].cpu().numpy()
            hidden[idx] = 0
            hidden[idx, :h.shape[1]] = h
            masks[idx] = 0
            masks[idx, :h.shape[1]] = x['attention_mask'].cpu().numpy()
            if start % (BATCH * 10) == 0:
                print(f'{task}/{split}: cache {min(start+BATCH,len(texts))}/{len(texts)} in {time.monotonic()-started:.1f}s', flush=True)
        hidden.flush(); masks.flush()
        meta.write_text(json.dumps({'sha256': digest, 'revision': REVISION, 'shape': shape}))
        del hidden, masks
    return np.load(folder / 'hidden.npy', mmap_mode='r'), np.load(folder / 'mask.npy', mmap_mode='r')


def batch(states, indexes):
    h, mask = states
    m = np.array(mask[indexes], copy=True)
    length = int(m.sum(1).max())
    return torch.from_numpy(np.array(h[indexes, :length], copy=True)), torch.from_numpy(m[:, :length]).float()


def pooled(block, hidden, mask):
    attention = (1. - mask[:, None, None, :]) * torch.finfo(hidden.dtype).min
    output = block(hidden, attention_mask=attention)[0]
    return (output * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


def embeddings(block, states, indexes):
    block.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(indexes), BATCH):
            h, mask = batch(states, indexes[start:start + BATCH])
            results.append(pooled(block, h, mask).numpy())
    return np.concatenate(results)


def initialize_head(frozen_embeddings, labels, classes):
    scaler = StandardScaler().fit(frozen_embeddings)
    clf = LogisticRegression(C=.1, max_iter=2000).fit(scaler.transform(frozen_embeddings), labels)
    weight = clf.coef_ / scaler.scale_[None, :]
    bias = clf.intercept_ - weight @ scaler.mean_
    if classes == 2:
        weight, bias = np.concatenate([-.5 * weight, .5 * weight]), np.concatenate([-.5 * bias, .5 * bias])
    head = torch.nn.Linear(768, classes)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        head.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return head


def train_epochs(encoder, states, frozen_embeddings, y, indexes, classes, epochs, run_seed):
    seed(run_seed)
    block = copy.deepcopy(encoder.model.encoder.layer[-1])
    block.requires_grad_(True)
    head = initialize_head(frozen_embeddings[indexes], y[indexes], classes)
    optimizer = torch.optim.AdamW([{'params': block.parameters(), 'lr': LR},
                                  {'params': head.parameters(), 'lr': 1e-4}], weight_decay=.01)
    rng = np.random.default_rng(run_seed)
    for epoch in range(1, epochs + 1):
        block.train(); head.train()
        order = rng.permutation(indexes)
        total, count = 0., 0
        for start in range(0, len(order), BATCH):
            ix = order[start:start + BATCH]
            h, mask = batch(states, ix)
            labels = torch.from_numpy(y[ix]).long()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(pooled(block, h, mask)), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*block.parameters(), *head.parameters()], 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(ix); count += len(ix)
        print(f'  fine-tune epoch {epoch}: loss={total/count:.5f}', flush=True)
        yield epoch, block, total / count


def score_candidates(task, tr, states, encoder):
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(tr['text'], tr['y'], tr['group']))
    records = {(epoch, p): [] for epoch in range(EPOCHS + 1) for p in range(len(PARAMS))}
    training = []
    for fold, (fit, holdout) in enumerate(folds):
        assert not set(tr['group'][fit]) & set(tr['group'][holdout])
        print(f'{task}: fold {fold+1}/3 ({len(fit)} fit, {len(holdout)} selection)', flush=True)
        def evaluate(epoch, efit, eholdout):
            for pidx, params in enumerate(PARAMS):
                model = HybridModel(**params).fit(tr['text'][fit], efit, tr['y'][fit])
                pred = model.raw_proba(tr['text'][holdout], eholdout).argmax(1)
                row = {'fold': fold + 1, 'macro_f1': float(f1_score(tr['y'][holdout], pred, average='macro')),
                       'accuracy': float(accuracy_score(tr['y'][holdout], pred))}
                records[epoch, pidx].append(row)
                print(f'  {task} epoch={epoch} hybrid={pidx}: selection F1={row["macro_f1"]:.5f}', flush=True)
        evaluate(0, tr['emb'][fit], tr['emb'][holdout])
        for epoch, block, loss in train_epochs(encoder, states, tr['emb'], tr['y'], fit, len(tr['classes']), EPOCHS, SEED + fold):
            training.append({'fold': fold+1, 'epoch': epoch, 'loss': loss})
            evaluate(epoch, embeddings(block, states, fit), embeddings(block, states, holdout))
    trace = []
    for (epoch, pidx), values in records.items():
        trace.append({'encoder_epochs': epoch, 'params': PARAMS[pidx], 'folds': values,
                      'mean_macro_f1': float(np.mean([r['macro_f1'] for r in values])),
                      'std_macro_f1': float(np.std([r['macro_f1'] for r in values])),
                      'mean_accuracy': float(np.mean([r['accuracy'] for r in values]))})
    # Predeclared tie-break favors fewer fine-tuning epochs, then candidate order.
    best = max(trace, key=lambda r: (r['mean_macro_f1'], -r['encoder_epochs']))
    return best, trace, training


def run_task(encoder, task):
    started = time.monotonic()
    data, digest = load_task(DATA, task)
    tr, va, te = (data[s] for s in ('train', 'val', 'test'))
    states = cached_states(encoder, task, 'train', tr['text'])
    # Check the cached penultimate states reconstruct the original encoder output.
    probe = np.arange(min(16, len(tr['y'])))
    restored = embeddings(encoder.model.encoder.layer[-1], states, probe)
    err = float(np.max(np.abs(restored - tr['emb'][probe])))
    if err > 2e-5:
        raise ValueError(f'Cached state parity failed: {err}')
    best, trace, training = score_candidates(task, tr, states, encoder)
    print(f'{task}: SELECTED epochs={best["encoder_epochs"]}, params={best["params"]}, CV F1={best["mean_macro_f1"]:.5f}', flush=True)
    selected_embeddings = {s: data[s]['emb'] for s in data}
    layer_name = None
    if best['encoder_epochs']:
        for _, final_block, _ in train_epochs(encoder, states, tr['emb'], tr['y'], np.arange(len(tr['y'])), len(tr['classes']), best['encoder_epochs'], SEED):
            pass
        selected_embeddings['train'] = embeddings(final_block, states, np.arange(len(tr['y'])))
        for split in ('val', 'test'):
            split_states = cached_states(encoder, task, split, data[split]['text'])
            selected_embeddings[split] = embeddings(final_block, split_states, np.arange(len(data[split]['y'])))
        layer_name = f'{task}_muril_last_layer.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in final_block.state_dict().items()}, str(ARTIFACTS / layer_name))
    model = HybridModel(**best['params']).fit(tr['text'], selected_embeddings['train'], tr['y'])
    model.calibrate(va['text'], selected_embeddings['val'], va['y'])
    p = model.predict_proba(te['text'], selected_embeddings['test'])
    frozen_best = max((r for r in trace if r['encoder_epochs'] == 0), key=lambda r: r['mean_macro_f1'])
    frozen = HybridModel(**frozen_best['params']).fit(tr['text'], tr['emb'], tr['y'])
    frozen.calibrate(va['text'], va['emb'], va['y'])
    report = {
        'task': task, 'classes': te['classes'].tolist(), 'dataset_sha256': digest,
        'protocol': 'Three-fold source-grouped CV entirely within train selects frozen vs final-block-finetuned MuRIL and hybrid settings. Refit train only; calibrate on val only; final test reporting never used for selection.',
        'fine_tuning': {'updated': 'MuRIL encoder.layer.11 and a supervised linear head; head is discarded after representation learning',
                        'frozen': 'token embeddings and encoder layers 0 through 10', 'encoder_learning_rate': LR,
                        'head_learning_rate': 1e-4, 'optimizer': 'AdamW', 'weight_decay': .01, 'batch_size': BATCH,
                        'max_epochs': EPOCHS, 'gradient_clip_norm': 1., 'seed': SEED,
                        'trainable_encoder_parameters': sum(p.numel() for p in encoder.model.encoder.layer[-1].parameters()),
                        'device': 'CPU', 'cached_state_max_absolute_error': err},
        'encoder': {'id': MODEL_ID, 'revision': REVISION, 'max_tokens': 128, 'pool': 'masked_mean', 'fine_tuned_last_layer': layer_name},
        'selected_params': best['params'], 'selected_encoder_epochs': best['encoder_epochs'],
        'selected_cv_macro_f1': best['mean_macro_f1'], 'best_frozen_cv_macro_f1': frozen_best['mean_macro_f1'],
        'split_counts': {s: len(data[s]['y']) for s in data}, 'search': trace, 'training_loss': training,
        'test': metrics(te['y'], p), 'test_uncalibrated': metrics(te['y'], model.raw_proba(te['text'], selected_embeddings['test'])),
        'best_frozen_test': metrics(te['y'], frozen.predict_proba(te['text'], te['emb'])),
        'per_language': {lang: metrics(te['y'][te['lang'] == lang], p[te['lang'] == lang]) for lang in np.unique(te['lang'])},
        'elapsed_seconds': round(time.monotonic() - started, 2),
    }
    backup = CACHE / 'previous_artifacts'
    backup.mkdir(exist_ok=True)
    previous = ARTIFACTS / f'{task}.joblib'
    if previous.exists() and not (backup / previous.name).exists():
        (backup / previous.name).write_bytes(previous.read_bytes())
    temp = ARTIFACTS / f'{task}.joblib.tmp'
    joblib.dump({'model': model, 'metadata': report}, temp, compress=3)
    os.replace(temp, previous)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f'{task}.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ARTIFACTS.iterdir()) if p.suffix in ('.joblib', '.safetensors')}
    (ARTIFACTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{task}: FINAL test accuracy={report["test"]["accuracy"]:.5f}, frozen={report["best_frozen_test"]["accuracy"]:.5f}, elapsed={report["elapsed_seconds"]}s', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['task2_attribution', 'task1_binary', 'task3_codemix'])
    args = parser.parse_args()
    for task in args.tasks:
        if task not in ['task1_binary', 'task2_attribution', 'task3_codemix']:
            raise ValueError('Unknown task')
    CACHE.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('error', category=ConvergenceWarning)
    encoder = MuRIL()
    encoder.model.requires_grad_(False)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for task in args.tasks:
            run_task(encoder, task)


if __name__ == '__main__':
    main()
