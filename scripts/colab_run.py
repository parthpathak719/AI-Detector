"""Run the fixed full-MuRIL search on a Colab GPU with Drive persistence.

The training protocol is unchanged. Only paths and progress persistence differ.
Run one Colab runtime per Drive run directory. This script requires CUDA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import scipy
import sklearn

from scripts import full_finetune as experiment
from scripts.colab_storage import restore_checkpoint
from scripts.train import TASKS, load_task


def attach_folder(link, destination):
    destination.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != destination.resolve():
            raise ValueError(f'Checkpoint link points to a different run: {link}')
    elif link.exists():
        raise ValueError(f'Refusing to replace an existing local checkpoint directory: {link}')
    else:
        link.symlink_to(destination.resolve(), target_is_directory=True)


def reusable_baselines(source, data_root):
    """Validate frozen-fold provenance and OOF predictions before reusing them."""
    protocol = json.loads((source / 'protocol.json').read_text())
    script_hash = hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest()
    if protocol['script_sha256'] != script_hash:
        raise ValueError('Committed baselines use different training code; rerun without baseline reuse')
    files = []
    for task in TASKS:
        data, digest = load_task(data_root, task)
        if digest != protocol['dataset_sha256'][task]:
            raise ValueError(f'Dataset differs from frozen baseline: {task}')
        tr = data['train']
        for fold, (_, holdout) in enumerate(experiment.grouped_folds(tr)):
            metadata = source / task / f'frozen_fold{fold}.json'
            arrays = metadata.with_suffix('.npz')
            rows = json.loads(metadata.read_text())
            with np.load(arrays, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['holdout_indices'], holdout)
                probabilities = z['probabilities']
                expected = (len(experiment.candidates()), len(holdout), len(tr['classes']))
                if probabilities.shape != expected or not np.isfinite(probabilities).all():
                    raise ValueError('Invalid frozen out-of-fold predictions')
                np.testing.assert_allclose(probabilities.sum(-1), 1., atol=1e-6)
                for index, item in enumerate(rows['candidates']):
                    expected_params = json.loads(json.dumps(experiment.candidates()[index]))
                    if item['params'] != expected_params:
                        raise ValueError('Frozen hybrid configuration differs')
                    actual = experiment.metrics(tr['y'][holdout], probabilities[index])
                    for metric in ('accuracy', 'macro_f1'):
                        if abs(actual[metric] - item[metric]) > 1e-10:
                            raise ValueError(f'Frozen metric disagrees with saved predictions: {metric}')
            files.extend((metadata, arrays))
    return files


def copy_baselines(source, destination, files):
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise ValueError(f'Refusing to mix different frozen results: {target}')
            continue
        temporary = target.with_suffix(target.suffix + '.tmp')
        shutil.copyfile(path, temporary)
        os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--recompute-baselines', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('No GPU is attached. In Colab choose Runtime > Change runtime type > T4 GPU, reconnect, then Run all.')
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Epoch snapshots must stay off Drive: replacement through Drive FUSE can
    # retain multi-GB old files in Trash. Completed fold reports remain durable.
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    local = Path('.cache/colab_local_checkpoints') / identity
    local.mkdir(parents=True, exist_ok=True)
    original_train = experiment.train_encoder
    def recover_then_train(tokenizer, data, fit, stop, learning_rate, folder, tag, **kwargs):
        relative = Path(folder).relative_to(local)
        if restore_checkpoint(root / 'checkpoints' / relative, folder):
            print(f'Recovered saved Drive checkpoint: {relative}', flush=True)
        return original_train(tokenizer, data, fit, stop, learning_rate, folder, tag, **kwargs)
    experiment.train_encoder = recover_then_train
    print('Epoch checkpoints stay on local disk. Completed folds and final models persist '
          'on Drive. A fresh runtime may repeat the unfinished fold.', flush=True)
    attach_folder(local / 'selected_artifacts', root / 'selected_artifacts')
    source = Path('reports/full_finetuning')
    reports = root / 'reports'
    reports.mkdir(exist_ok=True)
    if not args.recompute_baselines:
        files = reusable_baselines(source, Path('data/clean_strict'))
        copy_baselines(source, reports, files)
        experiment.write_json(root / 'baseline_provenance.json', dict(
            source='Committed CPU frozen baselines; original source-group folds and probabilities revalidated',
            source_protocol=json.loads((source / 'protocol.json').read_text()),
            copied_files=[str(p.relative_to(source)) for p in files]))
        print('Verified and reused all 15 frozen folds; GPU fine-tuned folds start independently.', flush=True)
    experiment.CACHE, experiment.REPORTS = local, reports
    # Stay in float32 and disable TF32 so CPU/GPU serving parity remains strict.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                       numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                       gpu=torch.cuda.get_device_name(0), total_gpu_memory=torch.cuda.get_device_properties(0).total_memory,
                       run_root=str(root), tf32=False)
    with (root / 'runtime_history.jsonl').open('a', encoding='utf-8') as history:
        history.write(json.dumps(environment) + '\n')
    original_status = experiment.status
    def persistent_status(**values):
        original_status(**values)
        experiment.write_json(root / 'status.json', json.loads((local / 'status.json').read_text()))
    experiment.status = persistent_status
    sys.argv = [sys.argv[0]]
    try:
        experiment.main()
    except Exception as error:
        if not isinstance(error, BlockingIOError):
            persistent_status(phase='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
