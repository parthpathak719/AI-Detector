"""Resumable all-layer MuRIL experiment. Never selects on the test split.

Run from repository root: python -m scripts.full_finetune
Results: reports/full_finetuning; private checkpoints: .cache/full_finetuning.
A completed experiment stages verified serving bundles; it does not overwrite
production artifacts. Interrupted epochs are replayed from their starting state.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import joblib
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer

from detector.embedding import MODEL_DIR, MODEL_ID, REVISION
from detector.model import HybridModel
from scripts.finetune import initialize_head
from scripts.train import TASKS, candidates, load_task, metrics

CACHE = Path('.cache/full_finetuning')
REPORTS = Path('reports/full_finetuning')
LEARNING_RATES = (2e-6, 5e-6, 1e-5)
SEED = 42
BATCH = 16
MAX_EPOCHS = 5
PATIENCE = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def status(**values):
    values.update(pid=os.getpid(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(CACHE / 'status.json', values)
    print(json.dumps(values), flush=True)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def grouped_folds(data, n_splits=5, seed_value=SEED):
    folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed_value)
                 .split(data['text'], data['y'], data['group']))
    for fit, holdout in folds:
        assert not set(data['group'][fit]) & set(data['group'][holdout])
        assert set(data['y'][fit]) == set(data['y'])
        assert set(data['y'][holdout]) == set(data['y'])
    return folds


def inner_split(data, outer_fit, fold):
    subset = {k: data[k][outer_fit] for k in ('text', 'y', 'group')}
    fit, stop = grouped_folds(subset, seed_value=SEED + 100 + fold)[0]
    return outer_fit[fit], outer_fit[stop]


def pooled(model, encoded):
    h = model(**encoded).last_hidden_state
    mask = encoded['attention_mask'].unsqueeze(-1).to(h.dtype)
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1)


def token_batch(tokenizer, texts, indices):
    return {k: v.to(DEVICE) for k, v in tokenizer(
        texts[indices].tolist(), padding=True, truncation=True, max_length=128,
        return_tensors='pt').items()}


def encode(model, tokenizer, texts):
    model.eval()
    result = np.empty((len(texts), 768), dtype=np.float32)
    # Length sorting reduces padding without changing example alignment.
    order = np.argsort([len(t) for t in texts])
    with torch.inference_mode():
        for start in range(0, len(order), BATCH):
            indices = order[start:start + BATCH]
            result[indices] = pooled(model, token_batch(tokenizer, texts, indices)).cpu().numpy()
    return result


def new_model():
    # Pooler is not part of the masked-mean representation. Every parameter of
    # the embedding module and all 12 transformer blocks is trainable.
    model = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                      add_pooling_layer=False).to(DEVICE)
    model.requires_grad_(True)
    return model


def save_weights(path, model):
    temporary = path.with_suffix('.tmp')
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temporary))
    os.replace(temporary, path)


def checkpoint(path, model, head, optimizer, **extra):
    temporary = path.with_suffix('.tmp')
    payload = dict(model=model.state_dict(), head=head.state_dict(), optimizer=optimizer.state_dict(),
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], **extra)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_encoder(tokenizer, data, fit, stop, learning_rate, folder, tag, *, epochs=MAX_EPOCHS):
    """Early stopping uses inner groups only. With stop=None, refit fixed epochs."""
    folder.mkdir(parents=True, exist_ok=True)
    seed(SEED + tag['fold'])
    model = new_model()
    head = initialize_head(data['emb'][fit], data['y'][fit], len(data['classes'])).to(DEVICE)
    parameters = list(model.named_parameters()) + [('head.' + k, v) for k, v in head.named_parameters()]
    groups = []
    for is_head in (False, True):
        for decay in (False, True):
            selected = [p for name, p in parameters if name.startswith('head.') == is_head
                        and (not (name.endswith('bias') or 'LayerNorm.weight' in name)) == decay]
            groups.append({'params': selected, 'lr': 1e-4 if is_head else learning_rate,
                           'base_lr': 1e-4 if is_head else learning_rate, 'weight_decay': .01 if decay else 0.})
    optimizer = torch.optim.AdamW(groups, foreach=False)
    steps_per_epoch = math.ceil(len(fit) / BATCH)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, math.ceil(total_steps * .1))
    start_epoch, best_epoch, best_score, stale, history = 0, 0, -1., 0, []
    state_path, best_path = folder / 'latest.pt', folder / 'best.safetensors'
    if state_path.exists():
        saved = torch.load(state_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(saved['model'])
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, best_epoch, best_score = saved['epoch'], saved['best_epoch'], saved['best_score']
        stale, history = saved['stale'], saved['history']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if DEVICE.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        del saved
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert all(p.requires_grad for p in model.parameters())
    # Verify gradients reach embeddings AND every block, not merely the head.
    gradient_audit = None
    for epoch in range(start_epoch, epochs):
        if stop is not None and stale >= PATIENCE:
            break
        model.train(); head.train()
        order = np.random.default_rng(SEED + tag['fold'] * 100 + epoch).permutation(fit)
        total_loss = 0.
        started = time.monotonic()
        for step, start in enumerate(range(0, len(order), BATCH)):
            indices = order[start:start + BATCH]
            global_step = epoch * steps_per_epoch + step
            factor = min((global_step + 1) / warmup_steps,
                         max(0., (total_steps - global_step) / max(1, total_steps - warmup_steps)))
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            optimizer.zero_grad(set_to_none=True)
            emb = pooled(model, token_batch(tokenizer, data['text'], indices))
            loss = torch.nn.functional.cross_entropy(head(emb), torch.tensor(data['y'][indices], device=DEVICE),
                                                     label_smoothing=.05)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite full fine-tuning loss')
            loss.backward()
            if gradient_audit is None:
                probes = {'word_embeddings': model.embeddings.word_embeddings.weight}
                probes.update({f'layer_{i}': block.attention.self.query.weight
                               for i, block in enumerate(model.encoder.layer)})
                gradient_audit = {k: float(p.grad.detach().norm()) if p.grad is not None else 0.
                                  for k, p in probes.items()}
                if not all(np.isfinite(x) and x > 0 for x in gradient_audit.values()):
                    raise RuntimeError(f'Not all encoder layers receive gradients: {gradient_audit}')
                write_json(folder / 'gradient_audit.json', dict(trainable_encoder_parameters=trainable,
                           first_batch_gradient_norms=gradient_audit))
            torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            if step % 20 == 0:
                status(phase='full_encoder_training', **tag, learning_rate=learning_rate,
                       epoch=epoch+1, step=step+1, total_steps=steps_per_epoch,
                       trainable_encoder_parameters=trainable, loss=float(loss.detach()),
                       elapsed_epoch_seconds=round(time.monotonic()-started, 1))
        score = None
        if stop is not None:
            stop_embeddings = encode(model, tokenizer, data['text'][stop])
            with torch.inference_mode():
                prediction = head(torch.from_numpy(stop_embeddings).to(DEVICE)).argmax(1).cpu().numpy()
            score = float(f1_score(data['y'][stop], prediction, average='macro'))
        history.append(dict(epoch=epoch+1, loss=total_loss / len(fit), inner_stop_macro_f1=score,
                            seconds=round(time.monotonic()-started, 2)))
        if stop is None or score > best_score + 1e-4:
            best_score = score if score is not None else -1.
            best_epoch, stale = epoch + 1, 0
            save_weights(best_path, model)
        else:
            stale += 1
        checkpoint(state_path, model, head, optimizer, epoch=epoch+1, best_epoch=best_epoch,
                   best_score=best_score, stale=stale, history=history)
        write_json(folder / 'history.json', history)
        status(phase='epoch_complete', **tag, learning_rate=learning_rate, **history[-1], best_epoch=best_epoch)
    model.load_state_dict(load_file(str(best_path), device=str(DEVICE)))
    del optimizer, head
    gc.collect()
    audit = json.loads((folder / 'gradient_audit.json').read_text())
    return model.eval(), dict(best_epoch=best_epoch, history=history, **audit)


def hybrid_scores(data, fit, holdout, fit_embeddings, holdout_embeddings):
    rows, predictions = [], []
    for index, params in enumerate(candidates()):
        model = HybridModel(**params).fit(data['text'][fit], fit_embeddings, data['y'][fit])
        p = model.raw_proba(data['text'][holdout], holdout_embeddings)
        rows.append(dict(candidate=index, params=params, macro_f1=float(f1_score(data['y'][holdout], p.argmax(1), average='macro')),
                         accuracy=float(accuracy_score(data['y'][holdout], p.argmax(1)))))
        predictions.append(p)
    return rows, np.stack(predictions)


def complete_fold(task, fold, variant, rows, holdout, probabilities, training=None):
    folder = REPORTS / task
    folder.mkdir(parents=True, exist_ok=True)
    key = f'{variant}_fold{fold}'
    # Prediction file precedes the completion marker, so incomplete runs replay.
    np.savez_compressed(folder / f'{key}.npz', holdout_indices=holdout, probabilities=probabilities)
    write_json(folder / f'{key}.json', dict(task=task, fold=fold, variant=variant, candidates=rows, training=training))


def cleanup_checkpoint(folder):
    for name in ('latest.pt', 'best.safetensors'):
        (folder / name).unlink(missing_ok=True)


def baseline(task, data, folds):
    for fold, (fit, holdout) in enumerate(folds):
        if (REPORTS / task / f'frozen_fold{fold}.json').exists():
            continue
        status(phase='frozen_baseline', task=task, fold=fold)
        rows, p = hybrid_scores(data, fit, holdout, data['emb'][fit], data['emb'][holdout])
        complete_fold(task, fold, 'frozen', rows, holdout, p)


def select(task, data):
    trace = []
    for variant in ['frozen'] + [f'lr_{lr:g}' for lr in LEARNING_RATES]:
        folds = [json.loads((REPORTS / task / f'{variant}_fold{i}.json').read_text()) for i in range(5)]
        for c, params in enumerate(candidates()):
            scores = [row['candidates'][c] for row in folds]
            trace.append(dict(variant=variant, candidate=c, params=params,
                              mean_macro_f1=float(np.mean([s['macro_f1'] for s in scores])),
                              std_macro_f1=float(np.std([s['macro_f1'] for s in scores])),
                              mean_accuracy=float(np.mean([s['accuracy'] for s in scores])),
                              folds=scores,
                              best_epochs=[row['training']['best_epoch'] for row in folds] if variant != 'frozen' else []))
    best = max(trace, key=lambda r: (r['mean_macro_f1'], r['variant'] == 'frozen'))
    frozen = max((r for r in trace if r['variant'] == 'frozen'), key=lambda r: r['mean_macro_f1'])
    tuned = max((r for r in trace if r['variant'] != 'frozen'), key=lambda r: r['mean_macro_f1'])
    # Locked before any test predictions are computed. Selection CV scores are
    # used for tuning and must not be presented as an unbiased final estimate.
    selection = dict(task=task, selected=best, best_frozen=frozen, best_full_finetuned=tuned,
                     search=trace, test_used_for_selection=False)
    write_json(REPORTS / task / 'selection.json', selection)
    return selection


def refit(task, data, digest, selection, tokenizer):
    output = CACHE / 'selected_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / task / 'final.json'
    if report_path.exists() and (output / f'{task}.joblib').exists():
        return
    selected = selection['selected']
    tr, va, te = (data[s] for s in ('train', 'val', 'test'))
    vectors = {s: data[s]['emb'] for s in data}
    full_name, train_report = None, None
    folder = CACHE / task / 'refit'
    model = None
    if selected['variant'] != 'frozen':
        learning_rate = float(selected['variant'].removeprefix('lr_'))
        epochs = max(1, int(np.median(selected['best_epochs'])))
        model, train_report = train_encoder(tokenizer, tr, np.arange(len(tr['y'])), None, learning_rate,
                                           folder, dict(task=task, fold=99, stage='refit'), epochs=epochs)
        full_name = f'{task}_muril_full.safetensors'
        save_weights(output / full_name, model)
        for split in vectors:
            vectors[split] = encode(model, tokenizer, data[split]['text'])
    hybrid = HybridModel(**selected['params']).fit(tr['text'], vectors['train'], tr['y'])
    hybrid.calibrate(va['text'], vectors['val'], va['y'])
    p = hybrid.predict_proba(te['text'], vectors['test'])
    frozen = HybridModel(**selection['best_frozen']['params']).fit(tr['text'], tr['emb'], tr['y'])
    frozen.calibrate(va['text'], va['emb'], va['y'])
    report = dict(task=task, classes=tr['classes'].tolist(), dataset_sha256=digest,
                  protocol='5-fold source-grouped train-only selection. Inner grouped early stopping. Validation-only calibration; test after selection lock.',
                  selected_params=selected['params'], selected_cv_macro_f1=selected['mean_macro_f1'],
                  encoder=dict(id=MODEL_ID, revision=REVISION, max_tokens=128, pool='masked_mean', fine_tuned_full=full_name),
                  selection=selected, final_training=train_report,
                  test=metrics(te['y'], p), best_frozen_test=metrics(te['y'], frozen.predict_proba(te['text'], te['emb'])),
                  per_language={lang: metrics(te['y'][te['lang'] == lang], p[te['lang'] == lang]) for lang in np.unique(te['lang'])})
    joblib.dump(dict(model=hybrid, metadata=report), output / f'{task}.joblib', compress=3)
    # Store small parity fixtures locally; a final check uses the actual serving
    # loader, not an independent approximation of its inference code.
    probes = np.array([np.flatnonzero(te['lang'] == lang)[0] for lang in np.unique(te['lang'])])
    np.savez_compressed(output / f'{task}_parity.npz', texts=te['text'][probes],
                        embeddings=vectors['test'][probes], probabilities=p[probes])
    write_json(report_path, report)
    if model is not None:
        del model
        cleanup_checkpoint(folder)
        gc.collect()


def verify_staged():
    from detector.service import Detector
    output = CACHE / 'selected_artifacts'
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
                if p.suffix in ('.safetensors', '.joblib')}
    write_json(output / 'manifest.json', manifest)
    previous = os.environ.get('ARTIFACT_DIR')
    os.environ['ARTIFACT_DIR'] = str(output.resolve())
    try:
        detector = Detector()
        checks = {}
        for task in TASKS:
            with np.load(output / f'{task}_parity.npz', allow_pickle=False) as expected:
                texts = expected['texts'].tolist()
                e = detector.encoder.encode(texts, task=task)
                p = detector.artifacts[task]['model'].predict_proba(texts, e)
                np.testing.assert_allclose(e, expected['embeddings'], atol=3e-6, rtol=3e-4)
                np.testing.assert_allclose(p, expected['probabilities'], atol=3e-5, rtol=3e-4)
                checks[task] = dict(examples=len(texts), max_score_error=float(np.abs(p-expected['probabilities']).max()))
        write_json(REPORTS / 'serving_parity.json', checks)
    finally:
        if previous is None:
            os.environ.pop('ARTIFACT_DIR', None)
        else:
            os.environ['ARTIFACT_DIR'] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    # OS-level lock is released even after a crash; prevents concurrent writers.
    lock = (CACHE / 'run.lock').open('a+')
    if os.name == 'posix':
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        import msvcrt
        lock.seek(0); lock.write('0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    all_data = {task: load_task(Path('data/clean_strict'), task) for task in TASKS}
    protocol = dict(seed=SEED, folds=5, inner_folds=5, learning_rates=LEARNING_RATES, max_epochs=MAX_EPOCHS,
                    patience=PATIENCE, batch_size=BATCH, label_smoothing=.05, head_learning_rate=1e-4,
                    weight_decay=.01, warmup_fraction=.1, max_tokens=128,
                    hybrid_candidates=candidates(), selection_metric='mean_outer_fold_macro_f1',
                    fine_tuning='All embedding parameters and all 12 transformer layers; unused pooler excluded',
                    early_stopping='Inner grouped validation macro-F1 of temporary supervised head; hybrid fits all outer-fit examples',
                    final_epoch_rule='Median of selected learning-rate inner best epochs, then train on all original train',
                    device=str(DEVICE), model_revision=REVISION,
                    dataset_sha256={t: digest for t, (_, digest) in all_data.items()},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # JSON roundtrip normalizes tuples so resumption compares the same types.
    protocol = json.loads(json.dumps(protocol))
    path = REPORTS / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Protocol/data/code changed: use a separate experiment directory instead of mixing runs')
    write_json(path, protocol)
    with threadpool_limits(limits=4):
        for task in ('task2_attribution', 'task1_binary', 'task3_codemix'):
            data, _ = all_data[task]
            folds = grouped_folds(data['train'])
            baseline(task, data['train'], folds)
        if args.baseline_only:
            status(phase='baselines_complete')
            return
        for task in ('task2_attribution', 'task1_binary', 'task3_codemix'):
            data, digest = all_data[task]
            tr = data['train']
            folds = grouped_folds(tr)
            for lr in LEARNING_RATES:
                variant = f'lr_{lr:g}'
                for fold, (fit, holdout) in enumerate(folds):
                    folder = CACHE / task / f'{variant}_fold{fold}'
                    marker = REPORTS / task / f'{variant}_fold{fold}.json'
                    if marker.exists():
                        cleanup_checkpoint(folder)
                        continue
                    inner_fit, stop = inner_split(tr, fit, fold)
                    assert not set(tr['group'][stop]) & set(tr['group'][holdout])
                    model, training = train_encoder(tokenizer, tr, inner_fit, stop, lr, folder,
                                                     dict(task=task, fold=fold, stage='cv'))
                    fit_embeddings = encode(model, tokenizer, tr['text'][fit])
                    holdout_embeddings = encode(model, tokenizer, tr['text'][holdout])
                    rows, p = hybrid_scores(tr, fit, holdout, fit_embeddings, holdout_embeddings)
                    training['counts'] = dict(encoder_fit=len(inner_fit), early_stop=len(stop), hybrid_fit=len(fit), outer_holdout=len(holdout))
                    complete_fold(task, fold, variant, rows, holdout, p, training)
                    del model
                    cleanup_checkpoint(folder)
                    gc.collect()
            selection = select(task, tr)
            refit(task, data, digest, selection, tokenizer)
        verify_staged()
        status(phase='complete', message='All 5-fold comparisons, refits, test reports and serving parity passed. Selected bundles staged; production not yet changed.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not clobber the running worker's status if a second invocation is locked out.
        if not isinstance(error, BlockingIOError):
            status(phase='failed', error=repr(error))
        raise
