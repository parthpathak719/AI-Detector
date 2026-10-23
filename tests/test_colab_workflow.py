import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from scripts.colab_run import attach_folder, copy_baselines
from scripts.export_full_run import export_run
from scripts.train import TASKS


def test_persistent_checkpoint_link_preserves_existing_data(tmp_path):
    link, destination = tmp_path / 'local', tmp_path / 'drive'
    attach_folder(link, destination)
    (link / 'checkpoint').write_text('saved')
    attach_folder(link, destination)
    assert (destination / 'checkpoint').read_text() == 'saved'
    with pytest.raises(ValueError):
        attach_folder(link, tmp_path / 'another_run')
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(ValueError):
        attach_folder(existing, destination)


def test_baseline_copy_refuses_to_mix_runs(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'reports'
    source.mkdir()
    path = source / 'fold.json'
    path.write_text('original')
    copy_baselines(source, output, [path])
    copy_baselines(source, output, [path])
    path.write_text('changed')
    with pytest.raises(ValueError):
        copy_baselines(source, output, [path])


def test_export_requires_complete_run_and_excludes_private_examples(tmp_path):
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    put(tmp_path / 'status.json', {'phase': 'full_encoder_training'})
    with pytest.raises(ValueError, match='not complete'):
        export_run(tmp_path)
    put(tmp_path / 'status.json', {'phase': 'complete'})
    reports, artifacts = tmp_path / 'reports', tmp_path / 'selected_artifacts'
    artifacts.mkdir()
    put(reports / 'serving_parity.json', {task: {'examples': 1, 'max_score_error': 0.} for task in TASKS})
    manifest = {}
    for task in TASKS:
        selected = {'variant': 'frozen', 'mean_macro_f1': .85, 'mean_accuracy': .86}
        put(reports / task / 'selection.json', dict(selected=selected, best_frozen=selected,
                                                   best_full_finetuned={'mean_macro_f1': .84}))
        put(reports / task / 'final.json', dict(selection=selected, encoder={'fine_tuned_full': None},
                                               test={'accuracy': .83}))
        blob = f'{task} test artifact'.encode()
        (artifacts / f'{task}.joblib').write_bytes(blob)
        manifest[f'{task}.joblib'] = hashlib.sha256(blob).hexdigest()
    put(artifacts / 'manifest.json', manifest)
    (artifacts / 'private_parity.npz').write_bytes(b'private training passage')
    summary = export_run(tmp_path)
    assert '83.00%' in summary.read_text()
    with ZipFile(tmp_path / 'selected_models.zip') as archive:
        assert set(archive.namelist()) == {'artifacts/' + n for n in [*manifest, 'manifest.json']}
    with ZipFile(tmp_path / 'results.zip') as archive:
        assert all('parity.npz' not in n for n in archive.namelist())
    (artifacts / f'{TASKS[0]}.joblib').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        export_run(tmp_path)


def test_colab_runner_requires_gpu(tmp_path, monkeypatch):
    import scripts.colab_run as runner
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(runner.sys, 'argv', ['colab_run', '--run-root', str(tmp_path / 'run')])
    with pytest.raises(SystemExit, match='No GPU is attached'):
        runner.main()
    assert not (tmp_path / 'run').exists()


def test_restore_legacy_snapshot_preserves_drive_and_local_progress(tmp_path):
    from scripts.colab_storage import restore_checkpoint
    source, target = tmp_path / 'drive', tmp_path / 'local' / 'fold'
    source.mkdir()
    for name in ('latest.pt', 'best.safetensors', 'gradient_audit.json'):
        (source / name).write_bytes(b'original')
    assert restore_checkpoint(source, target)
    (target / 'latest.pt').write_bytes(b'new progress')
    assert not restore_checkpoint(source, target)
    assert (target / 'latest.pt').read_bytes() == b'new progress'
    assert (source / 'latest.pt').read_bytes() == b'original'


def test_restore_rejects_incomplete_snapshot(tmp_path):
    from scripts.colab_storage import restore_checkpoint
    source, target = tmp_path / 'drive', tmp_path / 'local'
    source.mkdir()
    (source / 'latest.pt').write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='Incomplete'):
        restore_checkpoint(source, target)
    assert not target.exists()


def test_inventory_counts_durable_folds_and_excludes_weights(tmp_path):
    from scripts.inspect_full_run import inspect_run
    root = tmp_path / 'run'
    folder = root / 'reports' / 'task2_attribution'
    folder.mkdir(parents=True)
    (folder / 'lr_2e-06_fold0.json').write_text('{}')
    (folder / 'lr_2e-06_fold0.npz').write_bytes(b'arrays')
    (folder / 'lr_2e-06_fold1.json').write_text('{}')
    (root / 'checkpoints').mkdir()
    (root / 'checkpoints' / 'latest.pt').write_bytes(b'weights')
    output = tmp_path / 'inventory.zip'
    result = inspect_run(root, output)
    assert result['completed_finetuned_folds'] == 1
    assert result['warnings']
    with ZipFile(output) as archive:
        assert all(not name.endswith(('.pt', '.npz')) for name in archive.namelist())
    assert (root / 'checkpoints' / 'latest.pt').exists()


def test_failed_restore_leaves_no_partial_checkpoint(tmp_path, monkeypatch):
    import scripts.colab_storage as storage
    source, target = tmp_path / 'drive', tmp_path / 'local' / 'fold'
    source.mkdir()
    for name in ('latest.pt', 'best.safetensors', 'gradient_audit.json'):
        (source / name).write_bytes(b'original')
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(storage.shutil, 'copyfile', fail)
    with pytest.raises(OSError, match='disk full'):
        storage.restore_checkpoint(source, target)
    assert not target.exists()
    assert not list(target.parent.glob('.restore-*'))
    assert (source / 'latest.pt').read_bytes() == b'original'
