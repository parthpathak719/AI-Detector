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
