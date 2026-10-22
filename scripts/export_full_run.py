"""Export a completed, serving-verified run without including training text."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from scripts.train import TASKS
from scripts.verify_artifacts import verify


def export_run(root):
    root = Path(root)
    status = json.loads((root / 'status.json').read_text())
    if status.get('phase') != 'complete':
        raise ValueError('Experiment is not complete; refusing to export an unselected model')
    reports, artifacts = root / 'reports', root / 'selected_artifacts'
    parity = json.loads((reports / 'serving_parity.json').read_text())
    if set(parity) != set(TASKS):
        raise ValueError('Serving parity results are incomplete')
    verify(artifacts)
    manifest = json.loads((artifacts / 'manifest.json').read_text())
    expected_files, lines = set(), [
        '# Full MuRIL five-fold experiment', '',
        'Models were selected by mean five-fold source-grouped CV macro-F1 within the training split. '
        'Early stopping used separate inner groups. Test accuracy was measured after selection; it did not choose the model.', '',
        '| Task | Frozen CV F1 | Full-tuned CV F1 | Selected | Selected CV accuracy | Selected test accuracy |',
        '|---|---:|---:|---|---:|---:|',
    ]
    for task in TASKS:
        selection = json.loads((reports / task / 'selection.json').read_text())
        final = json.loads((reports / task / 'final.json').read_text())
        if final['selection'] != selection['selected']:
            raise ValueError(f'Selection and final model report disagree: {task}')
        expected_files.add(f'{task}.joblib')
        full = final['encoder'].get('fine_tuned_full')
        if full:
            if full != f'{task}_muril_full.safetensors':
                raise ValueError('Unexpected full-encoder checkpoint name')
            expected_files.add(full)
        selected = selection['selected']
        if bool(full) != (selected['variant'] != 'frozen'):
            raise ValueError('Selected encoder and checkpoint disagree')
        lines.append(f"| {task} | {selection['best_frozen']['mean_macro_f1']:.2%} | "
                     f"{selection['best_full_finetuned']['mean_macro_f1']:.2%} | {selected['variant']} | "
                     f"{selected['mean_accuracy']:.2%} | {final['test']['accuracy']:.2%} |")
    if set(manifest) != expected_files:
        raise ValueError('Artifact manifest does not match the three selected models')
    lines.extend(['', 'CV scores above were used for tuning and are not unbiased final estimates. '
                  'Small differences do not establish statistical superiority. The existing test set has been examined historically. '
                  'These are separate short-text task metrics, not document-level/cascade accuracy.', '',
                  'The selected_models.zip archive contains an artifacts/ directory for the application. '
                  'It excludes private parity examples and training checkpoints. Back up the current artifacts directory '
                  'before replacing it, then run scripts.verify_artifacts and restart the application.', ''])
    summary = root / 'results.md'
    summary.write_text('\n'.join(lines), encoding='utf-8')
    # Only result reports are exported; no optimizer checkpoints, source text,
    # local parity fixtures, datasets, authentication material or Drive contents.
    with ZipFile(root / 'results.zip.tmp', 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(reports.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.npz'):
                archive.write(path, 'reports/full_finetuning/' + str(path.relative_to(reports)))
        archive.write(summary, 'results.md')
        for name in ('runtime_history.jsonl', 'baseline_provenance.json'):
            if (root / name).exists():
                archive.write(root / name, name)
    (root / 'results.zip.tmp').replace(root / 'results.zip')
    with ZipFile(root / 'selected_models.zip.tmp', 'w', compression=ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(expected_files | {'manifest.json'}):
            archive.write(artifacts / name, 'artifacts/' + name)
    (root / 'selected_models.zip.tmp').replace(root / 'selected_models.zip')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    print(export_run(args.run_root).read_text(encoding='utf-8'))
    print(f'Exports saved to {args.run_root}: results.zip and selected_models.zip')
