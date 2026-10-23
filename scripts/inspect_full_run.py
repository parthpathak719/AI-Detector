"""Read-only Drive run inventory; no GPU or training dependencies required."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

TASKS = ('task1_binary', 'task2_attribution', 'task3_codemix')


def inspect_run(root, output):
    root, output = Path(root), Path(output)
    if not root.is_dir():
        raise ValueError(f'Run folder not found: {root}')
    summary = {'run_root': str(root), 'tasks': {}, 'warnings': [], 'checkpoints': []}
    for task in TASKS:
        folder = root / 'reports' / task
        counts = {}
        for variant in ('frozen', 'lr_2e-06', 'lr_5e-06', 'lr_1e-05'):
            completed = []
            for fold in range(5):
                marker = folder / f'{variant}_fold{fold}.json'
                if marker.exists():
                    try:
                        json.loads(marker.read_text())
                        if not marker.with_suffix('.npz').is_file():
                            raise ValueError('missing prediction arrays')
                        completed.append(fold)
                    except (ValueError, OSError) as error:
                        summary['warnings'].append(f'{marker}: {error}')
            counts[variant] = completed
        summary['tasks'][task] = counts
    for path in sorted((root / 'checkpoints').rglob('*')):
        if path.is_file():
            summary['checkpoints'].append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size})
    summary['completed_finetuned_folds'] = sum(len(v) for counts in summary['tasks'].values()
                                              for k, v in counts.items() if k != 'frozen')
    summary['planned_finetuned_folds'] = 45
    summary['note'] = 'Counts check JSON and array presence, not prediction integrity. No files deleted. Checkpoints are not included.'
    output.parent.mkdir(parents=True, exist_ok=True)
    # Whitelist report formats; never include checkpoints or private parity examples.
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('inventory.json', json.dumps(summary, indent=2))
        for path in sorted((root / 'reports').rglob('*')):
            if path.is_file() and path.suffix == '.json':
                archive.write(path, str(path.relative_to(root)))
        for name in ('status.json', 'runtime_history.jsonl', 'baseline_provenance.json'):
            if (root / name).is_file():
                archive.write(root / name, name)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_run(args.run_root, args.output), indent=2))
