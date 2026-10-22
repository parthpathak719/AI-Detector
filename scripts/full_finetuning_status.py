"""Show saved experiment progress; timestamps do not guarantee a live worker."""
import json
from pathlib import Path


def main():
    reports = Path('reports/full_finetuning')
    status_path = Path('.cache/full_finetuning/status.json')
    result = {
        'last_worker_update': json.loads(status_path.read_text()) if status_path.exists() else None,
        'frozen_folds_completed': len(list(reports.glob('task*/frozen_fold*.json'))),
        'frozen_folds_planned': 15,
        'full_finetuned_folds_completed': len(list(reports.glob('task*/lr_*_fold*.json'))),
        'full_finetuned_folds_planned': 45,
        'tasks': {},
    }
    for name in ('task1_binary', 'task2_attribution', 'task3_codemix'):
        selection = reports / name / 'selection.json'
        if not selection.exists():
            result['tasks'][name] = {'state': 'pending'}
            continue
        data = json.loads(selection.read_text())
        result['tasks'][name] = {'state': 'selected', 'winner': data['selected']['variant'],
                                 'cv_macro_f1': data['selected']['mean_macro_f1']}
        final = reports / name / 'final.json'
        if final.exists():
            result['tasks'][name].update(state='refit_and_tested', test=json.loads(final.read_text())['test'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
