"""Exercise the failing optimizer path on real grouped training data before GPU work."""
import json
from pathlib import Path
import warnings

import numpy as np
import scipy
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from detector.model import HybridModel
from scripts.finetune import initialize_head
from scripts.full_finetune import grouped_folds, inner_split
from scripts.train import TASKS, load_task


def main():
    warnings.filterwarnings('error', category=ConvergenceWarning)
    report = dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                  torch=torch.__version__, tasks={})
    with threadpool_limits(limits=4):
        for task in TASKS:
            data, _ = load_task(Path('data/clean_strict'), task)
            tr = data['train']
            folds = grouped_folds(tr)
            for fold, (fit, holdout) in enumerate(folds):
                inner_fit, _ = inner_split(tr, fit, fold)
                head = initialize_head(tr['emb'][inner_fit], tr['y'][inner_fit], len(tr['classes']))
                assert all(torch.isfinite(p).all() for p in head.parameters())
            fit, holdout = folds[0]
            hybrid = HybridModel().fit(tr['text'][fit], tr['emb'][fit], tr['y'][fit])
            hybrid.calibrate(tr['text'][holdout], tr['emb'][holdout], tr['y'][holdout])
            probabilities = hybrid.predict_proba(tr['text'][holdout], tr['emb'][holdout])
            assert np.isfinite(probabilities).all()
            np.testing.assert_allclose(probabilities.sum(1), 1., atol=1e-6)
            report['tasks'][task] = dict(head_initializations_passed=5, hybrid_fit_and_calibration_passed=True)
            print(task, 'optimizer checks passed', flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
