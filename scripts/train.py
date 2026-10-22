"""Train-only grouped tuning; validation-only calibration; one final test report.

Run: python -m scripts.train --data-dir data/clean_strict
Never select hyperparameters using the resulting test scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from detector.embedding import MODEL_ID, REVISION
from detector.model import BASELINE, HybridModel

TASKS = ["task1_binary", "task2_attribution", "task3_codemix"]


def load_task(root, task):
    out = {}
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        path = root / f"{task}_{split}.jsonl"
        digest.update(path.read_bytes())
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        feature_path = root / "features" / f"{task}_{split}.npz"
        digest.update(feature_path.read_bytes())
        with np.load(feature_path, allow_pickle=False) as z:
            if [r["id"] for r in rows] != list(z["ids"]):
                raise ValueError(f"{task}/{split}: text and embedding IDs differ")
            data = {k: z[k] for k in ("emb", "y", "group", "lang", "classes")}
        data["text"] = np.asarray([r["text"] for r in rows])
        expected = [r.get("paired_with") or r["id"] for r in rows]
        if expected != list(data["group"]):
            raise ValueError("Source groups do not match records")
        classes = list(data["classes"])
        expected_labels = [classes.index(r["model_family"] if task == "task2_attribution" else r["label"]) for r in rows]
        if expected_labels != list(data["y"]):
            raise ValueError("Labels do not match records")
        if data["emb"].shape != (len(rows), 768) or not np.isfinite(data["emb"]).all():
            raise ValueError("Invalid MuRIL embedding matrix")
        out[split] = data
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(out[a]["group"]) & set(out[b]["group"]):
            raise ValueError(f"Source leakage: {task}/{a}/{b}")
        if set(out[a]["text"]) & set(out[b]["text"]):
            raise ValueError(f"Duplicate text leakage: {task}/{a}/{b}")
    return out, digest.hexdigest()


def metrics(y, probabilities):
    pred = probabilities.argmax(1)
    result = {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
              "log_loss": float(log_loss(y, probabilities, labels=np.arange(probabilities.shape[1]))),
              "brier_multiclass": float(np.mean(np.sum((probabilities - np.eye(probabilities.shape[1])[y]) ** 2, axis=1))),
              "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(probabilities.shape[1])).tolist()}
    return result


def candidates():
    # Predeclared, modest search; baseline included. Fits vocabulary/scaling per fold.
    grid = [BASELINE.copy()]
    for C in (1.0, 5.0, 20.0):
        for weight in (0.5, 1.0, 2.0):
            grid.append(dict(BASELINE, C=C, weight=weight))
    grid.extend([dict(BASELINE, ngram_range=(3, 5)), dict(BASELINE, min_df=2)])
    return grid


def train_task(data_dir, task, output, reports):
    data, digest = load_task(data_dir, task)
    tr, va, te = (data[s] for s in ("train", "val", "test"))
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(tr["text"], tr["y"], tr["group"]))
    trace = []
    for params in candidates():
        scores = []
        for fit_idx, eval_idx in folds:
            m = HybridModel(**params).fit(tr["text"][fit_idx], tr["emb"][fit_idx], tr["y"][fit_idx])
            pred = m.raw_proba(tr["text"][eval_idx], tr["emb"][eval_idx]).argmax(1)
            scores.append(float(f1_score(tr["y"][eval_idx], pred, average="macro")))
        trace.append({"params": params, "fold_macro_f1": scores, "mean_macro_f1": float(np.mean(scores))})
        print(task, params, round(np.mean(scores), 4), flush=True)
    best = max(trace, key=lambda t: t["mean_macro_f1"])
    model = HybridModel(**best["params"]).fit(tr["text"], tr["emb"], tr["y"])
    model.calibrate(va["text"], va["emb"], va["y"])
    p = model.predict_proba(te["text"], te["emb"])
    baseline = HybridModel(**BASELINE).fit(tr["text"], tr["emb"], tr["y"])
    baseline.calibrate(va["text"], va["emb"], va["y"])
    report = {
        "task": task, "classes": te["classes"].tolist(), "dataset_sha256": digest,
        "protocol": "3-fold source-grouped tuning on train only; calibration on val; final evaluation on existing test. These are not new unseen data: the historical study already used these splits.",
        "features": "768 frozen MuRIL mean-pooled embeddings + char_wb TF-IDF; no handcrafted features",
        "encoder": {"id": MODEL_ID, "revision": REVISION, "max_tokens": 128, "pool": "masked_mean"},
        "split_counts": {s: len(data[s]["y"]) for s in data}, "source_group_overlap": 0,
        "search": trace, "selected_params": best["params"],
        "test": metrics(te["y"], p),
        "test_uncalibrated": metrics(te["y"], model.raw_proba(te["text"], te["emb"])),
        "baseline_test": metrics(te["y"], baseline.predict_proba(te["text"], te["emb"])),
        "per_language": {lang: metrics(te["y"][te["lang"] == lang], p[te["lang"] == lang]) for lang in np.unique(te["lang"])},
    }
    joblib.dump({"model": model, "metadata": report}, output / f"{task}.joblib", compress=3)
    (reports / f"{task}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(task, 'TEST', report["test"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/clean_strict"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    with threadpool_limits(limits=2):
        for task in TASKS:
            train_task(args.data_dir, task, args.output, args.reports)
    manifest = {f"{task}.joblib": hashlib.sha256((args.output / f"{task}.joblib").read_bytes()).hexdigest() for task in TASKS}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
