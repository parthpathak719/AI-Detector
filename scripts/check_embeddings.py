"""Confirm the pinned encoder matches the archive's cached feature vectors."""
import json
from pathlib import Path
import numpy as np
from detector.embedding import MuRIL, MODEL_ID, REVISION
from detector.text import clean_text
from scripts.train import TASKS

if __name__ == "__main__":
    encoder = MuRIL()
    report = {"encoder": MODEL_ID, "revision": REVISION, "tasks": {}}
    for task in TASKS:
        base = Path("data/clean_strict")
        rows = [json.loads(line) for line in (base / f"{task}_train.jsonl").read_text().splitlines()]
        with np.load(base / "features" / f"{task}_train.npz", allow_pickle=False) as z:
            ref = z["emb"]
        indexes = [next(i for i, r in enumerate(rows) if r["language"] == lang) for lang in sorted({r["language"] for r in rows})]
        actual = encoder.encode([rows[i]["text"] for i in indexes])
        expected = ref[indexes]
        cosines = np.sum(actual * expected, axis=1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
        assert cosines.min() > .9999, "Encoder does not match the cached embeddings"
        changed = sum(clean_text(r["text"]) != r["text"] for r in rows)
        assert changed == 0, "Inference cleaning changes already-cleaned training text"
        report["tasks"][task] = {"checked_vectors": len(indexes), "minimum_cosine": float(cosines.min()),
                                 "max_absolute_error": float(np.abs(actual - expected).max()),
                                 "cleaning_changed_rows": changed, "cleaning_checked_rows": len(rows)}
    Path("reports/embedding_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
