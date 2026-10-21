import os
from pathlib import Path
import joblib
from detector.embedding import MuRIL
from detector.text import validate_text
from scripts.verify_artifacts import verify

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("task1_binary", "task2_attribution", "task3_codemix")


class Detector:
    def __init__(self):
        artifact_dir = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
        verify(artifact_dir)
        # Load only our locally trained, trusted artifacts; never accept uploaded models.
        self.artifacts = {task: joblib.load(artifact_dir / f"{task}.joblib") for task in TASKS}
        self.encoder = MuRIL()

    def predict(self, text, language):
        cleaned = validate_text(text, language)
        token_count = self.encoder.token_count(cleaned)
        # N-grams otherwise see the whole passage while MuRIL silently sees a prefix.
        if token_count > 128:
            raise ValueError("This passage exceeds MuRIL's 128-token training window. Please select a shorter passage; Indic words can use several tokens.")
        task = "task3_codemix" if language.endswith("_en") else "task1_binary"
        embedding = self.encoder.encode([cleaned])
        bundle = self.artifacts[task]
        probs = bundle["model"].predict_proba([cleaned], embedding)[0]
        label = bundle["metadata"]["classes"][int(probs.argmax())]
        warnings = ["Experimental estimate, not proof of authorship. Edited text and unseen generators may be misclassified."]
        if len(cleaned.split()) < 20:
            warnings.append("This is a short passage, so there are fewer clues for the detector.")
        if float(probs.max()) < 0.65:
            warnings.append("The detector's scores are close. Treat this result as uncertain.")
        family = None
        if label == "ai":
            if len(cleaned.split()) < 8:
                warnings.append("Family attribution needs at least 8 words, matching its training data.")
            else:
                f = self.artifacts["task2_attribution"]
                fp = f["model"].predict_proba([cleaned], embedding)[0]
                names = f["metadata"]["classes"]
                family = {"label": names[int(fp.argmax())], "score": float(fp.max()),
                          "uncertain": bool(fp.max() < 0.60),
                          "scores": dict(zip(names, map(float, fp)))}
                warnings.append("Family scores compare Gemini, Muse and Sarvam only; they cannot identify an unknown generator.")
        return {"label": label, "score": float(probs.max()), "uncertain": bool(probs.max() < 0.65),
                "scores": dict(zip(bundle["metadata"]["classes"], map(float, probs))),
                "family": family, "language": language, "task": task,
                "word_count": len(cleaned.split()), "token_count": token_count,
                "warnings": warnings, "model_version": "muril-char-v1"}
