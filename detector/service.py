import os
from pathlib import Path
import joblib
import numpy as np
from detector.embedding import MuRIL
from detector.text import validate_text, source_excerpts
from scripts.verify_artifacts import verify

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("task1_binary", "task2_attribution", "task3_codemix")


class Detector:
    def __init__(self):
        artifact_dir = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
        verify(artifact_dir)
        self.artifacts = {task: joblib.load(artifact_dir / f"{task}.joblib") for task in TASKS}
        self.encoder = MuRIL()
        for task, bundle in self.artifacts.items():
            layer = bundle["metadata"].get("encoder", {}).get("fine_tuned_last_layer")
            full = bundle["metadata"].get("encoder", {}).get("fine_tuned_full")
            if layer and full:
                raise ValueError("Conflicting encoder variants")
            if full:
                if Path(full).name != full:
                    raise ValueError("Invalid encoder artifact path")
                self.encoder.configure_full_task(task, artifact_dir / full)
            if layer:
                if Path(layer).name != layer:
                    raise ValueError("Invalid encoder artifact path")
                self.encoder.configure_task(task, artifact_dir / layer)

    def predict(self, text, language):
        cleaned = validate_text(text, language)
        chunks = self.encoder.chunks(cleaned)
        excerpts = source_excerpts(text, cleaned, chunks)
        task = "task3_codemix" if language.endswith("_en") else "task1_binary"
        embeddings = self.encoder.encode(chunks, task=task)
        bundle = self.artifacts[task]
        chunk_probs = bundle["model"].predict_proba(chunks, embeddings)
        word_counts = np.array([len(c.split()) for c in chunks])
        tokens = [self.encoder.token_count(c) for c in chunks]
        probs = np.average(chunk_probs, axis=0, weights=word_counts)
        names = bundle["metadata"]["classes"]
        ai_index = names.index("ai")
        labels = [names[int(p.argmax())] for p in chunk_probs]
        label = names[int(probs.argmax())]
        warnings = ["Experimental estimate, not proof of authorship. Edited text and unseen generators may be misclassified."]
        if len(chunks) > 1:
            warnings.append("Document scores are word-weighted averages of segment scores, not the percentage of text written by AI.")
        if len(cleaned.split()) < 20:
            warnings.append("This is a short passage, so there are fewer clues for the detector.")
        if float(probs.max()) < 0.65:
            warnings.append("The detector's scores are close. Treat this result as uncertain.")
        model_uncertain = bool(probs.max() < 0.65)
        # Scores were calibrated on short-text validation data, not documents.
        # Preserve scores and model labels; do not invent a confidence cap.
        assessment = "document_model_signal" if len(chunks) > 1 else "model_signal_only"
        warnings.insert(0, "Model scores are not the probability that the author is human or AI. A high human score can still be an AI-generated false negative.")
        family = None
        eligible = [i for i, outcome in enumerate(labels) if outcome == "ai" and word_counts[i] >= 8]
        if label == "ai":
            if not eligible:
                warnings.append("No AI-leaning segment contains the 8 words needed for family attribution.")
            else:
                f = self.artifacts["task2_attribution"]
                detection_encoder = bundle["metadata"].get("encoder", {})
                attribution_encoder = f["metadata"].get("encoder", {})
                detection_layer = (detection_encoder.get("fine_tuned_last_layer"), detection_encoder.get("fine_tuned_full"))
                attribution_layer = (attribution_encoder.get("fine_tuned_last_layer"), attribution_encoder.get("fine_tuned_full"))
                family_embeddings = embeddings[eligible]
                if detection_layer != attribution_layer:
                    family_embeddings = self.encoder.encode([chunks[i] for i in eligible], task="task2_attribution")
                family_probs = f["model"].predict_proba([chunks[i] for i in eligible], family_embeddings)
                fp = np.average(family_probs, axis=0, weights=word_counts[eligible] * chunk_probs[eligible, ai_index])
                families = f["metadata"]["classes"]
                family = {"label": families[int(fp.argmax())], "score": float(fp.max()),
                          "uncertain": bool(fp.max() < 0.60), "analyzed_segments": len(eligible),
                          "scores": dict(zip(families, map(float, fp)))}
                warnings.append("Family scores compare Gemini, Muse and Sarvam only; they cannot identify an unknown generator.")
        return {"label": label, "score": float(probs.max()), "uncertain": bool(model_uncertain or len(chunks) > 1),
                "model_uncertain": model_uncertain, "assessment": assessment,
                "source_text": text,
                "scores": dict(zip(names, map(float, probs))),
                "family": family, "language": language, "task": task,
                "word_count": len(cleaned.split()), "token_count": sum(t - 2 for t in tokens),
                "chunk_count": len(chunks), "ai_segments": labels.count("ai"),
                "aggregation": "word_weighted_mean" if len(chunks) > 1 else "single_segment",
                "segments": [{"index": i + 1, **excerpts[i], "model_text": c, "word_count": int(word_counts[i]),
                              "token_count": tokens[i], "label": labels[i],
                              "ai_score": float(chunk_probs[i, ai_index])} for i, c in enumerate(chunks)],
                "warnings": warnings, "model_version": "muril-char / document-v3",
                "encoder_variant": "fully fine-tuned" if bundle["metadata"].get("encoder", {}).get("fine_tuned_full") else "fine-tuned final block" if bundle["metadata"].get("encoder", {}).get("fine_tuned_last_layer") else "frozen"}
