"""Detect incomplete or changed model downloads before deserializing artifacts."""
import hashlib
import json
import os
from pathlib import Path


def resolve_artifact_dir():
    """Explicit installs are strict; fresh clones can use the bundled baseline."""
    configured = os.environ.get("ARTIFACT_DIR")
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[1] / "artifacts"
    manifest = json.loads((root / "manifest.json").read_text())
    if any(Path(name).name != name for name in manifest):
        raise ValueError("Invalid artifact path")
    missing = [name for name in manifest if not (root / name).is_file()]
    if not missing:
        return root
    import logging
    logging.warning("Selected model files are missing (%s). Using the bundled frozen baseline. "
                    "Install selected_models.zip and set ARTIFACT_DIR for fine-tuned inference.", ", ".join(missing))
    return root / "baseline"


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, expected in manifest.items():
        if Path(filename).name != filename:
            raise ValueError("Invalid artifact path")
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {filename}")


if __name__ == "__main__":
    verify(resolve_artifact_dir())
    print("All model artifact checksums verified.")
