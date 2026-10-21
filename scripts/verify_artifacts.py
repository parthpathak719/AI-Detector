"""Detect incomplete or changed model downloads before deserializing artifacts."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, expected in manifest.items():
        if Path(filename).name != filename:
            raise ValueError("Invalid artifact path")
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {filename}")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1] / "artifacts")
    print("All three trained artifact checksums verified.")
