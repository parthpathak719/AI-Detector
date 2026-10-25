import hashlib
import json
import pytest
from scripts import verify_artifacts as artifacts


def test_default_fallback_does_not_override_complete_or_explicit_models(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, '__file__', str(tmp_path/'scripts'/'verify_artifacts.py'))
    monkeypatch.delenv('ARTIFACT_DIR', raising=False)
    root=tmp_path/'artifacts';root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'encoder.safetensors':hashlib.sha256(b'weights').hexdigest()}))
    assert artifacts.resolve_artifact_dir()==root/'baseline'
    (root/'encoder.safetensors').write_bytes(b'weights')
    assert artifacts.resolve_artifact_dir()==root
    artifacts.verify(root)
    # Corruption must fail verification, never silently choose another model.
    (root/'encoder.safetensors').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):artifacts.verify(artifacts.resolve_artifact_dir())
    missing=tmp_path/'explicit-install'
    monkeypatch.setenv('ARTIFACT_DIR',str(missing))
    assert artifacts.resolve_artifact_dir()==missing
    with pytest.raises(FileNotFoundError):artifacts.verify(artifacts.resolve_artifact_dir())
