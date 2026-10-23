"""Local epoch checkpoints, with read-only recovery of older Drive snapshots."""
from pathlib import Path
import os
import shutil
import tempfile


def restore_checkpoint(source, destination):
    """Restore once atomically; never alter the original Drive checkpoint."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or not (source / 'latest.pt').is_file():
        return False
    required = ('latest.pt', 'best.safetensors', 'gradient_audit.json')
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete saved checkpoint at {source}: missing {missing}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.restore-', dir=destination.parent))
    try:
        for name in (*required, 'history.json'):
            if (source / name).is_file():
                shutil.copyfile(source / name, staging / name)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return True
