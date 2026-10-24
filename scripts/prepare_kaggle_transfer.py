"""Original Colab CPU transfer cell used before the Kaggle resume.

Paste into a Colab CPU cell with Drive access. This targets the interrupted v2
stage (Task 2 complete, Task 1 lr_1e-05 fold 0 interrupted), not arbitrary runs.
Writes only runtime-local ZIP downloads; original Drive files are read only.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import hashlib, json, subprocess, sys
from google.colab import drive, files

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/AI-Detector/full-muril-gpu-v2-scipy114')
checkpoint = 'checkpoints/task1_binary/lr_1e-05_fold0'
required = [
    'reports/protocol.json', 'reports/task2_attribution/final.json',
    'selected_artifacts/task2_attribution.joblib',
    'selected_artifacts/task2_attribution_muril_full.safetensors',
    'selected_artifacts/task2_attribution_parity.npz',
    *[checkpoint + '/' + name for name in
      ['latest.pt', 'best.safetensors', 'gradient_audit.json', 'history.json']],
]
for relative in required:
    assert (root / relative).is_file(), f'Missing: {relative}'
paths = []
for folder in ['reports', 'selected_artifacts', checkpoint]:
    paths.extend(p for p in (root / folder).rglob('*') if p.is_file())
for name in ['status.json', 'runtime_history.jsonl', 'baseline_provenance.json']:
    if (root / name).is_file(): paths.append(root / name)
output = Path('/content/muril-resume.zip')
subprocess.run([sys.executable, '-m', 'pip', 'install', 'gdown==5.2.0'], check=True)
import gdown
source_zip = Path('/content/project-progress.zip')
if not source_zip.exists():
    gdown.download(id='11ueCWmhonSNsHO1FCjsvWBQscwdzNxbH', output=str(source_zip), quiet=False)
manifest = {}
with ZipFile(output, 'w', ZIP_STORED, allowZip64=True) as archive:
    for path in sorted(set(paths)):
        name = 'run/' + path.relative_to(root).as_posix()
        digest = hashlib.sha256()
        with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(block)
                target.write(block)
        manifest[name] = digest.hexdigest()
    with ZipFile(source_zip) as original:
        for task in ['task1_binary', 'task2_attribution', 'task3_codemix']:
            for split in ['train', 'val', 'test']:
                for relative in [f'{task}_{split}.jsonl', f'features/{task}_{split}.npz']:
                    matches = [i for i in original.infolist()
                               if i.filename.endswith('/data/clean_strict/' + relative)
                               and '__MACOSX' not in i.filename]
                    assert len(matches) == 1 and matches[0].file_size < 32*1024**2, relative
                    blob = original.read(matches[0])
                    name = 'data/clean_strict/' + relative
                    archive.writestr(name, blob)
                    manifest[name] = hashlib.sha256(blob).hexdigest()
    archive.writestr('transfer_manifest.json', json.dumps(manifest))
print(f'Ready: {output.stat().st_size / 1e9:.2f} GB. Original Drive files unchanged.')
files.download(str(output))
