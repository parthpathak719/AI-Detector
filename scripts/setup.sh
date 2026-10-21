#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.lock
HF_HUB_DISABLE_XET=1 .venv/bin/python -m scripts.download_muril
.venv/bin/python -m scripts.verify_artifacts
