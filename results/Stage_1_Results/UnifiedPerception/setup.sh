#!/usr/bin/env bash
# setup.sh — build a FULLY self-contained environment for UnifiedPerception.
#
# Everything lives inside this folder:
#   ./.venv         the virtual environment (never reuses a sibling repo's venv)
#   ./.cache/hf     HuggingFace cache (DINOv2 weights)
#   ./.cache/torch  torch.hub cache
#   ./.cache/ultralytics  YOLO settings
#   ./third_party   vendored detection (sentry/) + action (models/) code
#   ./assets        vendored checkpoints (adapter, yolo26s, k400 head, classes)
#
# Usage:
#   bash setup.sh
#   PYTHON_BIN=python3.12 bash setup.sh        # pick the interpreter
#   SKIP_PREDOWNLOAD=1 bash setup.sh           # don't pre-fetch DINOv2 weights
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"

echo "[setup] root:   $ROOT"
echo "[setup] python: $($PYTHON_BIN --version 2>&1) ($PYTHON_BIN)"

# 1. Local virtual environment (created with the stdlib venv module only).
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv -> $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
else
  echo "[setup] reusing existing venv -> $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 2. Dependencies (pinned) + the package itself (editable).
python -m pip install --upgrade pip wheel setuptools
# Install the CUDA 12.4 build of torch/torchvision explicitly from PyTorch's own
# index first (this is the reliable way to get the +cu124 wheels); then the rest.
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$ROOT/requirements.txt"
python -m pip install -e "$ROOT"

# 3. Local caches so nothing is written outside this folder.
export HF_HOME="$ROOT/.cache/hf"
export TORCH_HOME="$ROOT/.cache/torch"
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$YOLO_CONFIG_DIR"

# 4. Pre-fetch the DINOv2 backbone weights into the local HF cache so later runs
#    are fully offline. Skip with SKIP_PREDOWNLOAD=1.
if [ "${SKIP_PREDOWNLOAD:-0}" != "1" ]; then
  echo "[setup] pre-downloading facebook/dinov2-small into local HF cache"
  python - <<'PY'
from transformers import AutoModel
AutoModel.from_pretrained("facebook/dinov2-small")
print("[setup] dinov2-small cached")
PY
fi

echo
echo "[setup] done."
echo "  activate:  source \"$VENV/bin/activate\""
echo "  run:       python -m unified_perception --source clip.mp4 --output-dir out"
echo "  (640 default; add --image-size 644 for the clean DINOv2 patch grid)"
