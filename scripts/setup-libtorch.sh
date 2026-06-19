#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv venv --python 3.12 "$ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" \
  "torch==2.11.0" \
  "torchvision==0.26.0" \
  "ultralytics==8.4.71"

"$ROOT/.venv/bin/python" - <<'PY'
import torch
print(f"PyTorch/LibTorch: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
PY

