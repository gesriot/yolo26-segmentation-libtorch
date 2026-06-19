#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "LibTorch environment is missing. Run scripts/setup-libtorch.sh first." >&2
  exit 1
fi

TORCH_LIB="$ROOT/.venv/lib/python3.12/site-packages/torch/lib"
export PATH="$ROOT/.venv/bin:$PATH"
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

exec cargo "$@"

