#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for a fresh WSL/Docker Ubuntu container.
# Installs system build tools + Python deps for cracked-inference.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Override this if you need a specific CUDA wheel index.
# Examples:
#   export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
#   export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "[bootstrap] installing apt packages..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    pkg-config \
    python3 \
    python3-venv \
    python3-pip \
    ninja-build
fi

echo "[bootstrap] creating virtual environment at $VENV_DIR ..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[bootstrap] upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "[bootstrap] installing PyTorch from: $TORCH_INDEX_URL"
python -m pip install --index-url "$TORCH_INDEX_URL" torch

echo "[bootstrap] installing project requirements..."
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo
echo "[bootstrap] done."
echo "Activate with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "Optional env for extension build cache:"
echo "  export TORCH_EXTENSIONS_DIR=\"$ROOT_DIR/.torch_extensions\""
