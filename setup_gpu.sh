#!/usr/bin/env bash
# ==============================================================================
# setup_gpu.sh — GPU / HuggingFace Transformers setup for ocr-gemma (Linux)
# ==============================================================================
# Usage: bash setup_gpu.sh
# Requires: an NVIDIA GPU with CUDA drivers installed (Linux).
# Idempotent: safe to run multiple times.
# ==============================================================================

set -euo pipefail

VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

echo ""
echo "============================================================"
echo "  ocr-gemma  |  GPU / HuggingFace Transformers setup"
echo "============================================================"
echo ""

# ------------------------------------------------------------------------------
# 1. Create virtual environment if it does not already exist
# ------------------------------------------------------------------------------
if [ -f "$VENV_PY" ]; then
    echo "[1/6] Virtual environment already exists at $VENV_DIR — skipping creation."
else
    echo "[1/6] Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "      Done."
fi

# Activate so subsequent pip/python calls use the venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------------------------
# 2. Upgrade pip
# ------------------------------------------------------------------------------
echo ""
echo "[2/6] Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip
echo "      Done."

# ------------------------------------------------------------------------------
# 3. Install CUDA-enabled PyTorch
# ------------------------------------------------------------------------------
echo ""
echo "[3/6] Installing CUDA-enabled PyTorch (cu124) ..."
echo "      NOTE: cu124 targets CUDA 12.4.  Adjust the index URL if your"
echo "      driver uses a different CUDA version (cu118, cu121, cu126, etc.)."
echo "      Check your version:  nvidia-smi  (top-right corner shows CUDA version)"
echo ""
"$VENV_PIP" install torch --index-url https://download.pytorch.org/whl/cu124
echo "      Done."

# ------------------------------------------------------------------------------
# 4. Install GPU dependencies
# ------------------------------------------------------------------------------
echo ""
echo "[4/6] Installing requirements-gpu.txt ..."
"$VENV_PIP" install -r requirements-gpu.txt
echo "      Done."

# ------------------------------------------------------------------------------
# 5. Install the package with GPU extras in editable mode
# ------------------------------------------------------------------------------
echo ""
echo "[5/6] Installing ocr-gemma[gpu] in editable mode ..."
"$VENV_PIP" install -e ".[gpu]"
echo "      Done."

# ------------------------------------------------------------------------------
# 6. Verify GPU availability
# ------------------------------------------------------------------------------
echo ""
echo "[6/6] Verifying GPU / CUDA availability ..."
echo ""
"$VENV_PY" -c "
import torch
avail = torch.cuda.is_available()
name  = torch.cuda.get_device_name(0) if avail else 'no-cuda'
print(f'  cuda_available={avail}  device={name}')
"
echo ""

# ------------------------------------------------------------------------------
# Optional vLLM
# ------------------------------------------------------------------------------
echo "============================================================"
echo "  Optional: vLLM high-throughput batch backend"
echo "============================================================"
echo ""
echo "  vLLM requires Linux + CUDA and is NOT installed by default."
echo "  To enable it:"
echo "    pip install -e .[vllm]"
echo "    -- or: pip install vllm>=0.6"
echo ""
echo "  When using vLLM, set in .env:"
echo "    OCR_BACKEND=vllm"
echo "    OCR_MODEL=google/gemma-4-e4b-it"
echo ""

# ------------------------------------------------------------------------------
# Done — next steps
# ------------------------------------------------------------------------------
echo "============================================================"
echo "  Setup complete!  Next steps:"
echo "============================================================"
echo ""
echo "  Activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Run OCR on an image (GPU / Transformers):"
echo "    gemma-ocr --image sample.png --env gpu"
echo ""
echo "  Or via Python module:"
echo "    python -m ocr_gemma.cli --image sample.png --env gpu"
echo ""
echo "  Copy .env.example to .env and set OCR_BACKEND=transformers,"
echo "  OCR_MODEL=google/gemma-4-e4b-it, OCR_DEVICE=auto."
echo ""
