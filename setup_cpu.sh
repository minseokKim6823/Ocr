#!/usr/bin/env bash
# ==============================================================================
# setup_cpu.sh — CPU / Ollama environment setup for ocr-gemma (Linux/Mac/WSL)
# ==============================================================================
# Usage: bash setup_cpu.sh
# Idempotent: safe to run multiple times.
# ==============================================================================

set -euo pipefail

VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

echo ""
echo "============================================================"
echo "  ocr-gemma  |  CPU / Ollama setup"
echo "============================================================"
echo ""

# ------------------------------------------------------------------------------
# 1. Create virtual environment if it does not already exist
# ------------------------------------------------------------------------------
if [ -f "$VENV_PY" ]; then
    echo "[1/5] Virtual environment already exists at $VENV_DIR — skipping creation."
else
    echo "[1/5] Creating virtual environment at $VENV_DIR ..."
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
echo "[2/5] Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip
echo "      Done."

# ------------------------------------------------------------------------------
# 3. Install CPU dependencies
# ------------------------------------------------------------------------------
echo ""
echo "[3/5] Installing requirements-cpu.txt ..."
"$VENV_PIP" install -r requirements-cpu.txt
echo "      Done."

# ------------------------------------------------------------------------------
# 4. Install the package in editable mode
# ------------------------------------------------------------------------------
echo ""
echo "[4/5] Installing ocr-gemma in editable mode (pip install -e .) ..."
"$VENV_PIP" install -e .
echo "      Done."

# ------------------------------------------------------------------------------
# 5. Ollama model pull
# ------------------------------------------------------------------------------
echo ""
echo "[5/5] Checking for Ollama CLI ..."

if command -v ollama &>/dev/null; then
    echo "      Ollama found at: $(command -v ollama)"
    echo ""
    echo "      Pulling gemma4:e2b (~3 GB) ..."
    ollama pull gemma4:e2b
    echo ""
    echo "      NOTE: For higher OCR accuracy (requires ~6 GB RAM):"
    echo "        ollama pull gemma4:e4b"
else
    echo ""
    echo "  [!] Ollama CLI not found on PATH."
    echo ""
    echo "  To complete setup:"
    echo "    1. Download and install Ollama from https://ollama.com/download"
    echo "    2. Restart your terminal (so PATH is refreshed)"
    echo "    3. Re-run this script:  bash setup_cpu.sh"
    echo "       -- or pull the model manually:"
    echo "          ollama pull gemma4:e2b"
    echo ""
fi

# ------------------------------------------------------------------------------
# Done — next steps
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete!  Next steps:"
echo "============================================================"
echo ""
echo "  Activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Run OCR on an image (CPU / Ollama):"
echo "    gemma-ocr --image sample.png --env cpu"
echo ""
echo "  Or via Python module:"
echo "    python -m ocr_gemma.cli --image sample.png --env cpu"
echo ""
echo "  Copy .env.example to .env and adjust settings as needed."
echo ""
