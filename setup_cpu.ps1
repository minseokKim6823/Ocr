#Requires -Version 5.1
# ==============================================================================
# setup_cpu.ps1 — CPU / Ollama environment setup for ocr-gemma (Windows)
# ==============================================================================
# Usage: .\setup_cpu.ps1
# Idempotent: safe to run multiple times.
# ==============================================================================

$ErrorActionPreference = "Stop"

$VenvDir   = ".\.venv"
$VenvPy    = "$VenvDir\Scripts\python.exe"
$VenvPip   = "$VenvDir\Scripts\pip.exe"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ocr-gemma  |  CPU / Ollama setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------------------------
# 1. Create virtual environment if it does not already exist
# ------------------------------------------------------------------------------
if (Test-Path $VenvPy) {
    Write-Host "[1/5] Virtual environment already exists at $VenvDir — skipping creation." -ForegroundColor Green
} else {
    Write-Host "[1/5] Creating virtual environment at $VenvDir with Python 3.12 ..." -ForegroundColor Yellow
    py -3.12 -m venv $VenvDir
    Write-Host "      Done." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# 2. Upgrade pip
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/5] Upgrading pip ..." -ForegroundColor Yellow
& $VenvPy -m pip install --upgrade pip
Write-Host "      Done." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 3. Install CPU dependencies
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/5] Installing requirements-cpu.txt ..." -ForegroundColor Yellow
& $VenvPip install -r requirements-cpu.txt
Write-Host "      Done." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 4. Install the package in editable mode
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/5] Installing ocr-gemma in editable mode (pip install -e .) ..." -ForegroundColor Yellow
& $VenvPip install -e .
Write-Host "      Done." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 5. Ollama model pull
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/5] Checking for Ollama CLI ..." -ForegroundColor Yellow

$OllamaCmd = Get-Command ollama -ErrorAction SilentlyContinue

if ($OllamaCmd) {
    Write-Host "      Ollama found at: $($OllamaCmd.Source)" -ForegroundColor Green
    Write-Host ""
    Write-Host "      Pulling gemma4:e2b (~3 GB) ..." -ForegroundColor Yellow
    ollama pull gemma4:e2b
    Write-Host ""
    Write-Host "      NOTE: For higher OCR accuracy (requires ~6 GB RAM):" -ForegroundColor Cyan
    Write-Host "        ollama pull gemma4:e4b" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "  [!] Ollama CLI not found on PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "  To complete setup:" -ForegroundColor Yellow
    Write-Host "    1. Download and install Ollama from https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "    2. Restart your terminal (so PATH is refreshed)" -ForegroundColor Yellow
    Write-Host "    3. Re-run this script:  .\setup_cpu.ps1" -ForegroundColor Yellow
    Write-Host "       -- or pull the model manually:" -ForegroundColor Yellow
    Write-Host "          ollama pull gemma4:e2b" -ForegroundColor Yellow
    Write-Host ""
}

# ------------------------------------------------------------------------------
# Done — next steps
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!  Next steps:" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Activate the environment:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "  Run OCR on an image (CPU / Ollama):" -ForegroundColor White
Write-Host "    gemma-ocr --image sample.png --env cpu" -ForegroundColor Green
Write-Host ""
Write-Host "  Or via Python module:" -ForegroundColor White
Write-Host "    python -m ocr_gemma.cli --image sample.png --env cpu" -ForegroundColor Green
Write-Host ""
Write-Host "  Copy .env.example to .env and adjust settings as needed." -ForegroundColor White
Write-Host ""
