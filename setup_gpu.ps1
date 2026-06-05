#Requires -Version 5.1
# ==============================================================================
# setup_gpu.ps1 — GPU / HuggingFace Transformers setup for ocr-gemma (Windows)
# ==============================================================================
# Usage: .\setup_gpu.ps1
# Requires: an NVIDIA GPU with a CUDA-compatible driver installed.
# Idempotent: safe to run multiple times.
# ==============================================================================

$ErrorActionPreference = "Stop"

$VenvDir   = ".\.venv"
$VenvPy    = "$VenvDir\Scripts\python.exe"
$VenvPip   = "$VenvDir\Scripts\pip.exe"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ocr-gemma  |  GPU / HuggingFace Transformers setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------------------------
# 1. Create virtual environment if it does not already exist
# ------------------------------------------------------------------------------
if (Test-Path $VenvPy) {
    Write-Host "[1/6] Virtual environment already exists at $VenvDir — skipping creation." -ForegroundColor Green
} else {
    Write-Host "[1/6] Creating virtual environment at $VenvDir with Python 3.12 ..." -ForegroundColor Yellow
    py -3.12 -m venv $VenvDir
    Write-Host "      Done." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# 2. Upgrade pip
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/6] Upgrading pip ..." -ForegroundColor Yellow
& $VenvPy -m pip install --upgrade pip
Write-Host "      Done." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 3. Install CUDA-enabled PyTorch
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/6] Installing CUDA-enabled PyTorch (cu124) ..." -ForegroundColor Yellow
Write-Host "      NOTE: cu124 targets CUDA 12.4.  Adjust the index URL if your" -ForegroundColor Cyan
Write-Host "      driver uses a different CUDA version (cu118, cu121, cu126, etc.)." -ForegroundColor Cyan
Write-Host "      Check your version:  nvidia-smi  (top-right corner shows CUDA version)" -ForegroundColor Cyan
Write-Host ""
& $VenvPip install torch --index-url https://download.pytorch.org/whl/cu124
Write-Host "      Done." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 4. Install GPU dependencies
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/6] Installing requirements-gpu.txt ..." -ForegroundColor Yellow
& $VenvPip install -r requirements-gpu.txt
Write-Host "      Done." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 5. Install the package with GPU extras in editable mode
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/6] Installing ocr-gemma[gpu] in editable mode ..." -ForegroundColor Yellow
& $VenvPip install -e ".[gpu]"
Write-Host "      Done." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 6. Verify GPU availability
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[6/6] Verifying GPU / CUDA availability ..." -ForegroundColor Yellow
Write-Host ""
& $VenvPy -c "import torch; avail = torch.cuda.is_available(); name = torch.cuda.get_device_name(0) if avail else 'no-cuda'; print(f'  cuda_available={avail}  device={name}')"
Write-Host ""

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [!] PyTorch CUDA check failed.  Verify your NVIDIA driver and CUDA version." -ForegroundColor Red
}

# ------------------------------------------------------------------------------
# vLLM note
# ------------------------------------------------------------------------------
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Optional: vLLM high-throughput batch backend" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  vLLM is Linux + CUDA focused and NOT installed by default." -ForegroundColor White
Write-Host "  To enable it (on a Linux machine with CUDA):" -ForegroundColor White
Write-Host "    pip install -e .[vllm]" -ForegroundColor Green
Write-Host "    -- or: pip install vllm>=0.6" -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------------------
# Done — next steps
# ------------------------------------------------------------------------------
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!  Next steps:" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Activate the environment:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "  Run OCR on an image (GPU / Transformers):" -ForegroundColor White
Write-Host "    gemma-ocr --image sample.png --env gpu" -ForegroundColor Green
Write-Host ""
Write-Host "  Or via Python module:" -ForegroundColor White
Write-Host "    python -m ocr_gemma.cli --image sample.png --env gpu" -ForegroundColor Green
Write-Host ""
Write-Host "  Copy .env.example to .env and set OCR_BACKEND=transformers," -ForegroundColor White
Write-Host "  OCR_MODEL=google/gemma-4-e4b-it, OCR_DEVICE=auto." -ForegroundColor White
Write-Host ""
