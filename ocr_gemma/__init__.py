"""ocr_gemma — Gemma-4 vision OCR package.

Importing this package does NOT trigger any import of torch, transformers,
vllm, or the ollama client.  Those are loaded lazily inside backend modules.

Quick start::

    from ocr_gemma import GemmaOCR, OCRConfig

    ocr = GemmaOCR(OCRConfig.for_cpu())
    text = ocr.read_image("scan.png")
    print(text)
"""

from ocr_gemma.config import DEFAULT_OCR_PROMPT, OCRConfig
from ocr_gemma.ocr import GemmaOCR, build_backend

__version__ = "0.1.0"

__all__ = [
    "GemmaOCR",
    "OCRConfig",
    "build_backend",
    "DEFAULT_OCR_PROMPT",
    "__version__",
]
