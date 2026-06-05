"""Backend sub-package for ocr_gemma.

Importing this module does NOT import torch, transformers, vllm, or ollama.
Those are loaded lazily inside each backend class.
"""

from ocr_gemma.backends.base import OCRBackend
from ocr_gemma.backends.ollama_backend import OllamaBackend
from ocr_gemma.backends.transformers_backend import TransformersBackend
from ocr_gemma.backends.vllm_backend import VLLMBackend

__all__ = [
    "OCRBackend",
    "OllamaBackend",
    "TransformersBackend",
    "VLLMBackend",
]
