"""Configuration dataclass for GemmaOCR backends."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_OCR_PROMPT = (
    "Extract ALL text from this image exactly as it appears, preserving "
    "reading order, line breaks, tables, and layout. Output only the text, "
    "no commentary."
)


@dataclass
class OCRConfig:
    """Runtime configuration for a GemmaOCR session.

    Attributes:
        backend: Which inference backend to use ("ollama", "transformers", "vllm").
        model: Ollama model tag or HuggingFace model id.
        device: Compute device ("cpu", "cuda", "auto").
        num_threads: CPU thread count; None means let the runtime decide.
        max_new_tokens: Maximum tokens the model may generate per image.
        temperature: Sampling temperature (0.0 = greedy / deterministic).
        dtype: Floating-point precision ("auto", "bfloat16", "float16").
        host: Base URL of the Ollama HTTP server.
        prompt: System instruction sent with every image.
    """

    backend: str = "ollama"                   # "ollama" | "transformers" | "vllm"
    model: str = "gemma4:e4b"                 # ollama tag OR hf model id
    device: str = "auto"                      # "cpu" | "cuda" | "auto"
    num_threads: Optional[int] = None         # CPU threads (None = auto)
    max_new_tokens: int = 2048
    temperature: float = 0.0
    dtype: str = "auto"                       # "auto" | "bfloat16" | "float16"
    host: str = "http://localhost:11434"      # ollama host
    prompt: str = DEFAULT_OCR_PROMPT

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_cpu(cls) -> "OCRConfig":
        """Return a config tuned for CPU-only inference via Ollama.

        Uses the smallest Gemma-4 variant (e2b) and sets thread count to
        the number of logical CPU cores available on this machine.
        """
        import os as _os

        return cls(
            backend="ollama",
            model="gemma4:e2b",
            device="cpu",
            num_threads=_os.cpu_count(),
        )

    @classmethod
    def for_gpu(cls) -> "OCRConfig":
        """Return a config tuned for GPU inference via Transformers.

        Uses the HuggingFace ``google/gemma-4-e4b-it`` checkpoint with
        bfloat16 precision on CUDA.
        """
        return cls(
            backend="transformers",
            model="google/gemma-4-e4b-it",
            device="cuda",
            dtype="bfloat16",
        )

    @classmethod
    def from_env(cls) -> "OCRConfig":
        """Build a config from environment variables, falling back to defaults.

        Recognised variables:
            OCR_BACKEND, OCR_MODEL, OCR_DEVICE, OCR_NUM_THREADS,
            OCR_MAX_NEW_TOKENS, OCR_HOST, OCR_TEMPERATURE, OCR_DTYPE.
        """
        base = cls()

        def _get(var: str, default):
            return os.environ.get(var, default)

        backend = _get("OCR_BACKEND", base.backend)
        model = _get("OCR_MODEL", base.model)
        device = _get("OCR_DEVICE", base.device)
        host = _get("OCR_HOST", base.host)
        dtype = _get("OCR_DTYPE", base.dtype)

        raw_threads = os.environ.get("OCR_NUM_THREADS")
        num_threads = int(raw_threads) if raw_threads is not None else base.num_threads

        raw_tokens = os.environ.get("OCR_MAX_NEW_TOKENS")
        max_new_tokens = int(raw_tokens) if raw_tokens is not None else base.max_new_tokens

        raw_temp = os.environ.get("OCR_TEMPERATURE")
        temperature = float(raw_temp) if raw_temp is not None else base.temperature

        return cls(
            backend=backend,
            model=model,
            device=device,
            num_threads=num_threads,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            dtype=dtype,
            host=host,
            prompt=base.prompt,
        )

    @classmethod
    def for_env(cls, env: str) -> "OCRConfig":
        """Convenience dispatcher.

        Args:
            env: ``"cpu"`` → :meth:`for_cpu`, ``"gpu"`` → :meth:`for_gpu`,
                 anything else → :meth:`from_env`.
        """
        if env == "cpu":
            return cls.for_cpu()
        if env == "gpu":
            return cls.for_gpu()
        return cls.from_env()
