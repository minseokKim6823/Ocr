"""High-level GemmaOCR API and backend dispatcher."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from ocr_gemma.config import OCRConfig
from ocr_gemma.backends.base import OCRBackend


def build_backend(config: OCRConfig) -> OCRBackend:
    """Instantiate the correct backend for *config*.

    Args:
        config: Populated :class:`~ocr_gemma.config.OCRConfig`.

    Returns:
        A ready-to-use :class:`~ocr_gemma.backends.base.OCRBackend` instance.

    Raises:
        ValueError: If ``config.backend`` is not a recognised value.
    """
    backend = config.backend.lower()
    if backend == "ollama":
        from ocr_gemma.backends.ollama_backend import OllamaBackend
        return OllamaBackend(config)
    if backend == "transformers":
        from ocr_gemma.backends.transformers_backend import TransformersBackend
        return TransformersBackend(config)
    if backend == "vllm":
        from ocr_gemma.backends.vllm_backend import VLLMBackend
        return VLLMBackend(config)
    raise ValueError(
        f"Unknown backend {config.backend!r}. "
        "Choose one of: 'ollama', 'transformers', 'vllm'."
    )


def _load_image(path_or_image) -> "Image.Image":  # noqa: F821
    """Load a PIL RGB image from a path string, Path, or PIL Image object.

    Args:
        path_or_image: File path (str or :class:`pathlib.Path`) or an
            already-loaded :class:`PIL.Image.Image`.

    Returns:
        RGB-mode :class:`PIL.Image.Image`.
    """
    from PIL import Image  # lazy but PIL is a hard dep for the whole package

    if isinstance(path_or_image, Image.Image):
        return path_or_image.convert("RGB")
    return Image.open(Path(path_or_image)).convert("RGB")


def _render_pdf_pages(path: Union[str, Path]) -> list:
    """Render each page of a PDF to a PIL RGB image.

    Tries ``pdf2image`` first, then ``fitz`` (PyMuPDF).  Raises a helpful
    :class:`RuntimeError` if neither is installed.

    Args:
        path: Path to the PDF file.

    Returns:
        List of PIL RGB images, one per page.
    """
    from PIL import Image

    path = Path(path)

    # -- Try pdf2image ---------------------------------------------------
    try:
        from pdf2image import convert_from_path  # type: ignore[import]
        pages = convert_from_path(str(path), dpi=200)
        return [p.convert("RGB") for p in pages]
    except ImportError:
        pass

    # -- Try PyMuPDF (fitz) ----------------------------------------------
    try:
        import fitz  # type: ignore[import]  # PyMuPDF

        doc = fitz.open(str(path))
        images = []
        for page in doc:
            # 200 DPI ≈ zoom factor 200/72
            zoom = 200 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
        doc.close()
        return images
    except ImportError:
        pass

    raise RuntimeError(
        "PDF rendering requires either 'pdf2image' or 'PyMuPDF'.\n"
        "Install one of them:\n"
        "  pip install pdf2image        # also needs poppler on PATH\n"
        "  pip install pymupdf          # pure-Python, no extra binaries"
    )


class GemmaOCR:
    """Main entry point for Gemma-4 OCR.

    Example::

        ocr = GemmaOCR(OCRConfig.for_cpu())
        text = ocr.read_image("scan.png")

    Args:
        config: Optional :class:`~ocr_gemma.config.OCRConfig`.
            Defaults to :meth:`OCRConfig.from_env`.
    """

    def __init__(self, config: Optional[OCRConfig] = None) -> None:
        if config is None:
            config = OCRConfig.from_env()
        self._config = config
        self._backend: OCRBackend = build_backend(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Human-readable identifier of the active backend."""
        return self._backend.name

    def warmup(self) -> None:
        """Pre-load model weights / warm caches (delegates to backend)."""
        self._backend.warmup()

    def read_image(
        self,
        path_or_image: Union[str, Path, "Image.Image"],  # noqa: F821
        prompt: Optional[str] = None,
    ) -> str:
        """Extract text from a single image.

        Args:
            path_or_image: File path (str / :class:`pathlib.Path`) or a
                pre-loaded :class:`PIL.Image.Image`.
            prompt: Optional instruction override for this call.

        Returns:
            Extracted text string.
        """
        image = _load_image(path_or_image)
        return self._backend.extract_text(image, prompt=prompt)

    def read_pdf(
        self,
        path: Union[str, Path],
        prompt: Optional[str] = None,
    ) -> List[str]:
        """Extract text from every page of a PDF, one string per page.

        Args:
            path: Path to the PDF file.
            prompt: Optional instruction override applied to every page.

        Returns:
            List of extracted text strings, one per page.
        """
        pages = _render_pdf_pages(path)
        return [self._backend.extract_text(page, prompt=prompt) for page in pages]

    def read_batch(
        self,
        paths: List[Union[str, Path, "Image.Image"]],  # noqa: F821
        prompt: Optional[str] = None,
    ) -> List[str]:
        """Extract text from multiple images in sequence.

        Args:
            paths: Iterable of file paths or PIL images.
            prompt: Optional instruction override applied to every item.

        Returns:
            List of extracted text strings in the same order as *paths*.
        """
        return [self.read_image(p, prompt=prompt) for p in paths]
