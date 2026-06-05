"""Abstract base class for OCR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class OCRBackend(ABC):
    """Protocol that every inference backend must satisfy.

    Subclasses must implement :meth:`extract_text` and :meth:`name`.
    The optional :meth:`warmup` hook lets callers pre-load model weights
    before processing the first image.
    """

    def __init__(self, config) -> None:  # config: OCRConfig (avoid circular import)
        self.config = config

    @abstractmethod
    def extract_text(self, image: "Image.Image", prompt: str | None = None) -> str:
        """Run OCR on a single PIL image and return extracted text.

        Args:
            image: RGB PIL image to process.
            prompt: Override the default extraction prompt for this call.

        Returns:
            Plain-text string of everything recognised in the image.
        """

    def warmup(self) -> None:
        """Pre-load model weights / warm JIT caches.

        Default implementation is a no-op; backends may override.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable identifier for this backend instance."""
