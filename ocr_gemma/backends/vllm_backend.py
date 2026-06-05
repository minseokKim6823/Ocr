"""vLLM backend for GemmaOCR (GPU batch inference).

All imports of ``vllm`` are deferred to ``__init__`` so that this module
can be imported on machines without a GPU / vLLM installation.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

from ocr_gemma.backends.base import OCRBackend

if TYPE_CHECKING:
    from PIL import Image


class VLLMBackend(OCRBackend):
    """High-throughput Gemma-4 OCR via vLLM.

    Suitable for batch workloads on multi-GPU servers.  A single
    :class:`vllm.LLM` instance is created during ``__init__`` and reused
    across all :meth:`extract_text` calls.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        try:
            from vllm import LLM, SamplingParams  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The 'vllm' package is required for the vLLM backend.\n"
                "It must be installed in a GPU environment:\n"
                "  pip install vllm"
            ) from exc

        self._llm = LLM(model=config.model)
        self._SamplingParams = SamplingParams

    # ------------------------------------------------------------------
    # OCRBackend interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"vllm:{self.config.model}"

    def extract_text(self, image: "Image.Image", prompt: str | None = None) -> str:
        """Run OCR on *image* via vLLM.

        Args:
            image: RGB PIL image.
            prompt: Optional override for the extraction prompt.

        Returns:
            Stripped plain-text OCR result.
        """
        text_prompt = prompt if prompt is not None else self.config.prompt

        sampling_params = self._SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        # Encode image to base64 PNG for the multimodal payload
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        outputs = self._llm.chat(messages=[messages], sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()

    def warmup(self) -> None:
        """No-op; vLLM loads the model during ``__init__``."""
