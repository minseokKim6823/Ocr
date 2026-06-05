"""HuggingFace Transformers backend for GemmaOCR.

All heavy imports (``torch``, ``transformers``) are deferred to ``__init__``
so that importing this module never fails on a machine without a GPU stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_gemma.backends.base import OCRBackend

if TYPE_CHECKING:
    from PIL import Image


class TransformersBackend(OCRBackend):
    """Run Gemma-4 vision inference locally via HuggingFace Transformers.

    Model weights and processor are loaded once during ``__init__`` and
    reused across :meth:`extract_text` calls.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        # -- Lazy imports ------------------------------------------------
        try:
            import torch  # type: ignore[import]
            from transformers import (  # type: ignore[import]
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' and 'torch' packages are required for the "
                "transformers backend.\n"
                "Install them:  pip install torch transformers accelerate"
            ) from exc

        # -- Resolve dtype -----------------------------------------------
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(config.dtype, "auto")

        # -- Resolve device_map ------------------------------------------
        if config.device == "cpu":
            device_map = "cpu"
        elif config.device == "cuda":
            device_map = "cuda"
        else:
            device_map = "auto"

        # -- Set CPU thread count ----------------------------------------
        if config.device == "cpu" and config.num_threads is not None:
            torch.set_num_threads(config.num_threads)

        # -- Load model and processor ------------------------------------
        self._processor = AutoProcessor.from_pretrained(config.model)
        self._model = AutoModelForImageTextToText.from_pretrained(
            config.model,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self._model.eval()
        self._torch = torch

    # ------------------------------------------------------------------
    # OCRBackend interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"transformers:{self.config.model}"

    def extract_text(self, image: "Image.Image", prompt: str | None = None) -> str:
        """Run OCR on *image* using the locally loaded Transformers model.

        Args:
            image: RGB PIL image.
            prompt: Optional override for the extraction prompt.

        Returns:
            Stripped plain-text OCR result.
        """
        text_prompt = prompt if prompt is not None else self.config.prompt

        # Canonical Gemma-3/4 multimodal pattern: embed the PIL image directly
        # in the message content so apply_chat_template emits the correct number
        # of image-placeholder tokens AND the matching pixel features in a single
        # pass. Splitting text/image into two processor calls desynchronises the
        # placeholder count from the pixel tensor and breaks generation.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def warmup(self) -> None:
        """No-op warmup; model is already loaded in ``__init__``."""
