"""Ollama HTTP backend for GemmaOCR.

All heavy imports (``ollama``, ``requests``) are deferred to method bodies so
that importing this module never fails on a machine without those packages.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

from ocr_gemma.backends.base import OCRBackend

if TYPE_CHECKING:
    from PIL import Image


class OllamaBackend(OCRBackend):
    """Talk to an Ollama server to perform OCR via a Gemma-4 vision model.

    The backend first tries the ``ollama`` Python package for structured
    communication.  If that package is unavailable it falls back to raw
    ``requests`` POSTs to the Ollama chat API endpoint.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _image_to_b64(self, image: "Image.Image") -> str:
        """Encode a PIL image as a base64 PNG string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _resolve_prompt(self, prompt: str | None) -> str:
        return prompt if prompt is not None else self.config.prompt

    def _options(self) -> dict:
        """Build the Ollama options dict from the current config."""
        opts: dict = {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_new_tokens,
        }
        if self.config.num_threads is not None:
            opts["num_thread"] = self.config.num_threads
        return opts

    # ------------------------------------------------------------------
    # OCRBackend interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"ollama:{self.config.model}"

    def warmup(self) -> None:
        """Load the model into RAM ahead of time so the first ``extract_text``
        call does not pay the (~minute on CPU) cold-load cost.

        Sends an empty-prompt generate request with ``keep_alive=-1`` so the
        model stays resident until Ollama is restarted. Best-effort: any
        failure here is swallowed and surfaces later on the real call with a
        clearer message.
        """
        try:
            import ollama  # type: ignore[import]

            client = ollama.Client(host=self.config.host)
            client.generate(model=self.config.model, prompt="", keep_alive=-1)
            return
        except ImportError:
            pass
        except Exception:
            pass  # fall back to requests / let the real call report it

        try:
            import requests  # type: ignore[import]

            requests.post(
                f"{self.config.host.rstrip('/')}/api/generate",
                json={"model": self.config.model, "keep_alive": -1},
                timeout=300,
            )
        except Exception:
            pass

    def extract_text(self, image: "Image.Image", prompt: str | None = None) -> str:
        """Send *image* to Ollama and return the extracted text.

        Args:
            image: RGB PIL image.
            prompt: Optional override for the extraction prompt.

        Returns:
            Plain-text OCR result from the model.

        Raises:
            RuntimeError: If Ollama is not reachable or neither ``ollama``
                nor ``requests`` is installed.
        """
        text_prompt = self._resolve_prompt(prompt)
        b64 = self._image_to_b64(image)

        # -- Try the official ollama package first -----------------------
        try:
            import ollama  # type: ignore[import]

            client = ollama.Client(host=self.config.host)
            response = client.chat(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": text_prompt,
                        "images": [b64],
                    }
                ],
                options=self._options(),
                think=False,  # OCR wants the answer, not a reasoning trace
            )
            return response["message"]["content"].strip()

        except ImportError:
            pass  # fall back to requests
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed ({exc}).\n"
                "Make sure Ollama is running and the model is pulled:\n"
                f"  ollama pull {self.config.model}"
            ) from exc

        # -- Fallback: raw requests POST ---------------------------------
        try:
            import requests  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "Neither the 'ollama' package nor 'requests' is installed.\n"
                "Install one of them:  pip install ollama   OR   pip install requests"
            ) from exc

        url = f"{self.config.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": text_prompt,
                    "images": [b64],
                }
            ],
            "options": self._options(),
            "stream": False,
            "think": False,  # OCR wants the answer, not a reasoning trace
        }

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.config.host}.\n"
                "Start Ollama and make sure the model is pulled:\n"
                f"  ollama pull {self.config.model}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
