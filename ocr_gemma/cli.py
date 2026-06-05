"""Command-line interface for GemmaOCR.

Runnable as::

    python -m ocr_gemma.cli --image scan.png
    gemma-ocr --pdf document.pdf --env cpu --output result.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gemma-ocr",
        description="Extract text from images and PDFs using Gemma-4 vision models.",
    )

    # -- Input -----------------------------------------------------------
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", metavar="PATH", help="Path to an image file.")
    input_group.add_argument("--pdf", metavar="PATH", help="Path to a PDF file.")

    # -- Environment / backend ------------------------------------------
    p.add_argument(
        "--env",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help=(
            "Preset environment: 'cpu' (Ollama/e2b), 'gpu' (Transformers/e4b), "
            "'auto' (read from env vars). Default: auto."
        ),
    )
    p.add_argument("--backend", choices=["ollama", "transformers", "vllm"],
                   help="Override the inference backend.")
    p.add_argument("--model", metavar="NAME",
                   help="Override the model tag or HuggingFace model id.")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"],
                   help="Override compute device.")
    p.add_argument("--prompt", metavar="TEXT",
                   help="Override the OCR extraction prompt.")
    p.add_argument("--max-new-tokens", type=int, metavar="N",
                   help="Maximum tokens to generate per image.")
    p.add_argument("--threads", type=int, metavar="N",
                   help="Number of CPU threads (Ollama / Transformers CPU).")
    p.add_argument("--host", metavar="URL",
                   help="Ollama server base URL (default: http://localhost:11434).")

    # -- Output ----------------------------------------------------------
    p.add_argument("--output", metavar="PATH",
                   help="Write extracted text to this file instead of stdout.")
    p.add_argument("--json", action="store_true",
                   help="Emit output as JSON (list of page strings for PDF, "
                        "single-element list for images).")

    # -- Extras ----------------------------------------------------------
    p.add_argument("--warmup", action="store_true",
                   help="Call backend warmup before processing.")
    p.add_argument("--benchmark", action="store_true",
                   help="Print elapsed seconds and chars/sec to stderr after processing.")

    return p


def main() -> None:
    """Entry point for the ``gemma-ocr`` console script."""
    parser = _build_parser()
    args = parser.parse_args()

    # -- Build config ----------------------------------------------------
    try:
        from ocr_gemma.config import OCRConfig
    except ImportError as exc:
        print(f"ERROR: Cannot import ocr_gemma: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        config = OCRConfig.for_env(args.env)

        # Apply explicit flag overrides
        if args.backend:
            config.backend = args.backend
        if args.model:
            config.model = args.model
        if args.device:
            config.device = args.device
        if args.prompt:
            config.prompt = args.prompt
        if args.max_new_tokens is not None:
            config.max_new_tokens = args.max_new_tokens
        if args.threads is not None:
            config.num_threads = args.threads
        if args.host:
            config.host = args.host

    except Exception as exc:
        print(f"ERROR building config: {exc}", file=sys.stderr)
        sys.exit(1)

    # -- Instantiate OCR -------------------------------------------------
    try:
        from ocr_gemma.ocr import GemmaOCR
        ocr = GemmaOCR(config)
    except Exception as exc:
        print(f"ERROR initialising backend: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Backend: {ocr.backend_name}", file=sys.stderr)

    if args.warmup:
        print("Warming up…", file=sys.stderr)
        ocr.warmup()

    # -- Run OCR ---------------------------------------------------------
    t0 = time.perf_counter()
    try:
        if args.image:
            results: list[str] = [ocr.read_image(args.image, prompt=args.prompt)]
        else:
            results = ocr.read_pdf(args.pdf, prompt=args.prompt)
    except Exception as exc:
        print(f"ERROR during OCR: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t0

    # -- Benchmark report ------------------------------------------------
    if args.benchmark:
        total_chars = sum(len(r) for r in results)
        cps = total_chars / elapsed if elapsed > 0 else float("inf")
        print(
            f"Benchmark: {elapsed:.2f}s elapsed, {total_chars} chars, "
            f"{cps:.0f} chars/sec",
            file=sys.stderr,
        )

    # -- Format output ---------------------------------------------------
    if args.json:
        output_text = json.dumps(results, ensure_ascii=False, indent=2)
    else:
        output_text = "\n\n--- page break ---\n\n".join(results)

    # -- Write or print --------------------------------------------------
    if args.output:
        try:
            Path(args.output).write_text(output_text, encoding="utf-8")
            print(f"Written to {args.output}", file=sys.stderr)
        except OSError as exc:
            print(f"ERROR writing output file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
