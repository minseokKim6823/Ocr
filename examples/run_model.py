"""지정한 Ollama 모델로 jpg/ 전체를 OCR하고 장당 시간과 결과를 출력한다.

사용법:
    python examples/run_model.py gemma4:e2b
    python examples/run_model.py gemma4:e4b

think=False(추론 끄기)로 답만 받아 빠르고, EOS로 일찍 멈춘다.
모델 비교용: 두 모델을 따로 돌려 출력 텍스트와 장당 시간을 비교한다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ocr_gemma import GemmaOCR, OCRConfig

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b"
JPG_DIR = Path(__file__).resolve().parent.parent / "jpg"
MAX_NEW_TOKENS = 512

cfg = OCRConfig.for_cpu()
cfg.model = MODEL
cfg.max_new_tokens = MAX_NEW_TOKENS
ocr = GemmaOCR(cfg)

print(f"=== MODEL: {ocr.backend_name} (threads={cfg.num_threads}, "
      f"max_new_tokens={cfg.max_new_tokens}) ===")

print("warmup...", flush=True)
t_warm = time.perf_counter()
ocr.warmup()
print(f"warmup: {time.perf_counter() - t_warm:.1f}s\n", flush=True)

images = sorted(p for p in JPG_DIR.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg"})

durations: list[float] = []
for i, path in enumerate(images, 1):
    t0 = time.perf_counter()
    text = ocr.read_image(path)
    dt = time.perf_counter() - t0
    durations.append(dt)
    print(f"[{i}/{len(images)}] {path.name}  ->  {dt:.2f}s, {len(text)} chars")
    print(text)
    print("-" * 60, flush=True)

total = sum(durations)
print(f"\n[{MODEL}] 총 {len(images)}장, 합계 {total:.1f}s")
print(f"[{MODEL}] 장당 평균 {total / len(durations):.2f}s "
      f"(최소 {min(durations):.2f}s / 최대 {max(durations):.2f}s)")
