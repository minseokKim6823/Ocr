"""jpg/ 폴더의 모든 이미지를 CPU(Ollama)로 OCR하고 장당 소요 시간을 출력한다.

가장 빠르게 만드는 핵심:
  1. warmup()으로 모델을 RAM에 미리 상주 -> 첫 장의 콜드 로드(~90초)를 타이밍에서 제외
  2. for_cpu()가 num_threads = 전체 CPU 코어로 설정
  3. max_new_tokens를 문서 길이에 맞게 제한 (이 폼 문서는 512면 충분)
"""

from __future__ import annotations

import time
from pathlib import Path

from ocr_gemma import GemmaOCR, OCRConfig

# jpg 폴더 = 이 스크립트(examples/) 기준 상위의 jpg/
JPG_DIR = Path(__file__).resolve().parent.parent / "jpg"
# think=False라 모델이 EOS로 일찍 멈춤 -> 이 값은 잘림 방지 "상한"일 뿐,
# 높여도 속도 손해 없음(실제 생성량 = 문서 텍스트 길이).
MAX_NEW_TOKENS = 512

config = OCRConfig.for_cpu()
config.max_new_tokens = MAX_NEW_TOKENS
ocr = GemmaOCR(config)

print(f"backend = {ocr.backend_name}, threads = {config.num_threads}, "
      f"max_new_tokens = {config.max_new_tokens}")

# 모델 미리 로딩 (타이밍 제외)
print("warmup... (모델을 RAM에 적재)", flush=True)
t_warm = time.perf_counter()
ocr.warmup()
print(f"warmup 완료: {time.perf_counter() - t_warm:.1f}s\n", flush=True)

images = sorted(p for p in JPG_DIR.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg"})
if not images:
    raise SystemExit(f"{JPG_DIR} 에 jpg 파일이 없습니다.")

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
print(f"\n총 {len(images)}장, 합계 {total:.1f}s")
print(f"장당 평균 {total / len(durations):.2f}s "
      f"(최소 {min(durations):.2f}s / 최대 {max(durations):.2f}s)")
