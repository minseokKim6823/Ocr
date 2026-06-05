"""jpg/ 폴더의 각 문서에서 '종목코드'와 '종목명'만 빠르게 추출한다.

필요한 두 필드만 뽑도록 프롬프트를 좁혀 출력 토큰을 최소화 -> 전체 OCR보다 훨씬 빠름.
모델은 e2b(기본), think=False(추론 끄기)는 백엔드에 이미 적용됨.

사용법:
    python examples/extract_fields.py            # e2b
    python examples/extract_fields.py gemma4:e4b # 다른 모델
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from ocr_gemma import GemmaOCR, OCRConfig

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b"
JPG_DIR = Path(__file__).resolve().parent.parent / "jpg"

# 두 필드만 요구하는 좁은 프롬프트. 출력 형식을 고정해 파싱을 쉽게 한다.
PROMPT = (
    "이 문서 이미지에서 '종목코드'와 '종목명' 값만 찾아 정확히 아래 형식으로만 출력하라. "
    "설명이나 다른 항목은 절대 출력하지 마라.\n"
    "종목코드: <값>\n"
    "종목명: <값>"
)

cfg = OCRConfig.for_cpu()
cfg.model = MODEL
cfg.max_new_tokens = 96  # 두 줄이면 충분 -> 빠름
ocr = GemmaOCR(cfg)

print(f"=== MODEL: {ocr.backend_name}, max_new_tokens={cfg.max_new_tokens} ===")
print("warmup...", flush=True)
ocr.warmup()


def parse(text: str) -> tuple[str, str]:
    code = name = ""
    for line in text.splitlines():
        if "종목코드" in line:
            code = line.split(":", 1)[-1].strip() if ":" in line else line.replace("종목코드", "").strip()
        elif "종목명" in line:
            name = line.split(":", 1)[-1].strip() if ":" in line else line.replace("종목명", "").strip()
    # 코드만 본문에 박혀 있으면 정규식 백업 (KR + 영숫자 10자)
    if not code:
        m = re.search(r"KR[0-9A-Z]{8,12}", text)
        if m:
            code = m.group(0)
    return code, name


images = sorted(p for p in JPG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
rows: list[tuple[str, str, str, float]] = []
for i, path in enumerate(images, 1):
    t0 = time.perf_counter()
    raw = ocr.read_image(path, prompt=PROMPT)
    dt = time.perf_counter() - t0
    code, name = parse(raw)
    rows.append((path.name, code, name, dt))
    print(f"[{i}/{len(images)}] {path.name}  ({dt:.1f}s)\n  종목코드: {code}\n  종목명: {name}",
          flush=True)

times = [r[3] for r in rows]
print(f"\n총 {len(rows)}장, 합계 {sum(times):.1f}s, 장당 평균 {sum(times)/len(times):.1f}s")
print("\n=== 표 ===")
print(f"{'파일':<46} {'종목코드':<14} 종목명")
for name_f, code, name, _ in rows:
    print(f"{name_f:<46} {code:<14} {name}")
