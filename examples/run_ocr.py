"""run_ocr.py — 단일 이미지 OCR 예제

사용법:
    python examples/run_ocr.py [이미지경로]

    이미지 경로를 생략하면 OCR_IMAGE 환경변수를 읽고,
    그것도 없으면 기본값 "sample.png"를 사용합니다.
    환경(cpu/gpu)은 OCR_ENV 환경변수로 지정합니다 (기본: cpu).

예:
    python examples/run_ocr.py invoice.png
    OCR_ENV=gpu python examples/run_ocr.py scan.jpg
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> None:
    # ------------------------------------------------------------------
    # 1. 인자 파싱: CLI 위치 인자 우선, 없으면 환경변수, 없으면 기본값
    # ------------------------------------------------------------------
    image_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OCR_IMAGE", "sample.png"))
    print("image_path :  ",image_path)
    env = os.environ.get("OCR_ENV", "cpu")  # "cpu" | "gpu" | "auto"

    # ------------------------------------------------------------------
    # 2. 이미지 파일 존재 여부 확인 — 없으면 친절한 안내 후 종료
    # ------------------------------------------------------------------
    if not image_path.exists():
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {image_path}")
        print()
        print("샘플 이미지를 준비하는 방법:")
        print("  - 인식할 이미지를 'sample.png' 이름으로 이 프로젝트 루트에 복사하세요.")
        print("  - 또는 인수로 경로를 직접 지정하세요: python examples/run_ocr.py path/to/image.png")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. OCRConfig 및 GemmaOCR 인스턴스 생성
    # ------------------------------------------------------------------
    from ocr_gemma import GemmaOCR, OCRConfig

    print(f"[설정] 환경: {env}  /  이미지: {image_path}")

    config = OCRConfig.for_env(env)
    ocr = GemmaOCR(config)

    print(f"[백엔드] {ocr.backend_name}")

    # ------------------------------------------------------------------
    # 4. 모델 예열 (첫 추론의 콜드-스타트 지연을 줄임)
    # ------------------------------------------------------------------
    print("[예열] 모델 워밍업 중...")
    warmup_start = time.perf_counter()
    ocr.warmup()
    warmup_elapsed = time.perf_counter() - warmup_start
    print(f"[예열] 완료 ({warmup_elapsed:.2f}초)")

    # ------------------------------------------------------------------
    # 5. 이미지 OCR 실행 및 타이밍 측정
    # ------------------------------------------------------------------
    print(f"[OCR] '{image_path}' 인식 중...")
    start = time.perf_counter()
    text = ocr.read_image(str(image_path))
    elapsed = time.perf_counter() - start

    # ------------------------------------------------------------------
    # 6. 결과 출력
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("인식 결과")
    print("=" * 60)
    print(text)
    print("=" * 60)
    print()
    print(f"[완료] 소요 시간: {elapsed:.3f}초  /  문자 수: {len(text)}자  /  속도: {len(text)/elapsed:.1f} 자/초")


if __name__ == "__main__":
    main()
