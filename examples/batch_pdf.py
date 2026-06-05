"""batch_pdf.py — PDF 전체 페이지 OCR 배치 처리 예제

PDF 파일을 받아 페이지별로 OCR을 수행하고,
각 페이지의 텍스트를 out/page_NN.txt 파일로 저장합니다.

사용법:
    python examples/batch_pdf.py [PDF경로]

    PDF 경로를 생략하면 OCR_PDF 환경변수를 읽고,
    없으면 기본값 "sample.pdf"를 사용합니다.
    환경(cpu/gpu)은 OCR_ENV 환경변수로 지정합니다 (기본: cpu).

예:
    python examples/batch_pdf.py report.pdf
    OCR_ENV=gpu python examples/batch_pdf.py scanned_book.pdf
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> None:
    # ------------------------------------------------------------------
    # 1. 인자 파싱
    # ------------------------------------------------------------------
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OCR_PDF", "sample.pdf"))
    env = os.environ.get("OCR_ENV", "cpu")

    # ------------------------------------------------------------------
    # 2. PDF 파일 존재 여부 확인
    # ------------------------------------------------------------------
    if not pdf_path.exists():
        print(f"[오류] PDF 파일을 찾을 수 없습니다: {pdf_path}")
        print()
        print("샘플 PDF를 준비하는 방법:")
        print("  - 인식할 PDF를 'sample.pdf' 이름으로 프로젝트 루트에 복사하세요.")
        print("  - 또는 경로를 직접 지정하세요: python examples/batch_pdf.py path/to/doc.pdf")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. 출력 디렉토리 준비
    # ------------------------------------------------------------------
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    print(f"[설정] PDF: {pdf_path}  /  환경: {env}  /  출력 디렉토리: {out_dir}/")

    # ------------------------------------------------------------------
    # 4. OCR 인스턴스 생성
    # ------------------------------------------------------------------
    from ocr_gemma import GemmaOCR, OCRConfig

    config = OCRConfig.for_env(env)
    ocr = GemmaOCR(config)
    print(f"[백엔드] {ocr.backend_name}")

    # ------------------------------------------------------------------
    # 5. PDF 전체 페이지 OCR (read_pdf -> list[str], 페이지 순서대로)
    # ------------------------------------------------------------------
    print(f"[OCR] PDF 인식 시작...")
    total_start = time.perf_counter()

    pages = ocr.read_pdf(str(pdf_path))

    total_elapsed = time.perf_counter() - total_start
    total_chars = sum(len(p) for p in pages)

    # ------------------------------------------------------------------
    # 6. 페이지별 파일 저장 및 타이밍 출력
    # ------------------------------------------------------------------
    print()
    print(f"{'페이지':>6}  {'문자 수':>8}  {'출력 파일'}")
    print("-" * 50)

    for i, page_text in enumerate(pages, start=1):
        out_file = out_dir / f"page_{i:02d}.txt"
        out_file.write_text(page_text, encoding="utf-8")

        print(f"{i:>6}  {len(page_text):>8}자  {out_file}")

    # ------------------------------------------------------------------
    # 7. 전체 요약
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print(f"[완료] 총 {len(pages)}페이지 처리")
    print(f"       총 소요 시간 : {total_elapsed:.2f}초")
    print(f"       총 문자 수   : {total_chars}자")
    print(f"       평균 속도    : {total_chars / total_elapsed:.1f} 자/초")
    print(f"       결과 위치    : {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
