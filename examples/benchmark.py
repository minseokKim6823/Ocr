"""benchmark.py — GemmaOCR 성능 벤치마크

지정한 이미지에 대해 N회 반복 추론을 수행하고,
최소/평균/중앙값 소요 시간, 처리 속도(chars/sec)를 마크다운 표로 출력합니다.

사용법:
    python examples/benchmark.py [옵션]

옵션:
    --image PATH      벤치마크할 이미지 경로 (기본: sample.png)
    --env {cpu,gpu}   환경 프리셋 (기본: cpu)
    --runs N          반복 횟수 (기본: 3)
    --warmup          첫 실행 전 모델 예열 (기본: 활성화)
    --no-warmup       모델 예열 건너뜀

예:
    python examples/benchmark.py --image invoice.png --env cpu --runs 5
    python examples/benchmark.py --image scan.png --env gpu --runs 10 --no-warmup
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GemmaOCR 성능 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default="sample.png",
        help="벤치마크할 이미지 파일 경로 (기본: sample.png)",
    )
    parser.add_argument(
        "--env",
        default="cpu",
        choices=["cpu", "gpu"],
        help="환경 프리셋: cpu (Ollama) 또는 gpu (Transformers) (기본: cpu)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="반복 추론 횟수 (기본: 3)",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="첫 추론 전 모델 예열 여부 (기본: --warmup)",
    )
    return parser.parse_args()


def fmt_seconds(value: float) -> str:
    """소수점 셋째 자리까지 초 단위로 포맷합니다."""
    return f"{value:.3f}s"


def print_markdown_table(rows: list[tuple[str, str]], title: str = "") -> None:
    """두 열짜리 마크다운 표를 출력합니다."""
    if title:
        print(f"\n### {title}\n")
    col_w = max(len(r[0]) for r in rows)
    val_w = max(len(r[1]) for r in rows)
    sep = f"| {'-' * col_w} | {'-' * val_w} |"
    print(f"| {'항목':<{col_w}} | {'값':<{val_w}} |")
    print(sep)
    for label, value in rows:
        print(f"| {label:<{col_w}} | {value:<{val_w}} |")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)

    # ------------------------------------------------------------------
    # 이미지 파일 존재 여부 확인
    # ------------------------------------------------------------------
    if not image_path.exists():
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {image_path}")
        print()
        print("샘플 이미지를 준비하는 방법:")
        print("  - 이미지를 'sample.png' 이름으로 프로젝트 루트에 복사하세요.")
        print("  - 또는 --image 옵션으로 경로를 지정하세요: --image path/to/image.png")
        sys.exit(1)

    if args.runs < 1:
        print("[오류] --runs 는 1 이상이어야 합니다.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # OCR 인스턴스 생성
    # ------------------------------------------------------------------
    from ocr_gemma import GemmaOCR, OCRConfig

    config = OCRConfig.for_env(args.env)
    ocr = GemmaOCR(config)

    print(f"[벤치마크] 이미지: {image_path}  /  백엔드: {ocr.backend_name}  /  반복: {args.runs}회")

    # ------------------------------------------------------------------
    # 모델 예열 (선택)
    # ------------------------------------------------------------------
    if args.warmup:
        print("[예열] 모델 워밍업 중...")
        w_start = time.perf_counter()
        ocr.warmup()
        w_elapsed = time.perf_counter() - w_start
        print(f"[예열] 완료 ({w_elapsed:.3f}초)")
    else:
        print("[예열] 건너뜀 (--no-warmup)")

    # ------------------------------------------------------------------
    # 반복 추론 및 타이밍 수집
    # ------------------------------------------------------------------
    timings: list[float] = []
    char_counts: list[int] = []

    print()
    print(f"{'실행':>4}  {'소요(초)':>10}  {'문자 수':>8}")
    print("-" * 30)

    for run in range(1, args.runs + 1):
        t0 = time.perf_counter()
        result = ocr.read_image(str(image_path))
        elapsed = time.perf_counter() - t0

        timings.append(elapsed)
        char_counts.append(len(result))
        print(f"{run:>4}  {elapsed:>10.3f}  {len(result):>8}자")

    # ------------------------------------------------------------------
    # 통계 계산
    # ------------------------------------------------------------------
    t_min = min(timings)
    t_mean = statistics.mean(timings)
    t_median = statistics.median(timings)
    c_mean = statistics.mean(char_counts)
    chars_per_sec = c_mean / t_mean if t_mean > 0 else 0.0

    # ------------------------------------------------------------------
    # 마크다운 표 출력
    # ------------------------------------------------------------------
    rows = [
        ("백엔드", ocr.backend_name),
        ("이미지", str(image_path)),
        ("반복 횟수", str(args.runs)),
        ("최소 시간", fmt_seconds(t_min)),
        ("평균 시간", fmt_seconds(t_mean)),
        ("중앙값 시간", fmt_seconds(t_median)),
        ("평균 문자 수", f"{c_mean:.0f}자"),
        ("처리 속도", f"{chars_per_sec:.1f} 자/초"),
    ]
    print_markdown_table(rows, title="벤치마크 결과")
    print()


if __name__ == "__main__":
    main()
