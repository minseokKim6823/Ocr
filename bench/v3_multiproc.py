"""성능 벤치마크 변종 v3 = "배치 멀티프로세스 병렬화".

목적
----
여러 이미지를 ``multiprocessing.Pool`` 워커 프로세스로 나눠 동시 처리 시
**총 처리시간(throughput)** 이 줄어드는지 측정한다.

워커 모델 로드 구조
-------------------
엔진(RapidOCR)은 pickle 불가 -> 인자로 전달 불가.
각 워커 프로세스가 ``initializer=_init`` 에서 **1회** 엔진을 생성해 전역 변수
``_ENGINE`` 에 보관하고, 이후 ``_work(path)`` 가 재사용한다.

Windows spawn 주의
------------------
Windows 는 multiprocessing 기본 start method가 ``spawn`` 이다.
spawn 은 자식 프로세스가 스크립트 최상위 코드를 **재실행** 하므로
반드시 ``if __name__ == "__main__":`` 가드 안에서 Pool 을 생성해야 한다.
그렇지 않으면 워커 기동 시 Pool 이 재귀적으로 생성되어 프로세스 폭발이 일어난다.

RAM 경고
--------
워커 2개 기준: 각 RapidOCR(PPOCRv5 Korean) 인스턴스 ~1.3 GB -> 합산 **~2.6 GB**.
여유 RAM 이 충분하지 않으면 OOM 또는 스왑 폭주가 발생하므로
이 스크립트는 구문 검사 전용으로 작성되었다. **절대 실행 금지.**

실행(여유 RAM 충분 시만, 기본 워커=2):
    python bench/v3_multiproc.py
    python bench/v3_multiproc.py --workers 1   # 비교용 직렬 실행
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

# ------------------------------------------------------------------ 정답 (공백 제거 부분일치 검증용)
_GT: dict[str, tuple[str, str]] = {
    "1":    ("KR6475941E13", "디비닉스제사십일차1"),
    "2":    ("KR6475941E13", "디비닉스제사십일차1"),
    "IRN1": ("KR6ZW0001VN6", "한화스마트7486"),
    "IRN2": ("KR6475941E13", "디비닉스제사십일차1"),
    "IRN3": ("KR6KB0001YG0", "KB증권7533"),
    "IRN4": ("KR6KS0004W15", "한국투자증권9013"),
    "IRN5": ("KR6KS0004VY7", "한국투자증권9010"),
    "IRN6": ("KR6NH0003MT5", "NHNow257"),
}

# ------------------------------------------------------------------ 워커 코드
# _init 과 _work 는 최상위 레벨(모듈 스코프)에 정의해야 spawn 에서 pickle 가능.

_ENGINE = None  # 각 워커 프로세스 내 전역 엔진 (메인 프로세스에서는 사용 안 함)


def _init() -> None:
    """워커 프로세스 초기화: 엔진 1회 생성."""
    global _ENGINE
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, OCRVersion

    _ENGINE = RapidOCR(params={
        "Rec.lang_type": LangRec.KOREAN,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    })


def _work(path_str: str) -> tuple[str, float, str, str]:
    """워커 함수: 이미지 1장 OCR -> (path_str, dt, code, name).

    주의: 멀티프로세스 동시 실행이므로 개별 dt 는 CPU 경합 영향을 받는다.
    throughput 측정은 main 의 wall-clock 으로 판단해야 한다.
    """
    import time as _time
    import sys
    from pathlib import Path as _Path

    # vis_fields 는 프로젝트 루트에 있으므로 sys.path 에 추가 (spawn 시 cwd 불확실)
    proj_root = str(_Path(__file__).resolve().parent.parent)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    import vis_fields
    from rapidocr.utils.load_image import LoadImage

    img = LoadImage()(path_str)
    t0 = _time.perf_counter()
    found = vis_fields.extract_fields(
        _ENGINE, img, ["종목코드", "종목명"], 0.3, 0.5
    )
    dt = _time.perf_counter() - t0

    code = found.get("종목코드", ("", "", None))[1]
    nm   = found.get("종목명",   ("", "", None))[1]
    return (path_str, dt, code, nm)


# ------------------------------------------------------------------ 정답 대조 헬퍼

def _gt_ok(stem: str, code: str, nm: str) -> tuple[bool, bool]:
    """정답과 공백 제거 부분일치 여부 (code_ok, name_ok)."""
    gt = _GT.get(stem)
    if gt is None:
        return (False, False)
    gt_code, gt_nm = gt
    code_ok = gt_code in code.replace(" ", "")
    name_ok  = gt_nm  in nm.replace(  " ", "")
    return (code_ok, name_ok)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="v3: multiprocessing pool OCR bench")
    ap.add_argument("--workers", type=int, default=2,
                    help="Pool 워커 수 (기본 2; RAM ~1.3 GB/워커 주의)")
    ap.add_argument("--img-dir", default="jpg",
                    help="이미지 폴더 (기본 jpg/)")
    args = ap.parse_args()

    K = max(1, args.workers)
    img_dir = Path(args.img_dir)

    # 대상 이미지 수집 (존재하는 것만)
    targets = ["1.jpg", "2.jpg",
               "IRN1.jpg", "IRN2.jpg", "IRN3.jpg",
               "IRN4.jpg", "IRN5.jpg", "IRN6.jpg"]
    paths = [str((img_dir / t).resolve())
             for t in targets
             if (img_dir / t).exists()]

    if not paths:
        print(f"[오류] {img_dir}/ 에서 대상 이미지를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    n = len(paths)
    print(f"[v3] workers={K}  처리={n}장")
    print(f"     RAM 경고: 워커{K}개 × ~1.3 GB ≈ {K * 1.3:.1f} GB 필요")
    print()

    with Pool(processes=K, initializer=_init) as pool:
        t0 = perf_counter()
        results = pool.map(_work, paths)
        wall = perf_counter() - t0

    # 결과 출력
    # 개별 dt 는 프로세스 경합 영향을 받으므로 참고용
    okc = okn = 0
    for path_str, dt, code, nm in results:
        stem = Path(path_str).stem
        name = Path(path_str).name
        # 동시 실행이라 개별 t 는 CPU 경합 영향 있음 — throughput 은 wall-clock 으로 판단
        print(f"[{name}] t={dt:.2f}s 종목코드={code!r} 종목명={nm!r}")
        c_ok, n_ok = _gt_ok(stem, code, nm)
        if c_ok:
            okc += 1
        if n_ok:
            okn += 1

    print()
    print(
        f"POOL workers={K} total_wall={wall:.1f}s "
        f"throughput_avg={wall / n:.2f}s/img | "
        f"종목코드 {okc}/{n} 종목명 {okn}/{n}"
    )


if __name__ == "__main__":
    main()
