# -*- coding: utf-8 -*-
"""벤치마크 변종 v4: int8 동적 양자화 rec 모델

목적
----
한국어 rec onnx(korean_PP-OCRv5_rec_mobile.onnx)를 onnxruntime.quantization으로
int8 동적 양자화한 뒤, **baseline과 동일한 설정(글자박스 ON, unclip 1.6→2.0)**에서
rec 모델만 교체해 양자화 효과(속도/정확도)를 분리 측정한다.

위험 사항 (주의)
---------------
Rec.model_path 오버라이드 시 한국어 사전(rec_keys / korean_char_dict.txt)이
자동으로 해결되는지 불확실하다.
- lang_type=KOREAN을 유지하므로 dict는 korean으로 잡힐 것으로 기대하나,
  인식 결과가 빈 문자열이거나 깨진 글자라면 dict 미해결일 수 있다.
- 그 경우 대비책: Rec.rec_keys_path 를 원본 모델 폴더의 dict 파일로
  명시적으로 설정해야 할 수 있다. 예:
    "Rec.rec_keys_path": ".venv/Lib/site-packages/rapidocr/models/<korean_dict>.txt"
  (지금은 model_path만 시도하고 결과로 판단)

실행:
    python bench/v4_int8.py

주의: OCR/양자화를 실제로 실행한다. 메모리/타이밍 오염을 피하기 위해
단독으로만 실행할 것.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# PYTHONIOENCODING=utf-8 보장
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- 상수/경로
# 원본 rec 모델 경로 (없으면 스크립트 종료)
SRC = r".venv\Lib\site-packages\rapidocr\models\korean_PP-OCRv5_rec_mobile.onnx"
# 양자화 결과 저장 경로 (raw string: 백슬래시 이스케이프 방지)
DST = r"bench\korean_rec_int8.onnx"

JPG_DIR = Path(__file__).resolve().parent.parent / "jpg"
IMAGE_NAMES = ["1.jpg", "2.jpg", "IRN1.jpg", "IRN2.jpg",
               "IRN3.jpg", "IRN4.jpg", "IRN5.jpg", "IRN6.jpg"]

FIELDS = ["종목코드", "종목명"]
BOX_THRESH = 0.3
TEXT_SCORE = 0.5

# 정답 (공백 제거 후 부분일치)
ANSWERS: dict[str, tuple[str, str]] = {
    "1.jpg":    ("KR6475941E13", "디비닉스제사십일차1(사모/콜)"),
    "2.jpg":    ("KR6475941E13", "디비닉스제사십일차1(사모/콜)"),
    "IRN1.jpg": ("KR6ZW0001VN6", "한화스마트7486(ELS)"),
    "IRN2.jpg": ("KR6475941E13", "디비닉스제사십일차1(사모/콜)"),
    "IRN3.jpg": ("KR6KB0001YG0", "KB증권7533(ELS)"),
    "IRN4.jpg": ("KR6KS0004W15", "한국투자증권9013(사모/ELS)"),
    "IRN5.jpg": ("KR6KS0004VY7", "한국투자증권9010(사모/ELS)"),
    "IRN6.jpg": ("KR6NH0003MT5", "NHNow257(공모/ELS)"),
}

# ---------------------------------------------------------------- import (지연)
# onnxruntime.quantization 은 양자화 단계에서만 사용; 무거운 import 경고 방지
from rapidocr import RapidOCR
from rapidocr.utils.load_image import LoadImage
from rapidocr.utils.typings import LangRec, OCRVersion
import vis_fields  # extract_fields(engine, img, fields, box_thresh, text_score)

_LOADER = LoadImage()


# ---------------------------------------------------------------- 채점
def norm(s: str) -> str:
    return (s or "").replace(" ", "")


def hit(value: str, truth: str) -> bool:
    """공백 제거 부분일치."""
    v, t = norm(value), norm(truth)
    if not v or not t:
        return False
    return t in v or v in t


# ---------------------------------------------------------------- 양자화
def ensure_quantized() -> None:
    """SRC -> DST int8 동적 양자화 (DST가 이미 있으면 건너뜀)."""
    if not os.path.exists(SRC):
        print(f"[오류] 원본 모델 없음: {SRC}")
        raise SystemExit(1)

    if os.path.exists(DST):
        src_size = os.path.getsize(SRC)
        dst_size = os.path.getsize(DST)
        print(f"[양자화] 이미 존재 -> 건너뜀")
        print(f"  원본 {src_size / 1024:.1f} KB  |  int8 {dst_size / 1024:.1f} KB"
              f"  ({dst_size / src_size * 100:.1f}%)")
        return

    # DST 디렉터리 생성
    os.makedirs(os.path.dirname(os.path.abspath(DST)), exist_ok=True)

    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"[양자화] 시작: {SRC} -> {DST}")
    t0 = time.perf_counter()
    quantize_dynamic(SRC, DST, weight_type=QuantType.QInt8)
    elapsed = time.perf_counter() - t0

    src_size = os.path.getsize(SRC)
    dst_size = os.path.getsize(DST)
    print(f"[양자화] 완료 {elapsed:.1f}s  "
          f"원본 {src_size / 1024:.1f} KB -> int8 {dst_size / 1024:.1f} KB"
          f"  ({dst_size / src_size * 100:.1f}%)")


# ---------------------------------------------------------------- main
def main() -> None:
    # 1) 양자화 (없으면 생성, 있으면 생략)
    ensure_quantized()

    # 2) 엔진 생성 (타이밍 제외)
    #    위험: Rec.model_path 오버라이드 시 rec_keys(사전)가 자동 해결되지 않을 수 있음.
    #    lang_type=KOREAN 유지로 dict는 korean으로 잡힐 것으로 기대.
    #    인식이 깨지면 Rec.rec_keys_path 를 명시적으로 설정할 것(위 docstring 참고).
    engine = RapidOCR(params={
        "Rec.lang_type": LangRec.KOREAN,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_path": os.path.abspath(DST),
    })

    # 3) 존재하는 이미지만 처리
    images = [n for n in IMAGE_NAMES if (JPG_DIR / n).exists()]
    n = len(images)
    if n == 0:
        print(f"[오류] {JPG_DIR} 에 대상 이미지 없음")
        raise SystemExit(1)
    if n < len(IMAGE_NAMES):
        missing = [nm for nm in IMAGE_NAMES if nm not in images]
        print(f"[경고] 없는 이미지 {len(missing)}장 건너뜀: {missing}")

    # 4) 루프
    total = 0.0
    okc = okn = 0
    for name in images:
        path = JPG_DIR / name
        code = nm = ""
        t0 = time.perf_counter()
        try:
            img = _LOADER(str(path))
            found = vis_fields.extract_fields(
                engine, img, FIELDS, BOX_THRESH, TEXT_SCORE
            )
            code = found.get("종목코드", (None, "", None))[1] if "종목코드" in found else ""
            nm   = found.get("종목명",   (None, "", None))[1] if "종목명"   in found else ""
        except Exception as exc:
            print(f"[{name}] 처리 실패: {type(exc).__name__}: {exc}")
        sec = time.perf_counter() - t0
        total += sec

        truth = ANSWERS.get(name, ("", ""))
        if hit(code, truth[0]):
            okc += 1
        if hit(nm, truth[1]):
            okn += 1
        print(f"[{name}] t={sec:.2f}s 종목코드={code!r} 종목명={nm!r}")

    avg = total / n if n else 0.0
    print(f"TOTAL {total:.1f}s avg {avg:.2f}s | 종목코드 {okc}/{n} 종목명 {okn}/{n}")


if __name__ == "__main__":
    main()
