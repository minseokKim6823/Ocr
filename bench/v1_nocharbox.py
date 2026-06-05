# -*- coding: utf-8 -*-
"""벤치마크 변종 v1: 글자박스 끄기 (region-level only)

return_word_box=False, return_single_char_box=False 로 인식해서
글자단위 박스 계산 비용을 없애고 더 빠른지 측정한다.

값 위치는 region(줄) 박스만으로 찾는다.

실행:
    python bench/v1_nocharbox.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# PYTHONIOENCODING=utf-8 보장
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from rapidocr import RapidOCR
from rapidocr.utils.load_image import LoadImage
from rapidocr.utils.typings import LangRec, OCRVersion

# ---------------------------------------------------------------- 설정
JPG_DIR = Path(__file__).resolve().parent.parent / "jpg"
IMAGE_NAMES = [n for n in ["1.jpg", "2.jpg", "IRN1.jpg", "IRN2.jpg",
                           "IRN3.jpg", "IRN4.jpg", "IRN5.jpg", "IRN6.jpg"]
               if (JPG_DIR / n).exists()]  # 삭제된 파일 제외(존재하는 것만)
UNCLIPS = (1.6, 2.0)
CODE_LABEL = "종목코드"
NAME_LABEL = "종목명"

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

_LOADER = LoadImage()


# ---------------------------------------------------------------- geometry
def cy(bb: list[float]) -> float:
    return (bb[1] + bb[3]) / 2


def height(bb: list[float]) -> float:
    return bb[3] - bb[1]


def box_to_xyxy(b) -> list[float]:
    """N×4×2 배열의 한 박스(shape 4×2)를 [x0,y0,x1,y1]로 변환."""
    import numpy as np
    arr = np.asarray(b, dtype=float)   # shape (4, 2)
    return [arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()]


# ---------------------------------------------------------------- OCR 결과 -> region 목록
def build_regions(res) -> list[dict]:
    """res.boxes, res.txts, res.scores 로 region 목록 구성.

    boxes: np.ndarray shape (N, 4, 2) 또는 list of (4,2)-like
    txts:  tuple[str] / list[str], 길이 N
    반환:  [{"text": str, "bbox": [x0,y0,x1,y1]}, ...]
    """
    regions: list[dict] = []
    boxes = getattr(res, "boxes", None)
    txts  = getattr(res, "txts", None)
    if boxes is None or txts is None:
        return regions
    for i, (b, t) in enumerate(zip(boxes, txts)):
        if t is None:
            continue
        text = str(t).replace(" ", "")  # region 텍스트 공백 제거(글자박스 버전과 동일 토큰화)
        if not text:
            continue
        bbox = box_to_xyxy(b)
        regions.append({"text": text, "bbox": bbox})
    return regions


# ---------------------------------------------------------------- 라벨 탐색
def best_label_region(regions: list[dict], label: str):
    """label을 포함하는 region 중 가장 '라벨다운' 것 반환. 없으면 None.

    tier 0: 단독(text == label)
    tier 1: 라벨로 시작(라벨+값 한 박스, text가 label보다 길고 index==0)
    tier 2: 긴 줄 속 부분일치 -> 기각(제목 등)
    """
    cands = [(r, r["text"].find(label)) for r in regions if label in r["text"]]
    if not cands:
        return None

    def tier_of(r: dict, idx: int) -> int:
        return 0 if r["text"] == label else (1 if idx == 0 else 2)

    r, idx = min(cands, key=lambda c: (tier_of(*c), len(c[0]["text"]), c[0]["bbox"][0]))
    tier = tier_of(r, idx)
    if tier == 2:
        return None
    return r, tier


def right_regions_on_row(regions: list[dict], anchor_bbox: list[float],
                         exclude=None) -> list[dict]:
    """anchor 오른쪽 + 같은 줄 + 세로 도장 제외. x0 순 정렬."""
    row_y = cy(anchor_bbox)
    row_h = height(anchor_bbox)
    out = [
        r for r in regions
        if r is not exclude
        and r["bbox"][0] >= anchor_bbox[2] - 2
        and abs(cy(r["bbox"]) - row_y) <= max(row_h * 0.6, 12)
        and height(r["bbox"]) <= row_h * 2.2
    ]
    return sorted(out, key=lambda r: r["bbox"][0])


# ---------------------------------------------------------------- 비율 split (라벨+값 한 박스)
def ratio_split_value(region: dict, label: str) -> tuple[str, list[float]]:
    """region bbox에서 label 부분을 제외한 값 영역을 비율로 추정한다.

    x0_val = x0 + (x1 - x0) * (len(label) / len(text))
    text_val = text[len(label):]

    주의: 글자 폭이 균일하지 않으면 오차 발생 가능.
    """
    text = region["text"]
    bbox = region["bbox"]
    x0, y0, x1, y1 = bbox
    ratio = len(label) / max(len(text), 1)
    x0_val = x0 + (x1 - x0) * ratio
    val_text = text[len(label):]
    val_bbox = [x0_val, y0, x1, y1]
    return val_text, val_bbox


# ---------------------------------------------------------------- 종목코드 찾기
def find_code(regions: list[dict]) -> str | None:
    """종목코드 라벨 옆 값 텍스트 반환. 못 찾으면 None."""
    hit = best_label_region(regions, CODE_LABEL)
    if hit is None:
        return None
    r, tier = hit

    if tier == 1:
        # 라벨+값 한 박스 -> 비율 split
        val_text, _ = ratio_split_value(r, CODE_LABEL)
        val_text = val_text.strip()
        if val_text:
            return val_text

    # 단독 라벨(tier 0) 또는 split 값 없음 -> 오른쪽 최근접 region
    rights = right_regions_on_row(regions, r["bbox"], exclude=r)
    if not rights:
        return None
    return rights[0]["text"].strip() or None


# ---------------------------------------------------------------- 종목명 찾기
def find_name(regions: list[dict]) -> str | None:
    """종목명 값 텍스트 반환. 못 찾으면 None.

    1단계: 단독 라벨 -> 오른쪽 최근접 / 라벨+값 한 박스 -> 비율 split
    2단계(폴백): 종목코드 값 region 기준 같은 줄 오른쪽 + 세로 도장 제외 중
                 가장 오른쪽 region (text가 종목명으로 시작하면 비율 split)
    """
    # --- 1단계
    hit = best_label_region(regions, NAME_LABEL)
    if hit is not None:
        r, tier = hit
        if tier == 1:
            val_text, _ = ratio_split_value(r, NAME_LABEL)
            val_text = val_text.strip()
            if val_text:
                return val_text
        else:
            # 단독 라벨
            rights = right_regions_on_row(regions, r["bbox"], exclude=r)
            if rights:
                return rights[0]["text"].strip() or None

    # --- 2단계: 위치 폴백 (종목코드 값 기준)
    code_hit = best_label_region(regions, CODE_LABEL)
    if code_hit is None:
        return None
    code_r, code_tier = code_hit

    # 종목코드 값 region 구하기
    if code_tier == 1:
        _, code_val_bbox = ratio_split_value(code_r, CODE_LABEL)
        # 값 bbox를 가진 region을 직접 만들어 anchor로 사용
        code_val_region: dict = {"text": "", "bbox": code_val_bbox}
        anchor_bbox = code_val_bbox
    else:
        rights = right_regions_on_row(regions, code_r["bbox"], exclude=code_r)
        if not rights:
            return None
        code_val_region = rights[0]
        anchor_bbox = code_val_region["bbox"]

    # 종목코드 값 anchor 기준 오른쪽 region들(세로 도장 제외)
    fallback_rights = right_regions_on_row(regions, anchor_bbox, exclude=code_val_region)
    if not fallback_rights:
        return None

    # 가장 오른쪽 region
    v = max(fallback_rights, key=lambda o: o["bbox"][2])
    text = v["text"]
    if text.startswith(NAME_LABEL):
        text, _ = ratio_split_value(v, NAME_LABEL)
    return text.strip() or None


# ---------------------------------------------------------------- 장별 처리
def process_image(engine: RapidOCR, img) -> tuple[str | None, str | None]:
    """이미지 한 장 처리 -> (종목코드값, 종목명값). 못 찾으면 None."""
    code: str | None = None
    name: str | None = None

    for u in UNCLIPS:
        need_code = code is None
        need_name = name is None
        if not need_code and not need_name:
            break

        res = engine(
            img,
            use_cls=False,
            use_rec=True,
            return_word_box=False,
            return_single_char_box=False,
            box_thresh=0.3,
            text_score=0.5,
            unclip_ratio=u,
        )
        regions = build_regions(res)

        if need_code:
            code = find_code(regions)
        if need_name:
            name = find_name(regions)

    return code, name


# ---------------------------------------------------------------- 정답 판정
def is_match(got: str | None, expected: str) -> bool:
    """공백 제거 후 부분일치."""
    if got is None:
        return False
    return expected.replace(" ", "") in got.replace(" ", "")


# ---------------------------------------------------------------- main
def main() -> None:
    engine = RapidOCR(params={
        "Rec.lang_type": LangRec.KOREAN,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    })

    ok_code = 0
    ok_name = 0
    total_time = 0.0

    for name in IMAGE_NAMES:
        path = JPG_DIR / name
        img = _LOADER(str(path))

        t_start = time.perf_counter()
        code, nm = process_image(engine, img)
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        total_time += elapsed

        exp_code, exp_name = ANSWERS[name]
        if is_match(code, exp_code):
            ok_code += 1
        if is_match(nm, exp_name):
            ok_name += 1

        print(f"[{name}] t={elapsed:.2f}s 종목코드={code!r} 종목명={nm!r}")

    n = len(IMAGE_NAMES)
    avg = total_time / n if n else 0.0
    print(f"TOTAL {total_time:.1f}s avg {avg:.2f}s | "
          f"종목코드 {ok_code}/{n} 종목명 {ok_name}/{n}")


if __name__ == "__main__":
    main()
