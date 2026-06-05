"""성능 벤치마크 변종 v2 = "det 1회 + 타깃 crop만 batch rec".

목적
----
문서 전체(~29줄)를 인식하지 않고, **검출(det)만 1회** 수행한 뒤
필요한 행(종목코드/종목명이 있는 줄)의 박스만 잘라서 **batch 인식(rec)** 해
문서 전체 인식 대비 더 빠른지 측정한다.

내부 API (설치된 rapidocr 소스로 시그니처 확인 완료)
----------------------------------------------------
- 이미지 로드: ``LoadImage()(path)`` -> BGR np.ndarray.
- 검출만:    ``engine(img, use_cls=False, use_rec=False, box_thresh=..., unclip_ratio=...)``
             -> ``use_rec=False`` 이면 ``TextDetOutput`` 반환, ``.boxes`` 는
             np.ndarray (N,4,2) 원본좌표(없으면 None).
- crop:      ``get_rotate_crop_image(img, np.array(box, dtype="float32"))`` (box 4x2).
- batch rec: ``engine.text_rec(TextRecInput(img=[crop0, crop1, ...], return_word_box=False))``
             -> ``TextRecOutput``; ``.txts`` 는 입력 crop 순서와 정렬된 tuple[str],
             ``.scores`` 는 list[float].

방어적 처리
-----------
설치 버전에 따라 ``TextRecInput`` / ``get_rotate_crop_image`` / ``engine.text_rec``
시그니처가 다를 수 있으므로 getattr/예외로 감싼다. 실패 시 해당 이미지는
종목코드='' , 종목명='' 로 기록하고 다음 이미지로 계속한다.

주의: 이 스크립트는 OCR(검출/인식) 을 실제로 실행한다. 호출자는 메모리/타이밍
오염을 피하기 위해 단독으로만 실행할 것.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rapidocr import RapidOCR
from rapidocr.ch_ppocr_rec import TextRecInput
from rapidocr.utils.load_image import LoadImage
from rapidocr.utils.process_img import get_rotate_crop_image
from rapidocr.utils.typings import LangRec, OCRVersion

# ----------------------------------------------------------------- 설정/상수
IMG_DIR = Path(__file__).resolve().parent.parent / "jpg"
IMAGES = [n for n in ["1.jpg", "2.jpg", "IRN1.jpg", "IRN2.jpg",
                      "IRN3.jpg", "IRN4.jpg", "IRN5.jpg", "IRN6.jpg"]
          if (IMG_DIR / n).exists()]  # 삭제된 파일 제외(존재하는 것만)
CODE_LABEL = "종목코드"
NAME_LABEL = "종목명"
BOX_THRESH = 0.3
UNCLIP_RATIO = 1.6

# 정답(공백 제거 부분일치로 채점). (종목코드, 종목명)
TRUTH = {
    "1.jpg":   ("KR6475941E13", "디비닉스제사십일차1(사모/콜)"),
    "2.jpg":   ("KR6475941E13", "디비닉스제사십일차1(사모/콜)"),
    "IRN2.jpg": ("KR6475941E13", "디비닉스제사십일차1(사모/콜)"),
    "IRN1.jpg": ("KR6ZW0001VN6", "한화스마트7486(ELS)"),
    "IRN3.jpg": ("KR6KB0001YG0", "KB증권7533(ELS)"),
    "IRN4.jpg": ("KR6KS0004W15", "한국투자증권9013(사모/ELS)"),
    "IRN5.jpg": ("KR6KS0004VY7", "한국투자증권9010(사모/ELS)"),
    "IRN6.jpg": ("KR6NH0003MT5", "NHNow257(공모/ELS)"),
}

_LOADER = LoadImage()


# ------------------------------------------------------------- geometry 헬퍼
def box_x0(box: np.ndarray) -> float:
    return float(box[:, 0].min())


def box_x1(box: np.ndarray) -> float:
    return float(box[:, 0].max())


def box_cy(box: np.ndarray) -> float:
    return float((box[:, 1].min() + box[:, 1].max()) / 2)


def box_h(box: np.ndarray) -> float:
    return float(box[:, 1].max() - box[:, 1].min())


def norm(s: str) -> str:
    return (s or "").replace(" ", "")


# ------------------------------------------------------------------- rec 배치
def batch_rec(engine, crops: list[np.ndarray]) -> list[str]:
    """crop 리스트를 한 번에 인식해 입력 순서대로 정렬된 텍스트 리스트 반환.

    방어적: text_rec / TextRecInput 시그니처가 다르면 예외를 올려보내고,
    호출부(per-image try)에서 그 이미지를 실패 처리한다.
    """
    if not crops:
        return []
    text_rec = getattr(engine, "text_rec", None)
    if text_rec is None:
        raise AttributeError("engine.text_rec 없음 - rapidocr 버전 불일치")
    rec_out = text_rec(TextRecInput(img=crops, return_word_box=False))
    txts = getattr(rec_out, "txts", None)
    if txts is None:
        return ["" for _ in crops]
    return [str(t) for t in txts]


def crop_box(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    return get_rotate_crop_image(img, np.array(box, dtype="float32"))


# ------------------------------------------------------------- 장당 추출 로직
def extract_one(engine, img: np.ndarray) -> tuple[str, str]:
    """det 1회 + 타깃 행 crop batch rec 로 (종목코드, 종목명) 추출.

    실패/미발견 시 빈 문자열을 반환한다. (예외는 호출부에서 잡음)
    """
    # 1) det -> boxes
    det = engine(img, use_cls=False, use_rec=False,
                 box_thresh=BOX_THRESH, unclip_ratio=UNCLIP_RATIO)
    boxes = getattr(det, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "", ""
    boxes = [np.asarray(b, dtype="float32") for b in boxes]
    W = img.shape[1]

    # 2) 왼쪽 라벨 후보(x0 < 0.25*W) -> batch rec -> "종목코드" 라벨 박스 찾기
    left_idx = [i for i, b in enumerate(boxes) if box_x0(b) < 0.25 * W]
    if not left_idx:
        return "", ""
    left_crops = [crop_box(img, boxes[i]) for i in left_idx]
    left_txts = batch_rec(engine, left_crops)

    code_label_box = None
    for i, txt in zip(left_idx, left_txts):
        if CODE_LABEL in norm(txt):
            code_label_box = boxes[i]
            break
    if code_label_box is None:
        return "", ""  # 코드 라벨 못 찾음 -> 실패

    row_y = box_cy(code_label_box)
    row_h = box_h(code_label_box)
    label_x1 = box_x1(code_label_box)

    # 3) 행 박스 = |세로중심 - row_y| <= row_h*0.6
    row_idx = [i for i, b in enumerate(boxes)
               if abs(box_cy(b) - row_y) <= row_h * 0.6]
    if not row_idx:
        return "", ""
    row_crops = [crop_box(img, boxes[i]) for i in row_idx]
    row_txts = batch_rec(engine, row_crops)
    # 박스별 정보: (box, text)
    row_items = [(boxes[i], row_txts[j]) for j, i in enumerate(row_idx)]

    # 4) 종목코드 값 = x0 >= 코드라벨 x1, 가장 가까운(min x0) 박스의 텍스트
    code_cands = [(b, t) for (b, t) in row_items if box_x0(b) >= label_x1]
    code_val = ""
    code_val_x1 = None
    if code_cands:
        cb, ct = min(code_cands, key=lambda it: box_x0(it[0]))
        code_val = ct.strip()
        code_val_x1 = box_x1(cb)

    # 5) 종목명 값: (a) 라벨-텍스트 매칭 우선 -> (b) 위치 폴백.
    name_val = ""
    # (a) 행 박스 중 "종목명"(공백 제거) 포함 박스. 병합형이면 접두 제거,
    #     단독 라벨이면 오른쪽 최근접. -> 제목/잡티(IRN2 'S') 오선택 방지.
    for b, t in row_items:
        nt = norm(t)
        pos = nt.find(NAME_LABEL)
        if pos < 0:
            continue
        if nt == NAME_LABEL:  # 단독 라벨 -> 오른쪽 최근접
            rights = [(bb, tt) for (bb, tt) in row_items
                      if bb is not b and box_x0(bb) >= box_x1(b) - 2]
            if rights:
                name_val = norm(min(rights, key=lambda it: box_x0(it[0]))[1])
        else:  # 병합형 -> 라벨 접두 제거(공백 정규화로 IRN4 'W' 케이스 해결)
            name_val = nt[pos + len(NAME_LABEL):]
        break
    # (b) 위치 폴백: 코드값 오른쪽, 세로도장 제외, 길이>=2('S' 잡티 제외), 가장 오른쪽.
    if not name_val and code_val_x1 is not None:
        cands = [(b, t) for (b, t) in row_items
                 if box_x0(b) > code_val_x1 and box_h(b) <= row_h * 2.2
                 and len(norm(t)) >= 2]
        if cands:
            nb, nt = max(cands, key=lambda it: box_x1(it[0]))
            nt = norm(nt)
            if nt.startswith(NAME_LABEL):
                nt = nt[len(NAME_LABEL):]
            name_val = nt

    return code_val, name_val


# ------------------------------------------------------------------- 채점
def hit(value: str, truth: str) -> bool:
    """공백 제거 부분일치(정답이 인식값에 포함되거나 그 반대)."""
    v, t = norm(value), norm(truth)
    if not v or not t:
        return False
    return t in v or v in t


# ------------------------------------------------------------------- main
def main() -> None:
    # 모델 1회 생성(타이밍 제외)
    engine = RapidOCR(params={
        "Rec.lang_type": LangRec.KOREAN,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    })

    total = 0.0
    okc = okn = 0
    n = len(IMAGES)
    for name in IMAGES:
        path = IMG_DIR / name
        code = nm = ""
        t0 = time.perf_counter()
        try:
            img = _LOADER(str(path))
            code, nm = extract_one(engine, img)
        except Exception as exc:  # 방어적: 이 이미지만 실패 처리하고 계속
            code, nm = "", ""
            print(f"[{name}] 처리 실패: {type(exc).__name__}: {exc}")
        sec = time.perf_counter() - t0
        total += sec

        truth = TRUTH.get(name, ("", ""))
        if hit(code, truth[0]):
            okc += 1
        if hit(nm, truth[1]):
            okn += 1
        print(f"[{name}] t={sec:.2f}s 종목코드={code!r} 종목명={nm!r}")

    avg = total / n if n else 0.0
    print(f"TOTAL {total:.1f}s avg {avg:.2f}s | "
          f"종목코드 {okc}/{n} 종목명 {okn}/{n}")


if __name__ == "__main__":
    main()
