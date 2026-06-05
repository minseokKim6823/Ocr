"""문서 이미지에서 '종목코드'·'종목명' 옆의 값만 빨간 박스로 표시한다. (PoC)

여러 장 한 번에:
    python vis_fields.py                      # jpg/ 폴더 전체
    python vis_fields.py jpg/IRN1.jpg --open  # 한 장 + 결과 띄움
    python vis_fields.py "jpg/IRN*.jpg"

동작:
  - 기본은 종목코드/종목명 -> 빠른 경로(det 1회 + 종목코드 행만 인식), 실패 시 full 폴백.
  - 값 찾기: 라벨 단독이면 같은 줄 오른쪽 박스, 라벨+값 한 박스면 라벨 접두 제거.
  - (종목명) 회색 라벨 인식이 불안정해서 '종목코드' 값 기준 오른쪽 박스로 폴백.
  - 양식별 최적 unclip이 달라 1.6 -> 2.0 순 재시도(못 찾은 필드만).
  - (--fields 로 다른 라벨도 지정 가능. '라벨 옆 값' 기본 규칙만 적용.)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.ch_ppocr_rec import TextRecInput
from rapidocr.utils.load_image import LoadImage
from rapidocr.utils.process_img import get_rotate_crop_image
from rapidocr.utils.typings import LangRec, OCRVersion

RED_BGR = (0, 0, 255)
DEFAULT_DIR = "jpg"
DEFAULT_FIELDS = "종목코드,종목명"
UNCLIPS = (1.6, 2.0)
CODE_LABEL = "종목코드"
NAME_LABEL = "종목명"
FAST_FIELDS = {"종목코드", "종목명"}  # 빠른 경로(det 1회+종목코드 행만 인식)가 커버하는 필드
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUT_SUFFIXES = ("_fields", "_redbox")
TAG = {"종목코드": "CODE", "종목명": "NAME"}  # 박스 위 영문 라벨(cv2 한글 불가). 없으면 F1,F2..
_LOADER = LoadImage()

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- geometry
def char_union_bbox(chars) -> list[float]:
    pts = np.concatenate(
        [np.array(b, dtype=float).reshape(-1, 2) for _t, b in chars], axis=0
    )
    return [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]


def cy(bb):
    return (bb[1] + bb[3]) / 2


def height(bb):
    return bb[3] - bb[1]


def build_lines(result) -> list[dict]:
    """word_results(글자 박스)로 줄 목록 구성 -> boxes와 인덱스 정합성 문제 회피."""
    lines = []
    for wl in (getattr(result, "word_results", None) or ()):
        if not wl:
            continue
        chars = [(str(t), b) for (t, _s, b) in wl if b is not None]
        if not chars:
            continue
        lines.append({
            "chars": chars,
            "text": "".join(t for t, _ in chars),
            "bbox": char_union_bbox(chars),
        })
    return lines


def best_label_line(lines: list[dict], label: str):
    """라벨을 포함한 줄 중 가장 '라벨다운' 것 선택. (ln, idx, tier) 또는 None.

    tier 0: 라벨 단독, 1: 라벨로 시작(라벨+값), 2: 긴 줄 속 부분일치(제목 등) -> 기각.
    """
    cands = [
        (ln, _label_pos(ln["text"], label))
        for ln in lines
        if _label_pos(ln["text"], label) >= 0
    ]
    if not cands:
        return None

    def tier_of(ln, idx):
        return 0 if _label_only(ln["text"], label) else (1 if idx == 0 else 2)

    ln, idx = min(cands, key=lambda c: (tier_of(*c), len(c[0]["text"]), c[0]["bbox"][0]))
    tier = tier_of(ln, idx)
    if tier == 2:  # 제목 등 긴 문장 속 부분일치 -> 신뢰 불가
        return None
    return ln, idx, tier


def right_boxes_on_row(lines, anchor_bbox, exclude=None):
    """anchor 오른쪽 + 같은 줄 + 세로 도장 제외. x순 정렬."""
    row_y, row_h = cy(anchor_bbox), height(anchor_bbox)
    out = [
        o for o in lines
        if o is not exclude
        and o["bbox"][0] >= anchor_bbox[2] - 2
        and abs(cy(o["bbox"]) - row_y) <= max(row_h * 0.6, 12)
        and height(o["bbox"]) <= row_h * 2.2
    ]
    return sorted(out, key=lambda o: o["bbox"][0])


def find_field(lines: list[dict], label: str):
    """라벨 옆 값 찾기. 라벨+값 한 박스면 글자박스로 값만, 단독이면 오른쪽 최근접 박스."""
    host = best_label_line(lines, label)
    if host is None:
        return None
    ln, idx, _tier = host
    after = ln["chars"][idx + len(label):]
    if after:  # 라벨+값 한 박스 -> 값 글자만
        return (label, "".join(t for t, _ in after), char_union_bbox(after))
    rights = right_boxes_on_row(lines, ln["bbox"], exclude=ln)
    if not rights:
        return None
    return (label, rights[0]["text"], rights[0]["bbox"])  # 단독 라벨 -> 오른쪽 최근접


def find_name_by_position(lines: list[dict]):
    """종목명 폴백: '종목코드' 값 기준 같은 줄 오른쪽 값 박스(세로 도장 제외)."""
    code = find_field(lines, CODE_LABEL)
    if code is None:
        return None
    rights = right_boxes_on_row(lines, code[2])
    if not rights:
        return None
    v = max(rights, key=lambda o: o["bbox"][2])  # 가장 오른쪽 = 종목명 값
    chars, text = v["chars"], v["text"]
    if text.startswith(NAME_LABEL):
        chars, text = chars[len(NAME_LABEL):], text[len(NAME_LABEL):]
    if not chars:
        return None
    return (NAME_LABEL, text, char_union_bbox(chars))


def locate(lines, field):
    """필드 -> (field, 값, bbox) 또는 None. (종목명은 위치 폴백 포함)"""
    r = find_field(lines, field)
    if r is None and field == NAME_LABEL:
        r = find_name_by_position(lines)
    if r:
        return (field, r[1], r[2])
    return None


def _norm(s: str) -> str:
    return (s or "").replace(" ", "")


def _code_label_variant_pos(text: str) -> int:
    """crop 후 '종목코드'의 첫 글자가 잘려 '목코드'처럼 읽히는 경우 보정."""
    t = _norm(text)
    for variant in ("종목코드", "목코드", "등목코드"):
        pos = t.find(variant)
        if pos >= 0 and len(t) <= len(CODE_LABEL) + 2:
            return pos
    return -1


def _label_pos(text: str, label: str) -> int:
    t = _norm(text)
    pos = t.find(label)
    if pos >= 0:
        return pos
    if label == CODE_LABEL:
        return _code_label_variant_pos(t)
    return -1


def _label_only(text: str, label: str) -> bool:
    t = _norm(text)
    if t == label:
        return True
    return label == CODE_LABEL and _code_label_variant_pos(t) >= 0


def extract_fields_full(engine, img, fields, box_thresh, text_score):
    found: dict[str, tuple] = {}
    for u in UNCLIPS:
        if all(K in found for K in fields):
            break
        res = engine(img, use_cls=False, use_rec=True,
                     return_word_box=True, return_single_char_box=True,
                     box_thresh=box_thresh, text_score=text_score, unclip_ratio=u)
        lines = build_lines(res)
        for K in fields:
            if K not in found:
                r = locate(lines, K)
                if r:
                    found[K] = r
    return found


# ------------------------------------------------- 빠른 경로(det 1회 + 종목코드 행만 인식)
def _xyxy(b) -> list[float]:
    a = np.asarray(b, dtype=float)
    return [float(a[:, 0].min()), float(a[:, 1].min()),
            float(a[:, 0].max()), float(a[:, 1].max())]

# 이미지 자르기
INPUT_CROP_TRIM_X = 0.02
INPUT_CROP_TRIM_Y = 0.10


def _crop_input_image(img):
    h, w = img.shape[:2]
    x0 = int(round(w * INPUT_CROP_TRIM_X))
    x1 = int(round(w * (1.0 - INPUT_CROP_TRIM_X)))
    y0 = int(round(h * INPUT_CROP_TRIM_Y))
    y1 = int(round(h * (1.0 - INPUT_CROP_TRIM_Y)))
    if x1 <= x0 or y1 <= y0:
        return img, 0, 0
    return img[y0:y1, x0:x1], x0, y0


def _offset_bbox(bb, dx, dy):
    return [bb[0] + dx, bb[1] + dy, bb[2] + dx, bb[3] + dy]


def _offset_found(found, dx, dy):
    if not found or (dx == 0 and dy == 0):
        return found
    return {
        k: (label, value, _offset_bbox(bb, dx, dy))
        for k, (label, value, bb) in found.items()
    }


def _batch_rec(engine, crops) -> list[str]:
    if not crops:
        return []
    out = engine.text_rec(TextRecInput(img=crops, return_word_box=False))
    txts = getattr(out, "txts", None) or [""] * len(crops)
    return [str(t) for t in txts]


def _candidate_label_boxes(bb, w):
    return [
        i for i, x in enumerate(bb)
        if x[0] <= 0.04 * w
        and 90 <= (x[2] - x[0]) <= 180
        and 22 <= height(x) <= 40
    ]


def _find_code_row_by_label(engine, img, boxes, bb, w):
    left = _candidate_label_boxes(bb, w)
    if not left:
        return None
    txts = _batch_rec(engine, [get_rotate_crop_image(img, boxes[i]) for i in left])
    for i, text in zip(left, txts):
        if _label_pos(text, CODE_LABEL) >= 0:
            return i
    return None


def _ratio_bbox(bb, text_norm, label):
    """병합형(라벨+값 한 박스) 값 bbox 근사: 라벨 글자수 비율만큼 왼쪽 잘라냄."""
    x0, y0, x1, y1 = bb
    r = len(label) / max(len(text_norm), 1)
    return [x0 + (x1 - x0) * r, y0, x1, y1]


def extract_fields_fast(engine, img, box_thresh):
    """det 1회 후 '종목코드' 행만 타깃 인식 -> 종목코드/종목명. 코드행 못 찾으면 None.

    값 텍스트 규칙은 full 경로와 동일(공백 정규화·라벨매칭 우선·잡티 제외).
    종목명 병합형 bbox는 비율 근사(값 텍스트는 정확).
    """
    det = engine(img, use_cls=False, use_rec=False,
                 box_thresh=box_thresh, unclip_ratio=UNCLIPS[0])
    boxes = getattr(det, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    boxes = [np.asarray(b, dtype="float32") for b in boxes]
    bb = [_xyxy(b) for b in boxes]
    w = img.shape[1]

    # 종목코드 행 찾기: 왼쪽 라벨열만 인식
    code_i = _find_code_row_by_label(engine, img, boxes, bb, w)
    if code_i is None:
        left = [i for i, x in enumerate(bb) if x[0] < 0.28 * w]
        if not left:
            return None
        left_txts = _batch_rec(engine, [get_rotate_crop_image(img, boxes[i]) for i in left])
        code_i = next((i for i, t in zip(left, left_txts)
                       if _label_pos(t, CODE_LABEL) >= 0), None)
    if code_i is None:
        return None

    row_y, row_h = cy(bb[code_i]), height(bb[code_i])
    row = [i for i in range(len(boxes))
           if abs(cy(bb[i]) - row_y) <= max(row_h * 0.9, 14)]
    row_txts = _batch_rec(engine, [get_rotate_crop_image(img, boxes[i]) for i in row])
    items = [(bb[i], row_txts[j].replace(" ", ""))
             for j, i in enumerate(row) if row_txts[j].strip()]

    found: dict[str, tuple] = {}
    label_x1 = bb[code_i][2]

    # 종목코드 값: 라벨 오른쪽 최근접 박스
    code_cands = [(x, t) for (x, t) in items if x[0] >= label_x1 - 2]
    if code_cands:
        cx, ct = min(code_cands, key=lambda it: it[0][0])
        if ct:
            found[CODE_LABEL] = (CODE_LABEL, ct, cx)
    code_x1 = found[CODE_LABEL][2][2] if CODE_LABEL in found else None

    # 종목명 값: (a) 라벨-텍스트 매칭 우선
    name = None
    for x, t in items:
        pos = t.find(NAME_LABEL)
        if pos < 0:
            continue
        if t == NAME_LABEL:  # 단독 라벨 -> 오른쪽 최근접
            rights = [(xx, tt) for (xx, tt) in items
                      if xx is not x and xx[0] >= x[2] - 2]
            if rights:
                nx, nt = min(rights, key=lambda it: it[0][0])
                name = (NAME_LABEL, nt, nx)
        else:  # 병합형 -> 라벨 접두 제거 + bbox 비율 추정
            name = (NAME_LABEL, t[pos + len(NAME_LABEL):], _ratio_bbox(x, t, NAME_LABEL))
        break
    # (b) 위치 폴백: 코드값 오른쪽, 세로도장 제외, 길이>=2('S' 잡티 제외), 가장 오른쪽
    if name is None and code_x1 is not None:
        cands = [(x, t) for (x, t) in items
                 if x[0] > code_x1 and height(x) <= row_h * 2.2 and len(t) >= 2]
        if cands:
            nx, nt = max(cands, key=lambda it: it[0][2])
            if nt.startswith(NAME_LABEL):
                name = (NAME_LABEL, nt[len(NAME_LABEL):], _ratio_bbox(nx, nt, NAME_LABEL))
            else:
                name = (NAME_LABEL, nt, nx)
    if name and name[1]:
        found[NAME_LABEL] = name

    return found


def extract_fields(engine, img, fields, box_thresh, text_score):
    """종목코드/종목명만 요청되면 빠른 경로 시도(실패 시 full 폴백). 그 외엔 full."""
    cropped, dx, dy = _crop_input_image(img)
    if set(fields) <= FAST_FIELDS:
        fast = extract_fields_fast(engine, cropped, box_thresh)
        if fast is not None and all(K in fast for K in fields):
            return _offset_found(fast, dx, dy)
    full = extract_fields_full(engine, cropped, fields, box_thresh, text_score)
    return _offset_found(full, dx, dy)


# ---------------------------------------------------------------- io
def draw_and_save(base_img, found, fields, src_path: Path) -> Path:
    out_img = base_img.copy()
    h, w = out_img.shape[:2]
    thick = max(2, round(min(h, w) / 400))
    for i, K in enumerate([k for k in fields if k in found], 1):
        _lab, _vtxt, bb = found[K]
        x0, y0, x1, y1 = (int(round(v)) for v in bb)
        cv2.rectangle(out_img, (x0, y0), (x1, y1), RED_BGR, thick)
        cv2.putText(out_img, TAG.get(K, f"F{i}"), (x0, max(12, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED_BGR, 2)
    out = src_path.with_name(f"{src_path.stem}_fields.jpg")
    cv2.imwrite(str(out), out_img)
    return out


def is_output(p: Path) -> bool:
    return any(p.stem.endswith(s) for s in OUT_SUFFIXES)


def resolve_images(items: list[str]) -> list[Path]:
    if not items:
        items = [DEFAULT_DIR]
    raw: list[Path] = []
    for a in items:
        if any(c in a for c in "*?["):
            raw += [Path(x) for x in glob.glob(a)]
        elif Path(a).is_dir():
            raw += sorted(Path(a).iterdir())
        else:
            raw.append(Path(a))
    seen, final = set(), []
    for p in raw:
        if p.suffix.lower() not in IMG_EXT or is_output(p) or not p.exists():
            continue
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            final.append(p)
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="파일/폴더/글로브 (없으면 jpg/ 폴더 전체)")
    ap.add_argument("--fields", default=DEFAULT_FIELDS)
    ap.add_argument("--box-thresh", type=float, default=0.3)
    ap.add_argument("--text-score", type=float, default=0.5)
    ap.add_argument("--open", action="store_true", help="1장일 때 결과를 뷰어로 띄움")
    ap.add_argument("--no-save", action="store_true",
                    help="박스 그리기/이미지 저장 생략(값만 빠르게 뽑을 때)")
    args = ap.parse_args()

    fields = [s.strip() for s in args.fields.split(",") if s.strip()]
    images = resolve_images(args.images)
    if not images:
        print("[오류] 처리할 이미지가 없습니다.")
        raise SystemExit(1)
    print(f"[설정] {len(images)}장 / 라벨={fields}\n")

    engine = RapidOCR(params={
        "Rec.lang_type": LangRec.KOREAN,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    })

    rows = []  # (name, found, t_ocr|None, t_draw)
    for idx, path in enumerate(images, 1):
        try:
            img = _LOADER(str(path))  # BGR 원본
        except Exception as exc:
            print(f"[{idx}/{len(images)}] {path.name}: 읽기 실패 ({exc})")
            rows.append((path.name, {}, None, 0.0))
            continue
        t = time.perf_counter()
        found = extract_fields(engine, img, fields, args.box_thresh, args.text_score)
        t_ocr = time.perf_counter() - t
        t_draw = 0.0
        if not args.no_save:
            td = time.perf_counter()
            draw_and_save(img, found, fields, path)
            t_draw = time.perf_counter() - td
        rows.append((path.name, found, t_ocr, t_draw))
        vals = "  ".join(f"{K}={found[K][1]!r}" if K in found else f"{K}=✗" for K in fields)
        extra = f"+저장 {t_draw:.2f}s" if not args.no_save else "저장생략"
        print(f"[{idx}/{len(images)}] {path.name}  (OCR {t_ocr:.2f}s {extra}): {vals}")

    print("\n===== 요약 =====")
    for K in fields:
        ok = sum(1 for _n, f, _o, _d in rows if K in f)
        print(f"  {K}: {ok}/{len(images)} 성공")
    miss = [n for n, f, _o, _d in rows if any(K not in f for K in fields)]
    if miss:
        print(f"  일부 미발견: {miss}")
    ocrs = [o for _n, _f, o, _d in rows if o is not None]
    draws = [_d for _n, _f, o, _d in rows if o is not None]
    if ocrs:
        line = (f"  OCR: 합계 {sum(ocrs):.1f}s | 장당 평균 {sum(ocrs)/len(ocrs):.2f}s "
                f"| 워밍업 후 최선 {min(ocrs):.2f}s")
        if not args.no_save:
            line += f"  ||  박스+저장: 합계 {sum(draws):.1f}s (장당 {sum(draws)/len(draws):.2f}s)"
        print(line)

    if len(images) == 1 and args.open and not args.no_save and rows[0][1]:
        os.startfile(str(images[0].with_name(f"{images[0].stem}_fields.jpg").resolve()))
        print("[표시] 기본 이미지 뷰어로 띄웠습니다.")


if __name__ == "__main__":
    main()
