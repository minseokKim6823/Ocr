# -*- coding: utf-8 -*-
"""수표(bill/check*.jpg)에서 금액 / 일련번호 / MICR 코드 인식.

vis_fields_colab 와 동일하게 RapidOCR(OpenVINO 백엔드)를 쓰되,
- 금액·일련번호: 한국어 PP-OCRv5 rec
- MICR 라인: 전용 micr_rec 모델(+ micr_dict) 로 하단 스트립을 인식
규칙:
- 금액   = 상단 'W…(금…원정)' 라인의 숫자(콤마·마침표 제거)
- 일련번호 = MICR 라인 선두 숫자(가장 신뢰도 높음), 보조로 상단 '앞…' 뒤 숫자
- MICR   = 하단 자기문자 라인(좌→우 정렬). A/B/C/D = 자기기호(⑆⑇⑈⑉)

실행: python check_ocr.py            (전체)
      python check_ocr.py --debug    (OCR 줄 전체 덤프)
빌드: pyinstaller --noconfirm --clean --name check_ocr
        --collect-all rapidocr --collect-all openvino
        --add-data "<micr_rec_250313.onnx 경로>;micr"
        --add-data "<micr_dict.txt 경로>;micr"  check_ocr.py
"""
from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path

import cv2

from rapidocr import RapidOCR
from rapidocr.utils.load_image import LoadImage
from rapidocr.utils.typings import LangRec, OCRVersion, EngineType

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BILL = Path(r"C:\Users\minseok\Desktop\PythonProject\bill")
# exe(frozen)면 --add-data 로 _internal\micr 에 번들된 모델, 스크립트면 원본 경로
if getattr(sys, "frozen", False):
    _MICR_DIR = Path(sys._MEIPASS) / "micr"
else:
    _MICR_DIR = Path(r"C:\Users\minseok\Desktop\rapid-csharp-demo\OcrOnnxForm\models")
MICR_ONNX = str(_MICR_DIR / "micr_rec_250313.onnx")
MICR_DICT = str(_MICR_DIR / "micr_dict.txt")
LOADER = LoadImage()
ET = EngineType.OPENVINO
MICR_STRIP = 0.80  # 하단 20%를 MICR 스트립으로
# 자기문자(E-13B) 기호 매핑: A->:  B->;  C-><  D->=
MICR_SYM = str.maketrans({"A": ":", "B": ";", "C": "<", "D": "="})


# det 기본값(limit_type=min)은 짧은 변을 736px까지 업스케일해 작은 수표/스트립에서
# 시간·메모리가 수십 배 폭증한다. max로 바꿔 원본 해상도 그대로 검출(960 초과 시만 축소).
DET_FAST = {"Det.limit_type": "max", "Det.limit_side_len": 960}


def build_engines():
    main = RapidOCR(params={
        "Rec.lang_type": LangRec.KOREAN, "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Det.engine_type": ET, "Rec.engine_type": ET, "Cls.engine_type": ET,
        **DET_FAST,
    })
    micr = RapidOCR(params={
        "Rec.model_path": MICR_ONNX, "Rec.rec_keys_path": MICR_DICT,
        "Det.engine_type": ET, "Rec.engine_type": ET, "Cls.engine_type": ET,
        # MICR 스트립은 read_micr 에서 2배 확대해 넣으므로 한도도 2배로
        "Det.limit_type": "max", "Det.limit_side_len": 1920,
    })
    return main, micr


def get_lines(res):
    """(x0,y0,x1,y1,text,score) 리스트, 위→아래·좌→우 정렬."""
    out = []
    if getattr(res, "boxes", None) is not None and res.boxes is not None:
        for b, t, s in zip(res.boxes, res.txts, res.scores):
            xs = [float(p[0]) for p in b]; ys = [float(p[1]) for p in b]
            out.append((min(xs), min(ys), max(xs), max(ys), str(t), float(s)))
    out.sort(key=lambda r: (r[1], r[0]))
    return out


_KD = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_KU = {"십": 10, "백": 100, "천": 1000}
_KB = {"만": 10000, "억": 100000000, "조": 10 ** 12}


def parse_korean_amount(text):
    """한글 금액('금일억원정', '(일천만원정)' 등) -> 정수. 못 읽으면 None."""
    t = re.sub(r"\s", "", text)
    m = re.search(r"금?([일이삼사오육칠팔구십백천만억조]{2,})원정", t)
    if not m:
        return None
    total = section = num = 0
    for ch in m.group(1):
        if ch in _KD:
            num = _KD[ch]
        elif ch in _KU:
            section += (num or 1) * _KU[ch]; num = 0
        elif ch in _KB:
            section += num; total += section * _KB[ch]; section = num = 0
    val = total + section + num
    return val or None


def find_amount(lines, H):
    # ① 금액 행 = 'W/₩' 표시 또는 '원정' 포함(없으면 '금'+'원')
    anchor = None
    for ln in lines:
        t = ln[4]
        if "₩" in t or re.search(r"[Ww]\s*[\d.,]*\d", t):
            anchor = t
            break

    # ② 앵커 행의 콤마형 숫자(인쇄 금액)
    if anchor:
        m = re.search(r"\d[\d.,]*\d", anchor)
        if m and "," in m.group():
            d = re.sub(r"\D", "", m.group())
            if len(d) >= 4:
                return d, "digits"

    # ③ 한글 금액 파싱(숫자가 도장에 가려/깨졌을 때 견고)
    for ln in lines:
        v = parse_korean_amount(ln[4])
        if v:
            return str(v), "korean"

    # ④ 폴백: 상단 콤마형 숫자 최댓값
    best, bk = None, -1
    for x0, y0, x1, y1, t, s in lines:
        if y0 > 0.70 * H:
            continue
        m = re.search(r"\d[\d.,]*\d", t)
        if m and "," in m.group():
            d = re.sub(r"\D", "", m.group())
            if len(d) >= 4 and len(d) > bk:
                bk, best = len(d), d
    return (best, "digits") if best else (None, None)


def find_serial(lines, H, micr_digits):
    """일련번호 = (한글 접두) + (숫자).

    숫자는 MICR 선두 숫자(신뢰도 높음) 우선. 접두 한글은 상단 OCR이 '한글+숫자'로
    또렷이 읽은 경우(가가06816210)에만 붙인다. 숫자로 오인된 장식체 접두는
    붙이지 않는다(틀린 접두 방지).
    """
    prefix = ""
    top_digits = None
    for x0, y0, x1, y1, t, s in lines:
        if y0 > 0.50 * H:
            continue
        tt = t.replace(" ", "")
        # 접두는 정확히 한글 2자만 인정('앞가가06816210'의 가가). 라벨 '앞' 뒤
        # 노이즈가 1자만 한글로 읽힌 것('앞2까…', '앞#류…')을 접두로 오인하지 않게.
        m = re.search(r"([가-힣]{2})(\d{6,})", tt)
        if m:
            prefix, top_digits = m.group(1), m.group(2)
            break
        if top_digits is None:
            m2 = re.search(r"[앞#](\d{6,})", tt)
            if m2:
                top_digits = m2.group(1)
    digits = micr_digits or top_digits
    return (prefix + digits) if digits else None


def read_micr(micr_engine, img):
    """하단 스트립을 micr_rec 로 인식, 좌→우 정렬해 조각 리스트 반환."""
    H, W = img.shape[:2]
    strip = img[int(H * MICR_STRIP):H, 0:W]
    # 자기문자 기호(⑈/⑉ 등) 판별에는 원본 높이 43~77px가 빠듯해 3배 확대
    strip = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res = micr_engine(strip, use_cls=False)
    pieces = []
    if getattr(res, "boxes", None) is not None and res.boxes is not None:
        for b, t in zip(res.boxes, res.txts):
            x0 = min(float(p[0]) for p in b)
            pieces.append((x0, str(t)))
        pieces.sort(key=lambda r: r[0])
    elif getattr(res, "txts", None):
        pieces = [(0, str(t)) for t in res.txts]
    return [t for _x, t in pieces]


def micr_serial(pieces):
    """MICR 선두 숫자 = 일련번호."""
    joined = "".join(re.sub(r"\s", "", p) for p in pieces)
    m = re.match(r"(\d{6,})", joined)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    main_eng, micr_eng = build_engines()
    imgs = sorted(BILL.glob("check*.jpg"))

    # 워밍업(첫 추론 컴파일 제외)
    if imgs:
        w = LOADER(str(imgs[0])); main_eng(w, use_cls=False); read_micr(micr_eng, w)

    rows = []
    for p in imgs:
        img = LOADER(str(p))
        H, W = img.shape[:2]
        # 금액·일련번호는 상단부 -> 상단 58%만 OCR(주소/푸터 제외). 큰 수표는 0.7배
        # 축소해 검출 비용을 줄인다(금액 큰 글씨라 정확도 유지, 한글 금액 인식에도 유리).
        amt_crop = img[0:int(H * 0.58), :]
        if amt_crop.shape[1] > 700:
            amt_crop = cv2.resize(amt_crop, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        Hc = amt_crop.shape[0]

        tm = time.perf_counter()
        lines = get_lines(main_eng(amt_crop, use_cls=False))
        t_main = time.perf_counter() - tm
        tmi = time.perf_counter()
        micr_pieces = read_micr(micr_eng, img)
        t_micr = time.perf_counter() - tmi
        dt = t_main + t_micr

        amount, amt_src = find_amount(lines, Hc)
        serial = find_serial(lines, Hc, micr_serial(micr_pieces))
        micr_str = " ".join(micr_pieces).translate(MICR_SYM)
        rows.append((p.name, amount, amt_src, serial, micr_str, dt, t_main, t_micr))

        if args.debug:
            print(f"\n===== {p.name} ({W}x{H}) =====")
            for x0, y0, x1, y1, t, s in lines:
                print(f"  ({x0:4.0f},{y0:4.0f}) [{s:.2f}] {t!r}")
            print(f"  MICR pieces: {micr_pieces}")

    print("\n" + "=" * 86)
    print(f"{'파일':<11}{'금액':>16}  {'일련번호':<12} {'MICR':<28} {'time(main/micr)'}")
    print("-" * 86)
    tot = 0.0
    for name, amount, amt_src, serial, micr_str, dt, t_main, t_micr in rows:
        amt = f"{int(amount):,}" if amount else "?"
        tot += dt
        print(f"{name:<11}{amt:>16}  {serial or '?':<12} {micr_str:<28} {dt:.2f}s ({t_main:.2f}/{t_micr:.2f})")
    print("-" * 86)
    n = max(len(rows), 1)
    print(f"{'장당 평균':<11}{'':>16}  {'':<12} {'':<28} {tot / n:.2f}s")

if __name__ == "__main__":




    main()

