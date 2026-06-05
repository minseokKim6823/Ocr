"""인식된 텍스트 영역을 '빨간 박스'로 표시한 이미지를 만들어 띄운다.

RapidOCR이 검출한 박스 좌표(result.boxes)를 원본 위에 빨간 사각형으로 그린다.
기본 result.vis()는 여러 색 + 좌우(원본/텍스트) 분할이지만,
이 스크립트는 '원본 한 장 + 빨간 박스만' 그려서 보기 편하게 한다.

사용법:
    python vis_boxes.py                      # 기본: 로컬 sample.png
    python vis_boxes.py jpg/1.jpg            # 로컬 이미지
    python vis_boxes.py jpg/1.jpg --no-open  # 띄우지 않고 저장만
    python vis_boxes.py https://...jpg       # URL도 그대로 됨(선택)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

# 기본은 로컬 파일. 인자로 다른 경로(또는 URL)를 주면 그걸 쓴다.
DEFAULT_IMG = "sample.png"

RED_BGR = (0, 0, 255)  # cv2는 BGR 순서 -> (0,0,255) = 빨강


def main() -> None:
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    no_open = "--no-open" in sys.argv
    src = positional[0] if positional else DEFAULT_IMG

    print(f"[입력] {src}")
    engine = RapidOCR()

    try:
        result = engine(src)
    except Exception as exc:  # URL 다운로드 실패 등
        print(f"[오류] OCR 실행 실패: {exc}")
        print("       오프라인이면 로컬 이미지 경로를 인자로 주세요. 예) python vis_boxes.py jpg/1.jpg")
        sys.exit(1)

    boxes = getattr(result, "boxes", None)
    img = getattr(result, "img", None)
    if img is None or boxes is None or len(boxes) == 0:
        print("[결과] 인식된 텍스트 영역이 없습니다.")
        sys.exit(0)

    img = img.copy()  # BGR np.ndarray
    h, w = img.shape[:2]
    thickness = max(2, round(min(h, w) / 400))  # 이미지 크기에 맞춰 선 굵기 조절

    for box in boxes:
        pts = np.int32(box).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=RED_BGR, thickness=thickness)

    # 저장 경로: 로컬 파일이면 <원본이름>_redbox.jpg, URL이면 vis_redbox.jpg
    src_path = Path(src)
    if src_path.exists():
        out = src_path.with_name(f"{src_path.stem}_redbox.jpg")
    else:
        out = Path("vis_redbox.jpg")
    cv2.imwrite(str(out), img)

    out_abs = out.resolve()
    print(f"[완료] 빨간 박스 {len(boxes)}개를 그렸습니다 -> {out_abs}")

    if no_open:
        return
    if os.name == "nt":
        os.startfile(str(out_abs))  # Windows 기본 이미지 뷰어로 띄우기
        print("[표시] 기본 이미지 뷰어로 띄웠습니다.")
    else:
        print("[안내] Windows가 아니어서 자동으로 띄우지 않았습니다. 위 경로의 파일을 열어보세요.")


if __name__ == "__main__":
    main()
