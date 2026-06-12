"""Colab용 vis_fields 벤치마크 드라이버 — onnxruntime vs openvino 속도 비교.

원본 vis_fields.py를 '수정 없이' import 해서 같은 이미지로 두 엔진의
OCR 속도를 잰다. Colab(리눅스)에 맞춰 입력은 업로드/폴더 모두 지원하고,
결과 박스는 cv2_imshow로 인라인 표시한다(os.startfile 미사용).

────────────────────────────────────────────────────────────────────
⚠️ 주의: Colab의 CPU는 이 노트북(i5-8250U)이 아닙니다.
   Colab은 보통 Intel Xeon(AVX-512·VNNI 탑재 가능)이라, 여기서 측정한
   'OpenVINO 배수'는 노트북에서 그대로 재현되지 않습니다.
   - Colab으로 검증 가능: 동작 여부 · OCR 값 정확도 동일성 · 대략적 경향
   - Colab으로 알 수 없음: i5-8250U에서의 실제 배수 (그건 노트북에서 직접 측정해야 정확)
────────────────────────────────────────────────────────────────────

Colab 셀 사용법:
    # 1) 설치 (openvino는 별도 패키지)
    !pip -q install rapidocr openvino

    # 2) 파일 업로드: vis_fields.py + vis_fields_colab.py + 이미지(IRN1.jpg 등)
    from google.colab import files; files.upload()

    # 3a) 노트북에서 바로
    import vis_fields_colab as vc
    vc.compare("IRN1.jpg", repeats=5)            # 한 장
    vc.compare(["IRN1.jpg", "IRN2.jpg"], repeats=5)

    # 3b) 또는 셸로
    !python vis_fields_colab.py IRN1.jpg --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from rapidocr import RapidOCR
from rapidocr.utils.typings import EngineType, LangRec, OCRVersion

import vis_fields as vf  # 원본 로직 재사용(extract_fields, draw_and_save, resolve_images, _LOADER)

ENGINES = ("onnxruntime", "openvino")


def build_engine(engine_type: str) -> RapidOCR:
    """원본 main()과 동일한 설정 + det/rec/cls 백엔드만 engine_type으로 교체.

    원본은 Rec만 한국어/PP-OCRv5로 지정하고 Det는 기본값(ch·PP-OCRv4 mobile)을
    쓰므로 그 동작을 그대로 유지한다. engine_type만 바꿔 공정 비교.

    engine_type은 문자열("onnxruntime"/"openvino")을 받아 RapidOCR이 요구하는
    EngineType enum으로 변환해 넘긴다(문자열 그대로 주면 검증 에러).
    """
    et = EngineType(engine_type)  # "openvino" -> EngineType.OPENVINO
    return RapidOCR(params={
        "Rec.lang_type": LangRec.KOREAN,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Det.engine_type": et,
        "Rec.engine_type": et,
        "Cls.engine_type": et,  # use_cls=False라 미사용이지만 일관성 위해 지정
    })


def _measure(engine, imgs, field_list, box_thresh, text_score, repeats):
    """워밍업 1패스(모델 로드+최초 추론) 후 repeats 패스 측정. (times, found_by_name)."""
    for _p, img in imgs:  # 워밍업(측정 제외)
        vf.extract_fields(engine, img, field_list, box_thresh, text_score)
    times, found_by_name = [], {}
    for _ in range(repeats):
        for p, img in imgs:
            t = time.perf_counter()
            found = vf.extract_fields(engine, img, field_list, box_thresh, text_score)
            times.append(time.perf_counter() - t)
            found_by_name[p.name] = found
    return times, found_by_name


def _vals(found, field_list):
    return {K: (found[K][1] if K in found else None) for K in field_list}


def compare(images=None, fields=vf.DEFAULT_FIELDS, repeats=5,
            box_thresh=0.3, text_score=0.5, engines=ENGINES,
            show=True, draw_with="openvino"):
    """engines를 같은 이미지로 돌려 속도/정확도 비교. 결과 요약 dict 반환."""
    field_list = [s.strip() for s in fields.split(",") if s.strip()]
    items = [images] if isinstance(images, str) else (list(images) if images else [])
    paths = vf.resolve_images(items)
    if not paths:
        raise SystemExit("[오류] 처리할 이미지가 없습니다. (업로드 후 파일명을 넘기세요)")
    imgs = [(p, vf._LOADER(str(p))) for p in paths]
    print(f"[설정] {len(imgs)}장 / 라벨={field_list} / 엔진={list(engines)} / repeats={repeats}")

    results = {}
    for et in engines:
        print(f"\n[{et}] 엔진 생성 중...")
        try:
            engine = build_engine(et)
        except Exception as exc:  # openvino 미설치 등
            print(f"  ✗ 생성/실행 실패: {exc}")
            continue
        try:
            times, found = _measure(engine, imgs, field_list, box_thresh, text_score, repeats)
        except Exception as exc:
            print(f"  ✗ 추론 실패: {exc}")
            continue
        results[et] = {"times": times, "found": found, "engine": engine}
        print(f"  측정 {len(times)}회 → 중앙값 {statistics.median(times):.3f}s "
              f"| 최선 {min(times):.3f}s | 평균 {statistics.mean(times):.3f}s")

    # ---- 속도 비교 ----
    ok = [et for et in engines if et in results]
    if len(ok) >= 2:
        a, b = ok[0], ok[1]
        ma, mb = statistics.median(results[a]["times"]), statistics.median(results[b]["times"])
        faster, mult = (b, ma / mb) if mb < ma else (a, mb / ma)
        print(f"\n===== 속도(중앙값) =====")
        print(f"  {a}: {ma:.3f}s   {b}: {mb:.3f}s")
        print(f"  → {faster} 가 {mult:.2f}배 빠름")

        # ---- 정확도 동일성 ----
        diffs = []
        for p, _img in imgs:
            va, vb = _vals(results[a]["found"][p.name], field_list), _vals(results[b]["found"][p.name], field_list)
            if va != vb:
                diffs.append((p.name, va, vb))
        if diffs:
            print(f"\n  ⚠️ 인식값 불일치 {len(diffs)}건 (OpenVINO 전환 시 정확도 검증 필요):")
            for name, va, vb in diffs:
                print(f"    - {name}: {a}={va}  {b}={vb}")
        else:
            print(f"  ✅ 두 엔진 인식값 동일 (정확도 보존)")

    # ---- 결과 박스 표시 ----
    if show and results:
        et = draw_with if draw_with in results else ok[0]
        p0, img0 = imgs[0]
        out = vf.draw_and_save(img0, results[et]["found"][p0.name], field_list, p0)
        _show(str(out), title=f"{p0.name} ({et})")

    return {et: {"median": statistics.median(r["times"]), "best": min(r["times"]),
                 "times": r["times"]} for et, r in results.items()}


def _show(img_path: str, title: str = ""):
    """Colab이면 인라인 표시, 아니면 경로만 출력."""
    import cv2
    img = cv2.imread(img_path)
    if title:
        print(f"[표시] {title} -> {img_path}")
    try:
        from google.colab.patches import cv2_imshow  # Colab 전용
        cv2_imshow(img)
    except Exception:
        print(f"  (비-Colab 환경) 저장됨: {img_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="onnxruntime vs openvino OCR 속도 비교(Colab)")
    ap.add_argument("images", nargs="*", help="파일/폴더/글로브 (없으면 jpg/ 폴더)")
    ap.add_argument("--fields", default=vf.DEFAULT_FIELDS)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--box-thresh", type=float, default=0.3)
    ap.add_argument("--text-score", type=float, default=0.5)
    ap.add_argument("--engines", default=",".join(ENGINES),
                    help="비교할 엔진 (쉼표): onnxruntime,openvino")
    ap.add_argument("--no-show", action="store_true", help="결과 박스 표시 생략")
    # Colab의 %run 등이 주입하는 미지의 인자는 무시
    args, _unknown = ap.parse_known_args()

    compare(
        images=args.images or None,
        fields=args.fields,
        repeats=args.repeats,
        box_thresh=args.box_thresh,
        text_score=args.text_score,
        engines=tuple(s.strip() for s in args.engines.split(",") if s.strip()),
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
