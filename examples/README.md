# examples/ — 예제 스크립트 모음

`ocr_gemma` 패키지의 사용법을 보여주는 독립 실행 예제입니다.
모든 예제는 가상환경을 활성화한 상태에서 **프로젝트 루트**에서 실행하세요.

```powershell
# PowerShell — 가상환경 활성화
.venv\Scripts\Activate.ps1
```

---

## 예제 파일

### run_ocr.py — 단일 이미지 OCR

이미지 한 장을 받아 텍스트를 인식하고 결과와 소요 시간을 출력합니다.
모델 예열(warmup)을 수행해 첫 추론의 콜드-스타트 지연을 줄입니다.

```bash
# 기본 실행 (sample.png, CPU 환경)
python examples/run_ocr.py

# 이미지 경로 직접 지정
python examples/run_ocr.py invoice.png

# GPU 환경으로 실행
OCR_ENV=gpu python examples/run_ocr.py scan.jpg
```

---

### batch_pdf.py — PDF 배치 OCR

PDF 파일 전체를 페이지 단위로 OCR하고,
각 페이지의 텍스트를 `out/page_NN.txt` 파일로 저장합니다.

```bash
# 기본 실행 (sample.pdf, CPU 환경)
python examples/batch_pdf.py

# PDF 경로 직접 지정
python examples/batch_pdf.py report.pdf

# GPU 환경으로 실행
OCR_ENV=gpu python examples/batch_pdf.py scanned_book.pdf
```

결과 파일은 프로젝트 루트의 `out/` 디렉토리에 생성됩니다.

---

### benchmark.py — 성능 벤치마크

지정한 이미지에 대해 N회 반복 추론을 수행하고
최소/평균/중앙값 소요 시간, 처리 속도(chars/sec)를 마크다운 표로 출력합니다.

```bash
# 기본 실행 (sample.png, CPU, 3회 반복)
python examples/benchmark.py

# 옵션 지정
python examples/benchmark.py --image invoice.png --env cpu --runs 5

# 예열 없이 실행
python examples/benchmark.py --image scan.png --runs 10 --no-warmup

# GPU 환경 벤치마크
python examples/benchmark.py --image doc.png --env gpu --runs 5
```

---

## 샘플 이미지 준비 방법

예제를 바로 실행하려면 `sample.png` 파일이 필요합니다.

```powershell
# 방법 1: 직접 이미지를 복사
Copy-Item C:\path\to\your\image.png sample.png

# 방법 2: 스크린샷 찍기 (Windows 캡처 도구 또는 Snipping Tool)
#   → sample.png 로 저장

# 방법 3: 경로를 직접 지정 (파일 복사 불필요)
python examples/run_ocr.py C:\path\to\your\image.png
```

이미지 파일이 없는 상태에서 실행하면 예제가 안내 메시지를 출력하고 종료합니다.

---

## 환경 변수로 설정 변경

| 변수 | 설명 | 기본값 |
|---|---|---|
| `OCR_ENV` | 환경 프리셋 (`cpu` / `gpu` / `auto`) | `cpu` |
| `OCR_IMAGE` | `run_ocr.py` 기본 이미지 경로 | `sample.png` |
| `OCR_PDF` | `batch_pdf.py` 기본 PDF 경로 | `sample.pdf` |

`.env` 파일을 만들어 두면 매번 환경변수를 입력하지 않아도 됩니다
(`python-dotenv` 설치 시 자동으로 읽힙니다).
