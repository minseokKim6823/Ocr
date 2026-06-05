# ocr-gemma: Gemma 4 기반 로컬 OCR

Gemma 4 비전-언어 모델을 사용하는 로컬 OCR 라이브러리입니다.
CPU 노트북(Ollama)과 GPU(HuggingFace Transformers / vLLM) 두 환경을 모두 지원하며,
인터넷 연결 없이 완전히 오프라인으로 동작합니다.

---

## 특징

| 항목 | 내용 |
|---|---|
| 모델 | Gemma 4 (Google DeepMind, 2026-04-02 출시, Apache 2.0) |
| OCR 정확도 | 최신 다목적 비전-언어 모델 기반의 높은 인식 정확도 |
| 다국어 지원 | 140개 이상 언어 (한국어 포함) |
| 로컬 실행 | 완전 오프라인 — 데이터가 외부로 나가지 않음 |
| 백엔드 | Ollama(CPU 최적화) / Transformers / vLLM(GPU 배치) |
| 입력 형식 | PNG, JPG 등 이미지 + PDF (페이지 단위) |
| 컨텍스트 | 최대 256K 토큰 |

---

## 환경별 빠른 시작

### CPU 노트북 (Ollama 백엔드)

> 현재 이 머신(Windows 11, GPU 없음)에서 권장되는 방법입니다.

**1. 환경 설치**

```powershell
# Windows PowerShell
.\setup_cpu.ps1
```

```bash
# Linux / macOS / WSL
bash setup_cpu.sh
```

**2. Gemma 4 모델 다운로드**

```bash
# 빠른 시작용 (약 7.2 GB, 속도 우선 — 멀티모달 비전 타워 포함)
ollama pull gemma4:e2b

# 더 높은 정확도 (약 8 GB+, 정확도 우선)
ollama pull gemma4:e4b

# 주의: e2b/e4b 모든 이미지 지원 변형은 7.2GB 이상입니다(q4_K_M 양자화 포함).
# 실행 시 모델 크기 + 여유분만큼의 가용 RAM이 필요합니다 (e2b 기준 약 8GB 권장).
```

**3. OCR 실행**

```bash
gemma-ocr --image sample.png --env cpu
```

**속도 최적화 팁:**

- `gemma4:e2b` 모델을 사용하면 가장 빠릅니다 (기본 설정).
- CPU 스레드 수를 명시적으로 지정해 성능을 개선할 수 있습니다: `--threads 8`
- 출력 토큰 수를 줄이면 짧은 문서에서 빠릅니다: `--max-new-tokens 512`
- Ollama가 백그라운드에서 실행 중인지 확인하세요: `ollama serve`

---

### GPU 서버 (HuggingFace Transformers / vLLM)

> GPU가 탑재된 별도의 머신에서 실행하는 경우입니다.

**1. 환경 설치**

```powershell
# Windows PowerShell (CUDA 포함 torch 별도 설치 필요)
.\setup_gpu.ps1
```

> **주의:** CUDA 버전에 맞는 PyTorch를 먼저 설치해야 합니다.
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

**2. OCR 실행**

```bash
gemma-ocr --image sample.png --env gpu
```

**vLLM 배치 처리 (Linux + CUDA):**

```bash
# vLLM 백엔드로 여러 이미지를 고속 처리
gemma-ocr --image doc1.png --backend vllm --model google/gemma-4-e4b-it
```

---

## Python API 사용 예

```python
from ocr_gemma import GemmaOCR, OCRConfig, DEFAULT_OCR_PROMPT

# --- Config 생성 ---

# CPU 노트북 (Ollama, gemma4:e2b)
config = OCRConfig.for_cpu()

# GPU 서버 (Transformers, google/gemma-4-e4b-it, bfloat16)
config = OCRConfig.for_gpu()

# 환경변수에서 읽기 (.env 또는 shell export)
config = OCRConfig.from_env()

# "cpu" 또는 "gpu" 문자열로 선택
config = OCRConfig.for_env("cpu")

# --- OCR 인스턴스 생성 ---
ocr = GemmaOCR(OCRConfig.for_cpu())

# 모델 예열 (첫 호출 지연 감소)
ocr.warmup()

# 현재 백엔드 이름 확인
print(ocr.backend_name)   # 예: "ollama:gemma4:e2b"

# --- 이미지 인식 ---
# 파일 경로(str), pathlib.Path, PIL.Image 모두 허용
text = ocr.read_image("invoice.png")
print(text)

# --- PDF 인식 (페이지별 텍스트 리스트 반환) ---
pages = ocr.read_pdf("document.pdf")
for i, page_text in enumerate(pages, 1):
    print(f"=== 페이지 {i} ===")
    print(page_text)

# --- 배치 처리 (이미지 목록 -> 텍스트 목록) ---
texts = ocr.read_batch(["page1.png", "page2.png", "page3.png"])
```

---

## CLI 레퍼런스

```bash
gemma-ocr [옵션]
# 또는
python -m ocr_gemma.cli [옵션]
```

| 플래그 | 설명 | 기본값 |
|---|---|---|
| `--image PATH` | 인식할 이미지 파일 경로 | — |
| `--pdf PATH` | 인식할 PDF 파일 경로 | — |
| `--env {cpu,gpu,auto}` | 환경 프리셋 선택 | `auto` |
| `--backend BACKEND` | 백엔드 직접 지정 (`ollama`/`transformers`/`vllm`) | — |
| `--model MODEL` | 모델 태그 또는 HF 모델 ID | — |
| `--device DEVICE` | 연산 장치 (`cpu`/`cuda`/`auto`) | — |
| `--prompt TEXT` | 커스텀 OCR 프롬프트 | 기본 프롬프트 |
| `--max-new-tokens N` | 최대 생성 토큰 수 | `2048` |
| `--threads N` | CPU 스레드 수 (Ollama) | 자동 |
| `--host URL` | Ollama 서버 주소 | `http://localhost:11434` |
| `--output PATH` | 결과 저장 파일 경로 | 표준 출력 |
| `--json` | JSON 형식으로 출력 | 비활성화 |
| `--warmup` | 추론 전 모델 예열 | 비활성화 |
| `--benchmark` | 타이밍 정보 출력 | 비활성화 |

**사용 예:**

```bash
# CPU로 이미지 OCR, JSON 출력
gemma-ocr --image invoice.png --env cpu --json

# PDF 전체 인식, 파일로 저장
gemma-ocr --pdf report.pdf --env cpu --output result.txt

# 정확도 우선 (e4b 모델), 예열 포함
gemma-ocr --image scan.png --env cpu --model gemma4:e4b --warmup

# 최대 토큰 수 줄여서 빠르게 처리
gemma-ocr --image note.png --env cpu --max-new-tokens 512 --threads 8
```

---

## 환경 변수 (.env) 설정

`.env.example`을 복사해 `.env`로 만든 후 값을 수정하세요.

```powershell
Copy-Item .env.example .env   # PowerShell
```

| 환경 변수 | 설명 | 기본값 |
|---|---|---|
| `OCR_BACKEND` | 백엔드 선택 (`ollama` / `transformers` / `vllm`) | `ollama` |
| `OCR_MODEL` | 모델 태그 또는 HF 모델 ID | `gemma4:e4b` |
| `OCR_DEVICE` | 연산 장치 (`cpu` / `cuda` / `auto`) | `auto` |
| `OCR_NUM_THREADS` | CPU 스레드 수 | 시스템 자동 |
| `OCR_MAX_NEW_TOKENS` | 최대 생성 토큰 수 | `2048` |
| `OCR_HOST` | Ollama 서버 URL | `http://localhost:11434` |

**CPU 노트북 프로파일 예:**

```dotenv
OCR_BACKEND=ollama
OCR_MODEL=gemma4:e2b
OCR_HOST=http://localhost:11434
OCR_MAX_NEW_TOKENS=2048
OCR_NUM_THREADS=8
```

**GPU 서버 프로파일 예:**

```dotenv
OCR_BACKEND=transformers
OCR_MODEL=google/gemma-4-e4b-it
OCR_DEVICE=auto
OCR_MAX_NEW_TOKENS=4096
```

---

## 속도 / 모델 선택 가이드

| 환경 | 백엔드 | 권장 모델 | 상대 속도 | 정확도 | 비고 |
|---|---|---|---|---|---|
| CPU 노트북 | `ollama` | `gemma4:e2b` | 매우 빠름 | 중간 | RAM 8 GB 이상, 기본 권장 |
| CPU 노트북 | `ollama` | `gemma4:e4b` | 빠름 | 높음 | RAM 16 GB 이상 권장 |
| GPU 서버 | `transformers` | `google/gemma-4-e4b-it` | 매우 빠름 | 높음 | CUDA bf16, VRAM 8 GB+ |
| GPU 서버 (배치) | `vllm` | `google/gemma-4-e4b-it` | 최고 처리량 | 높음 | Linux + CUDA 전용 |

---

## 트러블슈팅

### Ollama가 설치되지 않았거나 서버가 응답하지 않는 경우

```
ConnectionRefusedError: [Errno 111] Connection refused
```

- [https://ollama.com](https://ollama.com) 에서 Ollama를 설치합니다.
- 설치 후 `ollama serve`를 실행하거나, 시스템 트레이에서 Ollama가 실행 중인지 확인합니다.
- Ollama가 다른 포트에서 실행 중이라면 `--host` 플래그로 URL을 지정합니다:
  ```bash
  gemma-ocr --image img.png --host http://localhost:11435
  ```

### CUDA를 사용할 수 없는 경우

```
RuntimeError: CUDA is not available
```

- `--env cpu` 또는 `--device cpu`로 CPU 모드로 전환합니다.
- GPU 환경에서 사용하려면 CUDA 버전에 맞는 PyTorch를 설치해야 합니다:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```

### 모델 태그가 다르거나 찾을 수 없는 경우

```
Error: model 'gemma4:e2b' not found
```

- `ollama list`로 로컬에 설치된 모델 목록을 확인합니다.
- `--model` 플래그로 정확한 태그를 직접 지정합니다:
  ```bash
  gemma-ocr --image img.png --model gemma4:2b-instruct-q4_K_M
  ```

### PDF 변환 오류

```
ImportError: No module named 'fitz'
```

- PyMuPDF가 설치되지 않은 것입니다. 설치합니다:
  ```bash
  pip install pymupdf
  ```
- 이미 설치했는데도 오류가 발생하면 가상환경이 활성화됐는지 확인합니다:
  ```powershell
  .venv\Scripts\Activate.ps1   # PowerShell
  ```

### 한국어 / 특수문자 인식이 잘 안 되는 경우

- `gemma4:e4b` 또는 `google/gemma-4-e4b-it`(GPU) 모델로 전환해 정확도를 높입니다.
- `--max-new-tokens`를 늘려 긴 텍스트가 잘리지 않도록 합니다.
- 이미지 해상도가 낮으면 인식률이 떨어질 수 있습니다. 300 DPI 이상을 권장합니다.

---

## 라이선스

- **이 패키지 코드** — Apache 2.0 라이선스. 자유롭게 사용, 수정, 배포 가능합니다.
- **Gemma 4 모델 가중치** — [Gemma Terms of Use](https://ai.google.dev/gemma/terms) 적용.
  상업적 사용 전 반드시 약관을 확인하세요.
  Ollama를 통해 받은 GGUF 파일도 동일한 Gemma 약관의 적용을 받습니다.
