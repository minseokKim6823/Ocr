# Gemini Advisor Artifact

## Original user task

$ask-gemini 혹시 vis_fileds에서 성능적으로 개선할 부분이 있는지 확인해줘

## Final prompt sent to Gemini CLI

Review @vis_fields.py for performance improvements. Context: this script uses RapidOCR to extract Korean fields '종목코드' and '종목명' from document images. Current fast path crops input by x=0.05 each side and y=0.10 top/bottom, runs detection once, recognizes only left label candidates and then the target row, with full OCR fallback. Recent sample run over 6 IRN images: all fields found, average OCR about 2.40s/image. Constraints: no new dependencies, preserve behavior/accuracy, keep changes small and reversible. Please identify concrete performance bottlenecks and rank low-risk code changes, especially opportunities to reduce OCR calls, crop count, fallback cost, image copies, or repeated recognition. Do not suggest broad rewrites unless clearly justified.

## Gemini output (raw)

```text
[ask-gemini] Missing required local CLI binary: gemini
[ask-gemini] Install/configure gemini CLI, then verify with: gemini --version
```

## Concise summary

The local Gemini CLI is not installed or not discoverable on PATH, so the requested external advisor review could not run. Per the skill instructions, no MCP fallback was used.

## Action items / next steps

- Install/configure the local Gemini CLI.
- Verify it with `gemini --version`.
- Re-run the ask-gemini review after the binary is available.
