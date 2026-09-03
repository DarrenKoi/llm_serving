# OCR And Parser Services

> [2026-09-03] ui-venus / ui-tars / got-ocr 은 가중치와 기동 스크립트를 모두 삭제했다.
> 아래 그 세 모델 관련 절차는 역사 기록이며, 실행하면 파일이 없다.
> 현재 서빙은 mai-ui / paddleocr-vl-1.5 / qwen3.8-27b 셋뿐이다.

This document combines the OCR deployment notes and OmniParser installation guidance.

## 1. Service Role Split

| Service | Runtime style | Best role |
|---------|---------------|-----------|
| `PaddleOCR-VL-1.5` | `vLLM` | OCR, spotting, table/layout extraction |
| `GOT-OCR-2.0-hf` | `transformers`-style direct inference | hard crop OCR fallback, region reread |
| `OmniParser V2` | separate parser pipeline, not `vLLM` | interactable boxes, icon captions, SoM overlay |

## 2. PaddleOCR-VL-1.5

Use when you need:

- OCR with layout understanding
- `Spotting:` output with locations
- table-like or dense parameter extraction
- OCR evidence for `UI-Venus` or `MAI-UI`

Bring-up path:

```bash
cd /path/to/deploy_vlms
python scripts/start_paddleocr_vl.py
python scripts/check_vlm.py http://127.0.0.1:8004 paddleocr-vl-1.5
```

Operational guidance:

- use `OCR:` for broad reading
- use `Spotting:` when coordinates matter
- use `Table Recognition:` for structured panels

## 3. GOT-OCR-2.0-hf

Use when you need:

- crop-only rereads
- formatting-sensitive OCR
- exact text fallback on difficult small regions

It should not be treated as another general-purpose chat VLM.

Bring-up path:

```bash
cd /path/to/deploy_vlms
python scripts/run_got_ocr.py
```

Practical rule:

- use it as a region OCR tool once another model has already narrowed the target area

## 4. OmniParser V2

Use when you need:

- interactable boxes
- icon captions
- SoM-style marked UI parsing
- a structured parser sidecar for inaccessible or custom-rendered UI

Important deployment note:

- OmniParser is not a plain `vLLM` model
- deploy it as a separate parser service, not as another `/v1/chat/completions` model

## 5. OmniParser Offline Installation Checklist

Before office deployment, stage:

1. the OmniParser code repository
2. the OmniParser V2 weights
3. Florence assets needed for captioning/processor code
4. Hugging Face cache artifacts required by the runtime

Recommended operational pattern:

- use a dedicated `uv` or Python environment for OmniParser
- keep its dependencies separate from the main `vLLM` runtime
- run a small smoke test before wiring it into Flask

## 6. OmniParser Risks

Main concerns:

- extra dependency surface compared with plain `vLLM`
- separate deployment contract
- AGPL-3.0 implications around the detection component

Because of that, OmniParser is best treated as an optional parser service, not a mandatory part of the base serving stack.

## 7. Repo Integration Guidance

Suggested repo roles:

- `PaddleOCR-VL-1.5`: default OCR evidence engine for `poc/work2`
- `GOT-OCR-2.0-hf`: hard region fallback
- `OmniParser V2`: optional structured UI parser feeding hints or SoM images to the primary GUI model

Use-side guidance for `UI-Venus` plus OCR lives in `docs/gui_control/03-ui-venus-ocr-crop-retry.md`.
