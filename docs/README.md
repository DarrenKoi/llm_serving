# VLM Deployment Guide

> [2026-09-03] ui-venus / ui-tars / got-ocr 은 가중치와 기동 스크립트를 모두 삭제했다.
> 아래 그 세 모델 관련 절차는 역사 기록이며, 실행하면 파일이 없다.
> 현재 서빙은 mai-ui / paddleocr-vl-1.5 / qwen3.8-27b 셋뿐이다.

`docs/setup_vlms/` is now the canonical home for installation, runtime tuning, model-specific special settings, and repo integration for the GPU serving stack.

This folder covers:

- vLLM runtime layout and capacity planning
- UI-Venus / MAI-UI / UI-TARS bring-up
- OCR/parser service deployment
- Flask proxy and repo-side integration

It replaces the older fragmented setup notes and also absorbs the vLLM runtime and model-setting research that used to live in `docs/research/`.

## Current Environment Anchors

- cloud base URL: `http://vlm-host.internal/`
- Flask API root: `http://vlm-host.internal/api`
- cloud repo root: `/path/to/llm_serving`
- `deploy_vlms` root: `/path/to/deploy_vlms`

## Service Defaults

| Port | Service slug | Typical role |
|------|--------------|--------------|
| `8001` | `ui-venus` | primary full-screen grounding |
| `8002` | `mai-ui` | crop retry / zoom-in sidecar |
| `8003` | `ui-tars` | alternate primary / agent-style comparison |
| `8004` | `paddleocr-vl-1.5` | OCR/layout sidecar |
| `8005` | `got-ocr` | hard OCR fallback |

## Recommended Reading Order

1. [`01-runtime-layout-and-capacity.md`](./01-runtime-layout-and-capacity.md)
2. [`02-model-bringup-and-special-settings.md`](./02-model-bringup-and-special-settings.md)
3. [`03-ocr-and-parser-services.md`](./03-ocr-and-parser-services.md)
4. [`04-operations-integration-and-benchmarking.md`](./04-operations-integration-and-benchmarking.md)
5. [`06-recent-small-vlm-hf-survey-2026-06-25.md`](./06-recent-small-vlm-hf-survey-2026-06-25.md)
6. [`07-h100-downgrade-capacity.md`](./07-h100-downgrade-capacity.md)
7. [`08-serving-knobs-concepts.md`](./08-serving-knobs-concepts.md) — context window / KV cache / BF16·FP8 / prefix cache 개념 정리
8. [`09-inference-phases-and-capacity.md`](./09-inference-phases-and-capacity.md) — prefill / decode / TTFT / 멀티유저 용량 산정

## Non-Negotiable Rules

- use local absolute model paths on the GPU server
- keep common settings in `deploy_vlms/config/common.env`
- keep model-specific settings in `deploy_vlms/config/models/*.env`
- do not rely on live Hugging Face downloads from the office environment
- keep `poc/work2` independent from server-side Python imports and env assumptions
- use fixed service slugs and stable endpoint contracts for coworkers

## Repo Anchors

- `deploy_vlms/scripts/serve_vlm.py`
- `deploy_vlms/scripts/start_model.py`
- `deploy_vlms/scripts/check_vlm.py`
- `deploy_vlms/scripts/start_paddleocr_vl.py`
- `deploy_vlms/scripts/run_got_ocr.py`
- `poc/work2/flask_vlm.py`
- `poc/work2/vlm_client.py`
- `poc/work2/connection_check.py`
