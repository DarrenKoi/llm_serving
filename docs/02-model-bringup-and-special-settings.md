# Model Bring-Up And Special Settings

> [2026-09-03] ui-venus / ui-tars / got-ocr 은 가중치와 기동 스크립트를 모두 삭제했다.
> 아래 그 세 모델 관련 절차는 역사 기록이며, 실행하면 파일이 없다.
> 현재 서빙은 mai-ui / paddleocr-vl-1.5 / qwen3.8-27b 셋뿐이다.

This document consolidates the old bring-up notes and the model-specific runtime differences, especially the extra care required for `UI-TARS`.

## 1. Recommended Bring-Up Order

Start simple:

1. `UI-Venus` on `8001`
2. `MAI-UI` on `8002`
3. `UI-TARS` on `8003`

Reason:

- `UI-Venus` is the most natural primary grounding baseline in this repo
- `MAI-UI` is a strong zoom-in or crop-retry sidecar
- `UI-TARS` is valuable, but its runtime stack is more sensitive

## 2. Pre-Checks

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
vllm --help
ss -ltn | grep 800
```

Confirm:

- the expected GPUs are visible
- the serving runtime exists in the target environment
- required ports are free

## 3. Standard Start And Check Commands

```bash
cd /path/to/deploy_vlms

python scripts/start_model.py mai-ui

python scripts/check_vlm.py http://127.0.0.1:8001 ui-venus-1.5-8b
python scripts/check_vlm.py http://127.0.0.1:8002 mai-ui-8b
python scripts/check_vlm.py http://127.0.0.1:8003 ui-tars-1.5-7b
```

Generic bring-up is preferred for variants:

```bash
python scripts/start_model.py <slug>
```

## 4. Model-Specific Settings

| Model | Required basics | Special notes |
|------|------------------|---------------|
| `UI-Venus` | `TRUST_REMOTE_CODE=1`, image limit, stable alias | strongest default full-screen grounding candidate |
| `MAI-UI` | `TRUST_REMOTE_CODE=1`, image limit | best used as crop/zoom sidecar rather than always-on primary |
| `UI-TARS` | `TRUST_REMOTE_CODE=1`, `Qwen2.5-VL`-compatible runtime, image limit | most sensitive to processor/template/shard completeness |

### 4.1 UI-Venus

Recommended defaults:

```bash
SERVED_MODEL_NAME=ui-venus-1.5-8b
PORT=8001
GPU_ID=0
TRUST_REMOTE_CODE=1
LIMIT_MM_PER_PROMPT={"image": 1}
```

Operational role:

- primary full-screen grounding
- first comparison point for RCS-like screens

### 4.2 MAI-UI

Recommended defaults:

```bash
SERVED_MODEL_NAME=mai-ui-8b
PORT=8002
GPU_ID=1
TRUST_REMOTE_CODE=1
LIMIT_MM_PER_PROMPT={"image": 1}
```

Operational role:

- crop retry
- small target disambiguation
- local second opinion near dense UI clusters

### 4.3 UI-TARS

Recommended defaults:

```bash
SERVED_MODEL_NAME=ui-tars-1.5-7b
PORT=8003
GPU_ID=0
TRUST_REMOTE_CODE=1
LIMIT_MM_PER_PROMPT={"image": 1, "video": 0}
```

Important differences:

- `UI-TARS` is a `Qwen2.5-VL`-family runtime case, not the same stack expectation as `UI-Venus` or `MAI-UI`
- the model directory must include the full processor/template/shard set
- missing `chat_template.json`, `preprocessor_config.json`, `tokenizer_config.json`, or incomplete shard copies can break startup

Special handling rule:

- if `CHAT_TEMPLATE` is empty, first rely on the model directory's own template
- if runtime output or formatting is unstable, point `CHAT_TEMPLATE` at a `.jinja` file you add under `config/` (the UI-TARS template was removed with its weights, 2026-09-03/05)

## 5. Repo Role Interpretation

For this repository, the most practical split is:

| Role | Best model |
|------|------------|
| primary full-screen grounding | `UI-Venus` |
| alternate primary / agent-style comparison | `UI-TARS` |
| crop retry sidecar | `MAI-UI` |

This is an operational interpretation, not a claim that the model cards guarantee those exact roles.

## 6. Bring-Up Troubleshooting

### 6.1 Server Fails Before Serving

Check:

- `MODEL_ID` exists and is complete
- `TRUST_REMOTE_CODE=1`
- tokenizer, preprocessor, and template files are present
- the runtime can import the required multimodal architecture

### 6.2 UI-TARS Starts But Behaves Oddly

Check:

- runtime support for `Qwen2.5-VL`
- template selection
- shard completeness
- whether the prompt format matches the model's grounding style

### 6.3 Memory Or Slow Startup

Reduce in this order:

1. `MAX_NUM_SEQS`
2. `MAX_MODEL_LEN`
3. `GPU_MEMORY_UTILIZATION`

## 7. Validation Rule

Always validate both:

- `/v1/models` health
- one real screenshot request through the same code path used by `poc/work2`

The model is not truly ready until both pass.
