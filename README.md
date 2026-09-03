# llm-serving

vLLM serving stack + Flask API proxy for the in-house GPU server (H200 x 2).

Split out of `auto_recipe_creator` on 2026-09-04. Nothing here imports that
repo — the two folders were already free of cross-imports, which is why the
split is a plain copy rather than a refactor.

## Layout

```
deploy_vlms/
  config/
    common.env              shared knobs (dtype, utilization, offline policy)
    models/<instance>.env   per-model knobs; each file documents WHY its values are what they are
  scripts/
    serve_vlm.py            builds the `vllm serve` command line from the env files
    start_model.py / stop_model.py / start_all.py
    start_qwen.py / start_mai_ui.py / start_paddleocr_vl.py
    check_vlm.py            liveness probe
    check_host_ram.py       host RAM accounting (PSS, not RSS)
    check_kv_longctx.py     fp8 KV long-context retrieval check (needle-in-a-haystack)
    upload_model.py         resumable chunked weight upload client
  nginx/                    location blocks (body size + timeouts)
  UPLOAD.md                 model weight upload runbook

flask_api/
  __init__.py               register_flask_api(app) — mounts everything under /api
  vlm_serve/                per-model proxy routes + service registry (config.py is the SSOT)
  model_upload/             chunked upload endpoint (store / routes / config, 3 layers)

docs/                       moved from auto_recipe_creator/docs/setup_vlms/
tests/                      server-side proxy tests (flask_api); model_upload tests live next to their code
```

## Currently served

| Port | Slug | GPU | Role |
|------|------|-----|------|
| 8002 | `mai-ui` | 0 | UI grounding (primary) |
| 8004 | `paddleocr-vl-1.5` | 0 | OCR assist |
| 8006 | `qwen3.8-27b` | 1 | general reasoning / coding |
| 8007 | `mai-ui-2b` | 0 | registered but disabled (A/B bench only) |

Host RAM is 16GB and there is no swap, so **process count hits the wall before
GPU memory does**. Do not run a 4th resident instance.

## Running

```bash
uv sync --extra dev

# bring up
uv run python deploy_vlms/scripts/start_all.py
uv run python deploy_vlms/scripts/start_model.py qwen3.8-27b
uv run python deploy_vlms/scripts/check_vlm.py http://127.0.0.1:8006 qwen3.8-27b

# health checks
uv run python deploy_vlms/scripts/check_host_ram.py       # run warm, after models are up
uv run python deploy_vlms/scripts/check_kv_longctx.py     # 5-10 min, run when idle

# tests (no GPU, no server — these run on a laptop)
uv run pytest
```

## Conventions

- **No CLI arguments.** Configuration lives in `config/*.env` and in the constant
  block at the top of each entry-point script. Shell env overrides both.
- `.env` is for secrets only; everything else is a config file or a constant block.
- Korean docstrings, `[INFO]`/`[ERROR]`/`[WARNING]` prints (not the `logging` module).
  Exception: the `flask_api/vlm_serve/*.py` route stubs keep one-line English docstrings.
- Model weights are referenced by **local absolute paths**. The office environment is
  offline — never rely on a live Hugging Face download.
- `vllm` and `torch` are deliberately not in `pyproject.toml`. This repo shells out to
  `vllm serve`; the runtime lives in the GPU server's own environment.

## Docs

Read in this order:

1. [`docs/01-runtime-layout-and-capacity.md`](docs/01-runtime-layout-and-capacity.md) — layout, offline policy, host RAM
2. [`docs/02-model-bringup-and-special-settings.md`](docs/02-model-bringup-and-special-settings.md) — bring-up order, per-model quirks
3. [`docs/03-ocr-and-parser-services.md`](docs/03-ocr-and-parser-services.md)
4. [`docs/04-operations-integration-and-benchmarking.md`](docs/04-operations-integration-and-benchmarking.md)
5. [`docs/08-serving-knobs-concepts.md`](docs/08-serving-knobs-concepts.md) — **what each knob actually does**: the weights/KV/activations split, BF16 vs FP8, KV cache math, context window, RoPE/YaRN, prefix cache
6. [`docs/09-inference-phases-and-capacity.md`](docs/09-inference-phases-and-capacity.md) — prefill vs decode, TTFT/TPOT, why batching helps decode but not prefill, multi-user capacity

Also `05` (resource comparison), `06` (small-VLM survey), `07` (H100 downgrade capacity),
and `deploy_vlms/UPLOAD.md` (weight upload runbook).

## Relationship to auto_recipe_creator

`deploy_vlms/` and `flask_api/` still exist in `auto_recipe_creator` as well, because its
`web_main.py` imports `flask_api` and the GPU server currently deploys from that checkout.
Until that deployment is repointed here, **the two copies can diverge** — treat
`auto_recipe_creator` as the live one and this repo as where the split is being staged.

The client side stays there and is not duplicated here: `poc/workflow_3/vlm/flask_vlm.py`
holds the *client* service registry (slugs, proxy URLs), deliberately separate from the
server registry in `flask_api/vlm_serve/config.py`.
