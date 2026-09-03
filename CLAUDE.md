# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

vLLM serving stack + Flask API proxy for an in-house GPU server (H200 × 2). Split out of
`auto_recipe_creator` on 2026-09-04.

**The repo is public.** All site-specific values are placeholders (`${MODEL_ROOT}`,
`vlm-host.internal`); real paths and secrets live in the gitignored `.env`. Never commit an
internal hostname or absolute model path — the history was scrubbed once already to remove them.

`README.md` and `docs/README.md` are **deliberately empty** (emptied 2026-09-04 to reduce search
indexing). Do not repopulate them without being asked. Their content is recoverable from git history.

## Setup and commands

```bash
uv sync --extra dev

# Nothing auto-loads .env — export it before running anything:
cp .env_example .env      # then fill in MODEL_ROOT and the tokens
set -a; . ./.env; set +a
```

The **dev laptop uses `uv`; the GPU server uses pip + plain `python`** (vLLM 0.28.0,
Python 3.11.9). Drop the `uv run` prefix when running anything on the server.

```bash
# tests — no GPU, no server; these run on a laptop
uv run pytest
uv run pytest deploy_vlms/scripts/test_serve_vlm_env.py       # one file
uv run pytest tests/test_vlm_serve.py::test_models_proxy_uses_expected_upstream   # one test

# bring up / tear down (GPU server only)
uv run python deploy_vlms/scripts/start_all.py
uv run python deploy_vlms/scripts/start_model.py qwen3.8-27b
uv run python deploy_vlms/scripts/stop_model.py all
uv run python deploy_vlms/scripts/check_vlm.py http://127.0.0.1:8006 qwen3.8-27b

# health checks
uv run python deploy_vlms/scripts/check_host_ram.py     # run warm, after models are up
uv run python deploy_vlms/scripts/check_kv_longctx.py   # 5-10 min, run when idle
```

Tests live in **two places**: `tests/` (Flask proxy) and beside the code they cover
(`flask_api/model_upload/test_*.py`, `deploy_vlms/scripts/test_*.py`). `pytest` finds both via
`pythonpath = ["."]`. There is no linter or formatter configured — don't invent one.

## Architecture

Two halves that **never import each other**. They meet only through
`deploy_vlms/config/models/*.env`, which the launcher consumes as config and the Flask health
probe reads at runtime to discover what should be running.

### deploy_vlms/ — process launcher

`serve_vlm.py <instance>` is the core. It never imports vLLM; it builds an argv and calls
`os.execvpe`, replacing itself with `vllm serve`. The resolution chain:

1. Load `config/common.env`, then `config/models/<instance>.env`.
2. `load_env_file` **assigns unconditionally** (`os.environ[key] = value`), so config files win
   over the shell. Exporting `PORT=…` does *not* override a model's config.
3. Values are `os.path.expandvars`-expanded, so `${MODEL_ROOT}` from `.env` flows in. This is the
   only channel for shell → config. An unresolved var stays literal and trips the `MODEL_ID`
   `isabs` check rather than silently becoming a relative path.
4. Validate (absolute path, dir exists, under `ALLOWED_MODEL_ROOT`), optionally auto-tune GPU
   memory from the model's `config.json`, then exec.

`start_all.py` starts models **strictly one at a time**, largest first (`VLLM_MODELS` order,
`qwen3.8-27b` leading), waiting for each to answer `/v1/models` before the next. This is a host-RAM
guarantee, not a convenience: weight loading and CUDA-graph capture are each model's RSS peak, and
overlapping those peaks on a 16 GB swapless host invites the OOM killer. If a model times out while
still alive it is stopped before the next one starts — `deploy_vlms/scripts/test_start_all.py`
pins that. Per-model wrappers (`start_qwen.py`, `start_mai_ui.py`, `start_paddleocr_vl.py`) are
three-line shims over `start_model.py`.

`start_model.py` backgrounds it and writes `deploy_vlms/runtime/{logs,pids}/<instance>.{log,pid}`
(gitignored). `stop_model.py` resolves a target by instance name → PID file → falling back to
whoever holds the port. 

`AUTO_TUNE_GPU_MEMORY_UTILIZATION` assumes every layer holds KV. Do not enable it for
`qwen3.8-27b`: 48 of its 64 layers are GatedDeltaNet (linear attention, fixed-size state), so
auto-tune overestimates KV ~4× and refuses to start. The per-model `.env` files carry long
comments explaining *why* each value is what it is — read them before changing a number.

### flask_api/ — reverse proxy, mounted at `/api`

`register_flask_api(app)` is the only entry point. Blueprints nest:
`/api` → `/api/vlm_serve` → `/api/vlm_serve/<slug>`, plus `/api/model_upload`.

The proxy buffers the upstream response fully and returns it verbatim
(`service_template.py`). Per-service files (`mai_ui.py`, `qwen3_8_27b.py`, …) are ~13-line
`VLMServiceConfig` declarations over that shared template.

**A model's slug is declared in three places and they must agree:**

| Place | Declares |
|---|---|
| `flask_api/vlm_serve/config.py` → `ALL_VLM_SERVICES` | port, display name, `enabled` flag |
| `flask_api/vlm_serve/<model>.py` → `VLMServiceConfig` | port again, blueprint name |
| `deploy_vlms/config/models/<slug>.env` | `PORT` for the actual vLLM process |

The `.env` **filename stem is the slug** — `_configured_vlm_entries()` globs the directory and
uses `env_path.stem`. Renaming the file silently unregisters the service from health reporting.
Adding a model means touching all three, plus the import list in `vlm_serve/__init__.py`.

`enabled=False` in `config.py` deregisters the blueprint at import time. Deleted models are
removed outright rather than left disabled — a disabled entry falsely implies flipping the flag
would revive it, when the checkpoint is actually gone from the server.

`VLM_SERVE_TOKEN` (in `.env`) gates `/api/vlm_serve/<slug>/v1/*` with one shared team token,
checked in a `before_request` on each service blueprint. Empty means auth is **off**, so setting it
is opt-in and an existing deployment never breaks silently. `home` and `health` stay open for
monitoring. Callers may send `X-VLM-Token` or `Authorization: Bearer` (the OpenAI client uses the
latter) — and when the token is configured the caller's `Authorization` is **stripped before
forwarding**, so it cannot collide with the upstream `API_KEY` the proxy injects.

`/api/health` is a **reconciliation, not a stored flag**: it merges declared proxies with
discovered `.env` files, live-probes each upstream `/v1/models`, and reports `serving_mismatch`
when the served model name disagrees with what was configured.

### flask_api/model_upload/ — resumable chunked upload

Three layers, deliberately: `store.py` (filesystem + resume state, **knows nothing about HTTP**),
`routes.py` (Flask wrapper), `config.py` (env wiring). That split is why resume and integrity are
testable without a running server — keep new logic in `store.py`.

Staging must stay **inside** the destination root so `os.replace` is atomic on one filesystem.
`MODEL_UPLOAD_ROOT` must be set for the Flask process: it does not read `common.env`, so
`ALLOWED_MODEL_ROOT` never reaches it. Runbook in `deploy_vlms/UPLOAD.md`, nginx body-size and
timeout blocks in `deploy_vlms/nginx/`.

## Hard constraints

- **Host RAM is 16 GB with no swap — process count hits the wall before GPU memory does.** Do not
  run a fourth resident instance. Every model `.env` carries
  `--load-format safetensors --safetensors-load-strategy lazy --mm-processor-cache-gb 0` for this
  reason; the lazy/mmap flags are explicit because vLLM's default auto-prefetches on NFS.
- **The office environment is offline.** Model weights are referenced by local absolute paths.
  Never add a code path that relies on a live Hugging Face download.
- **`vllm` and `torch` are intentionally absent from `pyproject.toml`.** This repo shells out to
  `vllm serve`; the runtime lives in the GPU server's own environment, and declaring them here
  breaks installs on a dev laptop.
- Server runs **vLLM 0.28.0 / Python 3.11.9** (confirmed 2026-09-04). Comments dated 2026-09-03
  and earlier were written against 0.19.1 — treat version-specific claims in them as unverified.
- `--swap-space` was removed with the V1 engine and caused a startup failure here (2026-08-11).
- MTP speculative decoding is off for `qwen3.8-27b` on purpose: it crashed above ~26k tokens on
  0.19.1 (vllm#40756). The 0.28.0 upgrade may have fixed it, but that is **unconfirmed** — do not
  enable it without checking the issue and re-running `check_kv_longctx.py`. Long context is the
  requirement; MTP is only an optimization.

## Conventions

- **No CLI arguments.** Configuration lives in `config/*.env` and in the constant block at the top
  of each entry-point script.
- Korean docstrings and comments. `deploy_vlms/` scripts print `[INFO]` / `[WARNING]` / `[ERROR]`
  rather than using `logging`; `flask_api/vlm_serve/` goes through `logger.py`
  (`get_vlm_logger`), and its route stubs keep one-line English docstrings.
- `config.py` in each package is the single source of truth for that package's registry.

## Relationship to auto_recipe_creator

`deploy_vlms/` and `flask_api/` still exist there too — its `web_main.py` imports `flask_api`, and
the GPU server currently deploys from that checkout. **Until that deployment is repointed here,
the two copies can diverge; treat `auto_recipe_creator` as live and this repo as staging.**

The client-side registry (`poc/workflow_3/vlm/flask_vlm.py`) stays there and is deliberately
*not* duplicated here — it is kept separate from the server registry in
`flask_api/vlm_serve/config.py`.
