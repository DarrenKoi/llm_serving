# CLAUDE.md

## What this is

vLLM serving stack + Flask API proxy for an in-house GPU server (H200 × 2). Split out of
`auto_recipe_creator` on 2026-09-04.

## Setup and commands

```bash
pip install -e ".[dev]"

# Site values (MODEL_ROOT, VLLM_API_KEY) live inside deploy_vlms/ so they travel with a folder copy.
# serve_vlm.py and flask_api read the file themselves; a shell export wins over it:
cp deploy_vlms/config/site.env.example deploy_vlms/config/site.env   # then fill it in
```

**pip is the toolchain here, not `uv`** — it is the company standard and `uv` has known issues in
this environment. Use plain `python` / `pytest`; never add a `uv run` prefix to a command or doc.
The GPU server runs vLLM 0.19.1 on Python 3.11.9 (rolled back from 0.28.0 on 2026-09-04; see
Hard constraints for why).

```bash
# tests — no GPU, no server; these run on a laptop
pytest
pytest deploy_vlms/scripts/test_serve_vlm_env.py       # one file
pytest tests/test_vlm_serve.py::test_models_proxy_uses_expected_upstream   # one test

# bring up / tear down (GPU server only)
python deploy_vlms/scripts/start_all.py
python deploy_vlms/scripts/start_model.py qwen3.8-27b
python deploy_vlms/scripts/stop_model.py all
python deploy_vlms/scripts/check_vlm.py http://127.0.0.1:8006 qwen3.8-27b

# health checks
python deploy_vlms/scripts/check_host_ram.py     # run warm, after models are up
python deploy_vlms/scripts/check_kv_longctx.py   # 5-10 min, run when idle
python deploy_vlms/scripts/diagnose_paths.py     # why MODEL_ROOT/MODEL_ID does not resolve; dry-runs the launcher

# qwen3.8-27b thinking / effort / budget experiments — hit vLLM directly; knobs and traps in scripts/README.md
python scripts/effort_ladder.py
```

Tests live in **two places**: `tests/` (Flask proxy) and beside the code they cover
(`flask_api/model_upload/test_*.py`, `deploy_vlms/scripts/test_*.py`, `scripts/test_*.py`). `pytest`
finds them via `pythonpath = [".", "scripts"]`. There is no linter or formatter configured — don't invent one.

## Architecture

Two halves that **never import each other**. They meet only through
`deploy_vlms/config/`: `site.env` (`MODEL_ROOT` and the one `VLLM_API_KEY`, read by both) and `models/*.env`, which the launcher consumes as config and the Flask health
probe reads at runtime to discover what should be running.

### deploy_vlms/ — process launcher

Launcher internals — the env resolution chain, strict one-at-a-time start order, log rotation, and
why `GPU_MEMORY_UTILIZATION` is always explicit — live in `deploy_vlms/CLAUDE.md`, loaded when
working under `deploy_vlms/`.

### flask_api/ — reverse proxy, mounted at `/api`

`index.py` (WSGI target, exposes `application`) → `web_main.py` (builds `app`) →
`register_flask_api(app)`. `web_main.py` does nothing but mount the package; put routes in
`flask_api/`, never in `web_main.py`. Locally: `python index.py` serves on :5000.

Importing `flask_api` runs `load_site_env()`, which copies `deploy_vlms/config/site.env` into
`os.environ` (fill-only, existing keys win) **before** `model_upload` snapshots its config. That is
why the GPU server, which receives only `deploy_vlms/` and `flask_api/` copied by hand with no git
and no shell export, still gets `MODEL_ROOT` and `VLLM_API_KEY`. `SITE_ENV` overrides the path; the
root `conftest.py` points it at `os.devnull` so tests never read a developer's real file.

The proxy relays SSE (`text/event-stream`) upstream responses chunk by chunk and buffers everything
else, returning it verbatim (`service_template.py`). Coding harnesses (opencode, pi) can therefore
stream through `/api/vlm_serve/<slug>/v1` when port 8006 is not reachable from the client.
`vlm_serve/__init__.py` loops over the registry and builds one proxy blueprint per entry with `create_vlm_service_blueprint`; there are no per-model files.

**A model's slug is declared in two places and they must agree:**

| Place | Declares |
|---|---|
| `flask_api/vlm_serve/config.py` → `VLM_SERVICES` | slug, display name, upstream port |
| `deploy_vlms/config/models/<slug>.env` | `PORT` for the actual vLLM process |

The `.env` **filename stem is the slug** — `_configured_vlm_entries()` globs the directory and
uses `env_path.stem`. Renaming the file silently unregisters the service from health reporting.
Adding a model means one line in `config.py` plus the `.env`.

There is no `enabled` flag. Deleted models are removed outright — a disabled entry would falsely
imply flipping a flag could revive it, when the checkpoint is actually gone from the server.

`VLLM_API_KEY` (in `site.env`) is the **one key** for everything: vLLM's `--api-key`, the check on
`/api/vlm_serve/<slug>/v1/*` (a `before_request` on each service blueprint), and `/api/model_upload`.
Empty means auth is **off** everywhere. `home` and `health` stay open for monitoring. Callers may
send `X-VLM-Token` or `Authorization: Bearer` (the OpenAI client uses the latter); the proxy strips
the caller's `Authorization` and injects `Bearer $VLLM_API_KEY` upstream, so an `X-VLM-Token`
caller still reaches vLLM authenticated.

`/api/health` is a **reconciliation, not a stored flag**: it merges declared proxies with
discovered `.env` files, live-probes each upstream `/v1/models`, and reports `serving_mismatch`
when the served model name disagrees with what was configured.

### Dashboard at `/` and GPU telemetry

`flask_api/gpu_status.py` shells to `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
A missing or failing `nvidia-smi` is a
**state, not an error** — it returns `{"available": false, "reason": ...}` so the page still renders
on a laptop. `[N/A]` fields become `None`, never `0`, so a graph never shows a confident wrong value.

`uwsgi.ini` has **no `env =` lines**: Flask gets `site.env` through `load_site_env()`, which is
fill-only, so an `env =` line would silently override `site.env`. `deploy_vlms/config/common.env` is
read by the launcher, not by Flask. Timeouts nest outward:
nginx > `harakiri` (870) > `VLM_SERVE_READ_TIMEOUT_SEC` (300) — invert that and the app succeeds
while the caller sees a 504.

`flask_api/dashboard.py` serves `templates/dashboard.html` at `/` — registered by
`register_dashboard(app)`, separate from `register_flask_api(app)` because the latter's contract is
"everything under `/api`". The page polls **`/api/health` only**: that one payload already carries
`vlm_serve`, `gpu_status`, and `model_upload`, so no dashboard-specific endpoint exists.

**The page must stay dependency-free** — the office is offline, so a CDN link renders a half-dead
page. All CSS/JS is inline and `tests/test_dashboard.py` fails the build if a `src="http`/`href="http`
sneaks in. Models are grouped by the `gpu_id` that `vlm_serve` reads from `config/models/*.env`.

### flask_api/model_upload/ — resumable chunked upload

Layering (`store.py` knows nothing about HTTP) and the staging-inside-destination rule live in
`flask_api/model_upload/CLAUDE.md`. Runbook in `deploy_vlms/UPLOAD.md`, nginx body-size and
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
- **Host NVIDIA driver is the 570 branch (CUDA 12.8).** vLLM 0.20.0 switched its PyPI default wheel
  to CUDA 13.0, and that build dies at torch init with "NVIDIA driver … too old (found version
  12080)" for every model, before any weight loads. Server runs **vLLM 0.19.1 / Python 3.11.9**,
  the last PyPI default built for CUDA 12.9 (rolled back from 0.28.0 on 2026-09-04). To go newer
  without a host driver change: the `+cu129` wheel from the GitHub release with cu129 torch, or
  NVIDIA's `cuda-compat-13-0` package on `LD_LIBRARY_PATH`. A 580-branch driver removes the limit.
- `--swap-space` was removed with the V1 engine and caused a startup failure here (2026-08-11).
- MTP speculative decoding is off for `qwen3.8-27b` on purpose: it crashed above ~26k tokens on
  0.19.1 (vllm#40756), which is the version the server is on again, so the crash is live — do not
  enable it without checking the issue and re-running `check_kv_longctx.py`. Long context is the
  requirement; MTP is only an optimization.

## Conventions

- **No CLI arguments.** Configuration lives in `config/*.env` and in the constant block at the top
  of each entry-point script.
- Korean docstrings and comments. `deploy_vlms/` scripts print `[INFO]` / `[WARNING]` / `[ERROR]`
  rather than using `logging`; `flask_api/vlm_serve/` goes through `logger.py`
  (`get_vlm_logger`).
- `config.py` in each package is the single source of truth for that package's registry.
- Client-side reasoning knobs go in `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`
  ∈ low/medium/xhigh). The top-level `reasoning_effort` field is a trap on vLLM 0.19.1 with this
  model (`xhigh` → 400, `high` → template exception → 500). `scripts/qwen_client.py` encodes this.
  The server default is **medium**, set by `--default-chat-template-kwargs` in `qwen3.8-27b.env`
  (the chat template's own default is xhigh, which makes coding harnesses that send nothing think
  for minutes). A request's own `chat_template_kwargs` still wins over it.
- **`VLLM_API_KEY` is set and `HOST` is `0.0.0.0`.** vLLM therefore rejects unauthenticated `/v1/*`
  (`/health` stays open). Anything that probes an upstream must send
  `Authorization: Bearer <VLLM_API_KEY>` — `check_vlm.upstream_api_key()` and
  `qwen_client.API_KEY` are the two places that resolve it. A missing header does not surface as
  401: `start_all.py` reads it as "not up yet" and stops the model.

## Relationship to auto_recipe_creator

**auto_recipe_creator hosts the deployed Flask app. Do not move it.** Its `web_main.py` registers
`gpu_dashboard_dp` at `/gpu-dashboard` *and* `register_flask_api(app)`, and `poc/workflow_3` calls
that deployment's `/api` for all three models. Live work depends on it.

**This repo is where `deploy_vlms/` and `flask_api/` are authored** (decided 2026-09-05). Sync is
one-way: changes are made here, then ported there. Never the reverse, or there is no rule about
which copy wins.

**Porting is per-file, never `cp -r`.** The two copies have diverged in *both* directions.
The divergence table and the config trap (`site.env` / `load_site_env()`) that bites on copy
are in the `port-to-auto-recipe-creator` skill — load it before porting anything.

**Deployment lever:** `uwsgi.ini` on the GPU server, and only that — there are no systemd rights.
`deploy_vlms/uwsgi/uwsgi.ini` is the template (three `[SET_ME]` values: `chdir`, `virtualenv`, `logto`); `deploy_vlms/nginx/` holds
the two location blocks it pairs with. Any restart or repoint advice has to fit inside editing
that one file.
