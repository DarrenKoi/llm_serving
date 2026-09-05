# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

vLLM serving stack + Flask API proxy for an in-house GPU server (H200 × 2). Split out of
`auto_recipe_creator` on 2026-09-04.

**The repo is public.** All site-specific values are placeholders (`${MODEL_ROOT}`,
`vlm-host.internal`); real paths and secrets live in the gitignored
`deploy_vlms/config/site.env`. Never commit an
internal hostname or absolute model path — the history was scrubbed once already to remove them.

`README.md` and `docs/README.md` are **deliberately empty** (emptied 2026-09-04 to reduce search
indexing). Do not repopulate them without being asked. Their content is recoverable from git history.

## Setup and commands

```bash
pip install -e ".[dev]"

# Site values (MODEL_ROOT, tokens) live inside deploy_vlms/ so they travel with a folder copy.
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
`deploy_vlms/config/`: `site.env` (site paths and tokens, read by both) and `models/*.env`, which the launcher consumes as config and the Flask health
probe reads at runtime to discover what should be running.

### deploy_vlms/ — process launcher

`serve_vlm.py <instance>` is the core. It never imports vLLM; it builds an argv and calls
`os.execvpe`, replacing itself with `vllm serve`. The resolution chain:

1. Load `config/site.env` if present, then `config/common.env`, then
   `config/models/<instance>.env`.
2. `load_env_file` **assigns unconditionally** (`os.environ[key] = value`), so config files win
   over the shell. Exporting `PORT=…` does *not* override a model's config. `site.env` is the
   exception: it is loaded with `override=False`, so a shell export beats it.
3. Values are `os.path.expandvars`-expanded, so `${MODEL_ROOT}` from `site.env` flows in. This is the
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
whoever holds the port. Each launch **rotates** the instance log: the previous run moves to
`<instance>.log.<timestamp>` and only `LOG_KEEP` (5) rotated files survive, so `<instance>.log`
is always the current run and a crash's tail is never lost. `SERVE_VLM_DRY_RUN=1` makes
`serve_vlm.py` run every check and print the vLLM command without starting it; `diagnose_paths.py`
uses that to show the launcher's own view of the config next to an in-process check.

`AUTO_TUNE_GPU_MEMORY_UTILIZATION` assumes every layer holds KV. Do not enable it for
`qwen3.8-27b`: 48 of its 64 layers are GatedDeltaNet (linear attention, fixed-size state), so
auto-tune overestimates KV ~4× and refuses to start. The per-model `.env` files carry long
comments explaining *why* each value is what it is — read them before changing a number.

### flask_api/ — reverse proxy, mounted at `/api`

`index.py` (WSGI target, exposes `application`) → `web_main.py` (builds `app`) →
`register_flask_api(app)`. `web_main.py` does nothing but mount the package; put routes in
`flask_api/`, never in `web_main.py`. Locally: `python index.py` serves on :5000.

Importing `flask_api` runs `load_site_env()`, which copies `deploy_vlms/config/site.env` into
`os.environ` (fill-only, existing keys win) **before** `model_upload` snapshots its config. That is
why the GPU server, which receives only `deploy_vlms/` and `flask_api/` copied by hand with no git
and no shell export, still gets `MODEL_ROOT` and the tokens. `SITE_ENV` overrides the path; the
root `conftest.py` points it at `os.devnull` so tests never read a developer's real file.

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

`VLM_SERVE_TOKEN` (in `site.env`) gates `/api/vlm_serve/<slug>/v1/*` with one shared team token,
checked in a `before_request` on each service blueprint. Empty means auth is **off**, so setting it
is opt-in and an existing deployment never breaks silently. `home` and `health` stay open for
monitoring. Callers may send `X-VLM-Token` or `Authorization: Bearer` (the OpenAI client uses the
latter) — and when the token is configured the caller's `Authorization` is **stripped before
forwarding**, so it cannot collide with the upstream `API_KEY` the proxy injects.

`/api/health` is a **reconciliation, not a stored flag**: it merges declared proxies with
discovered `.env` files, live-probes each upstream `/v1/models`, and reports `serving_mismatch`
when the served model name disagrees with what was configured.

### Dashboard at `/` and GPU telemetry

`flask_api/gpu_status.py` shells to `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
(same pattern as `serve_vlm.detect_gpu_total_memory_gib`). A missing or failing `nvidia-smi` is a
**state, not an error** — it returns `{"available": false, "reason": ...}` so the page still renders
on a laptop. `[N/A]` fields become `None`, never `0`, so a graph never shows a confident wrong value.

`flask_api/dashboard.py` serves `templates/dashboard.html` at `/` — registered by
`register_dashboard(app)`, separate from `register_flask_api(app)` because the latter's contract is
"everything under `/api`". The page polls **`/api/health` only**: that one payload already carries
`vlm_serve`, `gpu_status`, and `model_upload`, so no dashboard-specific endpoint exists.

**The page must stay dependency-free** — the office is offline, so a CDN link renders a half-dead
page. All CSS/JS is inline and `tests/test_dashboard.py` fails the build if a `src="http`/`href="http`
sneaks in. Models are grouped by the `gpu_id` that `vlm_serve` reads from `config/models/*.env`.

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
  (`get_vlm_logger`), and its route stubs keep one-line English docstrings.
- `config.py` in each package is the single source of truth for that package's registry.
- Client-side reasoning knobs go in `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`
  ∈ low/medium/xhigh). The top-level `reasoning_effort` field is a trap on vLLM 0.19.1 with this
  model (`xhigh` → 400, `high` → template exception → 500). `scripts/qwen_client.py` encodes this.
  The server default is **medium**, set by `--default-chat-template-kwargs` in `qwen3.8-27b.env`
  (the chat template's own default is xhigh, which makes coding harnesses that send nothing think
  for minutes). A request's own `chat_template_kwargs` still wins over it.
- **`API_KEY` is on and `HOST` is `0.0.0.0`.** vLLM therefore rejects unauthenticated `/v1/*`
  (`/health` stays open). Anything that probes an upstream must send
  `Authorization: Bearer <VLM_SERVE_UPSTREAM_API_KEY>` — `check_vlm.upstream_api_key()` and
  `qwen_client.API_KEY` are the two places that resolve it. A missing header does not surface as
  401: `start_all.py` reads it as "not up yet" and stops the model.

## Relationship to auto_recipe_creator

**auto_recipe_creator hosts the deployed Flask app. Do not move it.** Its `web_main.py` registers
`gpu_dashboard_dp` at `/gpu-dashboard` *and* `register_flask_api(app)`, and `poc/workflow_3` calls
that deployment's `/api` for all three models. Live work depends on it.

**This repo is where `deploy_vlms/` and `flask_api/` are authored** (decided 2026-09-05). Sync is
one-way: changes are made here, then ported there. Never the reverse, or there is no rule about
which copy wins.

**Porting is per-file, never `cp -r`.** The two copies have diverged in *both* directions:

| Only here | Only in auto_recipe_creator |
|---|---|
| `flask_api/dashboard.py`, `gpu_status.py`, `templates/` | `flask_api/vlm_serve/mai_ui_2b.py` |
| `flask_api/__init__.py`'s `load_site_env()` | `gpu_dashboard/` (outside `flask_api`) |
| `config/site.env`, `site.env.example` | `config/models/mai-ui-2b.env`, `scripts/models/` |
| `scripts/diagnose_paths.py`, most `test_*.py` | |

Two of those are load-bearing traps when porting config:

- **auto_recipe_creator has no `site.env` and its `flask_api/__init__.py` does not call
  `load_site_env()`.** Its `common.env` hardcodes `ALLOWED_MODEL_ROOT` instead. So this repo's
  `API_KEY=${VLM_SERVE_UPSTREAM_API_KEY}` copied there stays an **unexpanded literal** — vLLM comes
  up demanding a key nobody can produce, and the Flask process never learns one either. Either port
  `load_site_env()` + create a `site.env` there, or write literal values into that repo's own
  `common.env`.
- Copying this repo's `flask_api/` wholesale drops `mai_ui_2b` and can break that app's imports.

The client-side registry (`poc/workflow_3/vlm/flask_vlm.py`) stays there and is deliberately
*not* duplicated here — it is kept separate from the server registry in
`flask_api/vlm_serve/config.py`.

**Deployment lever:** `uwsgi.ini` on the GPU server, and only that — there are no systemd rights,
and the file is version-controlled in neither repo. Any restart or repoint advice has to fit
inside editing that one file.
