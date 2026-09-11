---
name: port-to-auto-recipe-creator
description: Port a deploy_vlms/ or flask_api/ change from this repo to auto_recipe_creator. Use before copying any file between the two repos - carries the divergence table and the site.env / VLLM_API_KEY / ${MODEL_ROOT} traps.
---

# Porting to auto_recipe_creator

Sync is one-way: changes are authored here, then ported there, per file. Never `cp -r`.
The two copies have diverged in *both* directions:

| Only here | Only in auto_recipe_creator |
|---|---|
| `flask_api/dashboard.py`, `gpu_status.py`, `templates/` | `gpu_dashboard/` (outside `flask_api`) |
| `deploy_vlms/uwsgi/uwsgi.ini` template | `scripts/models/` |
| `scripts/diagnose_paths.py`, most `test_*.py` | `scripts/prepare_variant_envs.py`, `start_*.py` shims, per-model `vlm_serve/<model>.py` (all deleted here 2026-09-05; delete there too when porting) |
| Launcher that expands `${MODEL_ROOT}` (`serve_vlm.py`, `common.env`, `models/*.env`) | Old auto-tune launcher: `common.env` hardcodes `ALLOWED_MODEL_ROOT` and per-knob defaults, `HOST=127.0.0.1` |

Shared since 2026-09-11: one `VLLM_API_KEY` in `deploy_vlms/config/site.env`, read by
`flask_api/__init__.py`'s `load_site_env()`, `serve_vlm.py`, `check_vlm.py`, the proxy
(`service_template.py`), and `model_upload/config.py` (upload root = `MODEL_ROOT`).

Load-bearing traps when porting:

- **auto_recipe_creator's launcher does not `expandvars`.** Its `load_env_file` assigns values
  literally, so a `models/*.env` or `common.env` from here that says `${MODEL_ROOT}/...` stays a
  literal there and trips the `MODEL_ID` `isabs` check. Port those `.env` files only together with
  this repo's `serve_vlm.py`.
- **`site.env` must stay gitignored there.** Its `.gitignore` only had `.env`, which does not match
  `site.env`; an explicit `deploy_vlms/config/site.env` line was added. Keep it.
- **workflow_3 must send the key before the server sets it.** `poc/workflow_3/vlm/flask_vlm.py`'s
  `resolve_service_api_key` returns `VLLM_API_KEY` for proxy-mode services. Once the server's
  `site.env` has a key, a client without it gets 401 on every `/api/vlm_serve` call.
- **No `env =` lines in the server's `uwsgi.ini`** for `VLLM_API_KEY` / `MODEL_ROOT`:
  `load_site_env()` is fill-only, so those lines silently beat `site.env`.
- Copying this repo's `flask_api/` wholesale also drops that app's `gpu_dashboard` registration in
  its `web_main.py`; port `flask_api/` files one at a time.
- Its `poc/workflow_2/docs/study/paddleOCR/README.md` still names `start_paddleocr_vl.py`; fix that
  line when porting the shim deletion.
- Its `test/flask_api/test_ui_tars_stream_proxy.py` tests the `force_stream` workaround that was
  deleted here on 2026-09-05 (already failing there since ui-tars left); delete it with the port.

The client-side registry (`poc/workflow_3/vlm/flask_vlm.py`) stays there and is deliberately
*not* duplicated here — it is kept separate from the server registry in
`flask_api/vlm_serve/config.py`.
