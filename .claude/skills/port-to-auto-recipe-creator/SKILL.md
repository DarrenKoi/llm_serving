---
name: port-to-auto-recipe-creator
description: Port a deploy_vlms/ or flask_api/ change from this repo to auto_recipe_creator. Use before copying any file between the two repos - carries the divergence table and the site.env / API_KEY expansion traps.
---

# Porting to auto_recipe_creator

Sync is one-way: changes are authored here, then ported there, per file. Never `cp -r`.
The two copies have diverged in *both* directions:

| Only here | Only in auto_recipe_creator |
|---|---|
| `flask_api/dashboard.py`, `gpu_status.py`, `templates/` | `gpu_dashboard/` (outside `flask_api`) |
| `flask_api/__init__.py`'s `load_site_env()` | `scripts/models/` |
| `config/site.env`, `site.env.example` | `scripts/prepare_variant_envs.py`, `start_*.py` shims, per-model `vlm_serve/<model>.py` (all deleted here 2026-09-05; delete there too when porting) |
| `scripts/diagnose_paths.py`, most `test_*.py` | |

Two of those are load-bearing traps when porting config:

- **auto_recipe_creator has no `site.env` and its `flask_api/__init__.py` does not call
  `load_site_env()`.** Its `common.env` hardcodes `ALLOWED_MODEL_ROOT` instead. So this repo's
  `API_KEY=${VLM_SERVE_UPSTREAM_API_KEY}` copied there stays an **unexpanded literal** — vLLM comes
  up demanding a key nobody can produce, and the Flask process never learns one either. Either port
  `load_site_env()` + create a `site.env` there, or write literal values into that repo's own
  `common.env`.
- Copying this repo's `flask_api/` wholesale also drops that app's `gpu_dashboard` registration in
  its `web_main.py`; port `flask_api/` files one at a time.
- Its `poc/workflow_2/docs/study/paddleOCR/README.md` still names `start_paddleocr_vl.py`; fix that
  line when porting the shim deletion.
- Its `test/flask_api/test_ui_tars_stream_proxy.py` tests the `force_stream` workaround that was
  deleted here on 2026-09-05 (already failing there since ui-tars left); delete it with the port.

The client-side registry (`poc/workflow_3/vlm/flask_vlm.py`) stays there and is deliberately
*not* duplicated here — it is kept separate from the server registry in
`flask_api/vlm_serve/config.py`.
