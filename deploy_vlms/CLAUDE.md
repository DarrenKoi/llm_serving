# deploy_vlms/ — process launcher

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
4. Validate (absolute path, dir exists, under `ALLOWED_MODEL_ROOT`, `config.json` present), then
   exec.

`start_all.py` starts models **strictly one at a time**, largest first (`VLLM_MODELS` order,
`qwen3.8-27b` leading), waiting for each to answer `/v1/models` before the next. This is a host-RAM
guarantee, not a convenience: weight loading and CUDA-graph capture are each model's RSS peak, and
overlapping those peaks on a 16 GB swapless host invites the OOM killer. If a model times out while
still alive it is stopped before the next one starts — `deploy_vlms/scripts/test_start_all.py`
pins that. There are no per-model wrapper scripts: `python start_model.py <slug>` is the only
single-model entry point.

`start_model.py` backgrounds it and writes `deploy_vlms/runtime/{logs,pids}/<instance>.{log,pid}`
(gitignored). `stop_model.py` resolves a target by instance name → PID file → falling back to
whoever holds the port. Each launch **rotates** the instance log: the previous run moves to
`<instance>.log.<timestamp>` and only `LOG_KEEP` (5) rotated files survive, so `<instance>.log`
is always the current run and a crash's tail is never lost. `SERVE_VLM_DRY_RUN=1` makes
`serve_vlm.py` run every check and print the vLLM command without starting it; `diagnose_paths.py`
uses that to show the launcher's own view of the config next to an in-process check.

`GPU_MEMORY_UTILIZATION` is always an explicit per-model number; the launcher does not estimate it
(the auto-tune engine was removed 2026-09-05 — it assumed every layer holds KV, which is ~4× wrong
for `qwen3.8-27b`, where 48 of 64 layers are GatedDeltaNet). The per-model `.env` files carry long
comments explaining *why* each value is what it is — read them before changing a number.
