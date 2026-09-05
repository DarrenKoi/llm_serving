# Repository Guidelines

## Project Structure & Module Organization

`flask_api/` contains the Flask reverse proxy, dashboard, GPU status, and resumable model-upload code. `deploy_vlms/` holds GPU-host launchers, model environment files, nginx configuration, and operational checks. Put reusable client experiments in `scripts/`, longer operational notes in `docs/`, and top-level integration tests in `tests/`. Tests that closely follow one module live beside it as `test_*.py`. `index.py` is the WSGI entry point; keep application routes inside `flask_api/`.

## Setup, Test, and Development Commands

- `pip install -e ".[dev]"` installs the package and pytest. Use `pip`, not `uv`.
- `pytest` runs the laptop-safe suite; no GPU or live model server is required.
- `pytest tests/test_dashboard.py` runs one test module; append `::test_name` for one case.
- `python index.py` starts the local Flask app on port 5000.
- `python deploy_vlms/scripts/start_all.py` starts configured models sequentially on the GPU host.
- `python deploy_vlms/scripts/diagnose_paths.py` validates model-path configuration without launching vLLM.

## Coding Style & Naming Conventions

Target Python 3.10+ and use four-space indentation, type hints, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Follow existing concise Korean comments and docstrings; service route stubs use one-line English docstrings. Launcher output uses `[INFO]`, `[WARNING]`, and `[ERROR]`; proxy logging goes through `flask_api/vlm_serve/logger.py`. No formatter or linter is configured, so match nearby code and keep imports grouped.

## Testing Guidelines

Pytest discovers `test_*.py` in `tests/`, `flask_api/`, `deploy_vlms/scripts/`, and `scripts/`. Add the smallest regression test near the changed behavior. Keep default tests offline, deterministic, and independent of NVIDIA hardware. Run `pytest` before opening a pull request.

## Commit & Pull Request Guidelines

Recent commits use Conventional Commit-style subjects such as `feat(model_upload): ...`, `fix(vlm_serve): ...`, and `docs(study): ...`. Keep each commit focused. Pull requests should explain the operational impact, list tests run, link the issue when applicable, and include screenshots for dashboard changes. Call out any GPU-host-only verification still pending.

## Security & Configuration

Copy `deploy_vlms/config/site.env.example` to the gitignored `site.env` for local site values. Never commit tokens, internal hostnames, or absolute model paths. The office is offline: avoid runtime CDN or model-download dependencies.
