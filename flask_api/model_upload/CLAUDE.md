# flask_api/model_upload/ — resumable chunked upload

Three layers, deliberately: `store.py` (filesystem + resume state, **knows nothing about HTTP**),
`routes.py` (Flask wrapper), `config.py` (env wiring). That split is why resume and integrity are
testable without a running server — keep new logic in `store.py`.

Staging must stay **inside** the destination root so `os.replace` is atomic on one filesystem.
`MODEL_UPLOAD_ROOT` must be set for the Flask process: it does not read `common.env`, so
`ALLOWED_MODEL_ROOT` never reaches it. Runbook in `deploy_vlms/UPLOAD.md`, nginx body-size and
timeout blocks in `deploy_vlms/nginx/`.
