"""실제 HTTP 왕복으로 클라이언트<->서버 계약을 검증한다.

로컬에 진짜 werkzeug 서버를 띄우고 requests 로 올린다:
    uv run pytest deploy_vlms/scripts/test_upload_model_e2e.py
"""

import importlib.util
import threading
from pathlib import Path

import pytest
from flask import Flask
from werkzeug.serving import make_server

from flask_api.model_upload.routes import create_model_upload_blueprint
from flask_api.model_upload.store import UploadStore

_SPEC = importlib.util.spec_from_file_location(
    "upload_model_e2e", Path(__file__).with_name("upload_model.py")
)
upload_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(upload_model)

TOKEN = "s3cret"


@pytest.fixture()
def server(tmp_path):
    """임시 목적지를 가진 업로드 서버를 실제 포트에 띄운다."""
    app = Flask(__name__)
    store = UploadStore(dest_root=tmp_path / "models", staging_root=tmp_path / "staging")
    app.register_blueprint(
        create_model_upload_blueprint(
            store=store, token=TOKEN, max_chunk_bytes=1024 * 1024
        ),
        url_prefix="/api/model_upload",
    )
    http_server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http_server.server_port}", tmp_path
    finally:
        http_server.shutdown()
        thread.join(timeout=5)


def _source_file(tmp_path, name="model-00001.safetensors", repeat=700):
    """여러 청크로 나뉘는 소스 파일을 만든다."""
    source = tmp_path / "src"
    source.mkdir(exist_ok=True)
    path = source / name
    path.write_bytes(bytes(range(256)) * repeat)
    return path


def test_real_http_roundtrip_reproduces_file(server):
    """실제 HTTP 로 올린 파일이 서버 디스크에서 바이트 동일하다."""
    base_url, tmp_path = server
    local = _source_file(tmp_path)
    transport = upload_model.HttpTransport(base_url=base_url, token=TOKEN)
    job = upload_model.plan_jobs(local, dest_prefix="MAI-UI-8B")[0]

    result = upload_model.upload_file(transport, job, chunk_size=64 * 1024)

    assert result.skipped is False
    placed = tmp_path / "models" / "MAI-UI-8B" / "model-00001.safetensors"
    assert placed.read_bytes() == local.read_bytes()


def test_wrong_token_fails_fast_without_retrying(server):
    """토큰이 틀리면 재시도하지 않고 즉시 실패한다."""
    base_url, tmp_path = server
    local = _source_file(tmp_path, repeat=4)
    transport = upload_model.HttpTransport(base_url=base_url, token="wrong")
    job = upload_model.plan_jobs(local, dest_prefix="MAI-UI-8B")[0]

    with pytest.raises(upload_model.UploadFailed) as excinfo:
        upload_model.upload_file(transport, job, chunk_size=64 * 1024)

    assert excinfo.value.status_code == 401


def test_resume_over_real_http_finishes_the_file(server):
    """절반만 올라간 상태에서 다시 실행하면 나머지만 올리고 끝난다."""
    base_url, tmp_path = server
    local = _source_file(tmp_path)
    transport = upload_model.HttpTransport(base_url=base_url, token=TOKEN)
    job = upload_model.plan_jobs(local, dest_prefix="MAI-UI-8B")[0]

    class _StopsHalfway(Exception):
        pass

    original_put = transport.put_chunk
    calls = {"n": 0}

    def _put(upload_id, offset, data, chunk_sha256):
        if calls["n"] >= 2:
            raise _StopsHalfway()
        calls["n"] += 1
        return original_put(upload_id, offset, data, chunk_sha256)

    transport.put_chunk = _put
    with pytest.raises(_StopsHalfway):
        upload_model.upload_file(transport, job, chunk_size=64 * 1024)

    transport.put_chunk = original_put
    upload_model.upload_file(transport, job, chunk_size=64 * 1024)

    placed = tmp_path / "models" / "MAI-UI-8B" / "model-00001.safetensors"
    assert placed.read_bytes() == local.read_bytes()
