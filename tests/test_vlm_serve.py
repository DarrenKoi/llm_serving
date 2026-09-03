"""flask_api.vlm_serve proxy tests."""

import json
import logging
from pathlib import Path

import pytest
from flask import Flask
from requests import RequestException

from flask_api import register_flask_api
import flask_api.vlm_serve as vlm_serve
from flask_api.vlm_serve import logger as vlm_logger_module


class DummyResponse:
    """requests.Response 대체용 더미."""

    def __init__(
        self,
        status_code: int,
        body: bytes,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}
        self._chunks = chunks or [body]

    @property
    def content(self) -> bytes:
        return self._body

    def iter_content(self, chunk_size: int = 8192):
        yield from self._chunks

    def close(self):
        return None


class DummyHealthResponse:
    """health probe 용 requests.Response 대체 객체."""

    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _create_test_app() -> Flask:
    app = Flask(__name__)
    register_flask_api(app)
    return app


def _clear_vlm_file_handlers() -> None:
    root_logger = logging.getLogger(vlm_logger_module.LOGGER_NAME)
    for handler in list(root_logger.handlers):
        if getattr(handler, vlm_logger_module.FILE_HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()


def _fake_vlm_health_get(url: str, timeout: float):
    del timeout
    if url == "http://127.0.0.1:8002/v1/models":
        return DummyHealthResponse(200, {"data": [{"id": "mai-ui-8b"}]})
    if url == "http://127.0.0.1:8004/v1/models":
        return DummyHealthResponse(200, {"data": [{"id": "paddleocr-vl-1.5"}]})
    raise RequestException(f"connection refused: {url}")


@pytest.fixture(autouse=True)
def cleanup_vlm_file_handlers():
    _clear_vlm_file_handlers()
    yield
    _clear_vlm_file_handlers()


def test_vlm_serve_root_returns_live_health_payload(monkeypatch):
    monkeypatch.setattr("flask_api.vlm_serve.requests.get", _fake_vlm_health_get)

    app = _create_test_app()
    client = app.test_client()

    response = client.get("/api/vlm_serve/")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "vlm_serve"
    assert payload["status"] == "ok"
    assert payload["mode"] == "proxy"
    assert payload["base_path"] == "/api/vlm_serve"
    assert set(payload["registered_vlms"]) == {
        "mai-ui",
        "paddleocr-vl-1.5",
        "qwen3.8-27b",
    }
    assert set(payload["serving_routes"]) == {
        "mai-ui",
        "paddleocr-vl-1.5",
    }


def test_deploy_model_env_root_defaults_to_repo_deploy_vlms(monkeypatch):
    monkeypatch.delenv("CONFIG_ROOT", raising=False)
    monkeypatch.delenv("DEPLOY_VLMS_ROOT", raising=False)

    expected = (
        Path(__file__).resolve().parents[1]
        / "deploy_vlms"
        / "config"
        / "models"
    )

    assert vlm_serve._deploy_model_env_root() == expected


def test_models_proxy_uses_expected_upstream(monkeypatch):
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return DummyResponse(
            status_code=200,
            body=json.dumps({"data": [{"id": "mai-ui-8b"}]}).encode("utf-8"),
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    app = _create_test_app()
    client = app.test_client()

    response = client.get("/api/vlm_serve/mai-ui/v1/models")

    assert response.status_code == 200
    assert response.get_json()["data"][0]["id"] == "mai-ui-8b"
    assert captured["method"] == "GET"
    assert captured["url"] == "http://127.0.0.1:8002/v1/models"


def test_chat_proxy_injects_upstream_api_key(monkeypatch):
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return DummyResponse(
            status_code=200,
            body=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                            }
                        }
                    ]
                }
            ).encode("utf-8"),
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)
    monkeypatch.setenv("VLM_SERVE_UPSTREAM_API_KEY", "internal-key")

    app = _create_test_app()
    client = app.test_client()

    response = client.post(
        "/api/vlm_serve/mai-ui/v1/chat/completions",
        json={
            "model": "mai-ui-8b",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    assert response.status_code == 200
    assert response.get_json()["choices"][0]["message"]["content"] == "ok"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8002/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer internal-key"


def test_chat_proxy_logs_request_and_response_details(monkeypatch, caplog):
    def fake_request(**kwargs):
        return DummyResponse(
            status_code=200,
            body=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "analysis complete",
                            }
                        }
                    ]
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)
    monkeypatch.setenv("VLM_SERVE_UPSTREAM_API_KEY", "internal-key")

    app = _create_test_app()
    client = app.test_client()

    with caplog.at_level(logging.INFO, logger="flask_api.vlm_serve"):
        response = client.post(
            "/api/vlm_serve/mai-ui/v1/chat/completions",
            json={
                "model": "mai-ui-8b",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 200
    log_text = caplog.text
    assert "request service=mai-ui method=POST" in log_text
    assert "response service=mai-ui" in log_text
    assert "Bearer internal-key" not in log_text


def test_chat_proxy_logs_upstream_request_exception(monkeypatch, caplog):
    def fake_request(**kwargs):
        raise RequestException("connection refused")

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    app = _create_test_app()
    client = app.test_client()

    with caplog.at_level(logging.INFO, logger="flask_api.vlm_serve"):
        response = client.post(
            "/api/vlm_serve/mai-ui/v1/chat/completions",
            json={
                "model": "mai-ui-8b",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 502
    assert response.get_json()["message"] == "connection refused"
    assert "upstream failed service=mai-ui" in caplog.text
    assert "connection refused" in caplog.text


def test_streaming_chat_proxy_logs_stream_summary(monkeypatch, caplog):
    def fake_request(**kwargs):
        chunks = [
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
            b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        ]
        return DummyResponse(
            status_code=200,
            body=b"".join(chunks),
            headers={"Content-Type": "text/event-stream"},
            chunks=chunks,
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    app = _create_test_app()
    client = app.test_client()

    with caplog.at_level(logging.INFO, logger="flask_api.vlm_serve"):
        response = client.post(
            "/api/vlm_serve/qwen3.8-27b/v1/chat/completions",
            json={
                "model": "qwen3.8-27b",
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
    )

    assert response.status_code == 200
    assert b"delta" in response.data
    assert "request service=qwen3.8-27b method=POST" in caplog.text
    assert "response service=qwen3.8-27b" in caplog.text


def test_get_vlm_logger_creates_cloud_repo_log_dir(monkeypatch, tmp_path):
    cloud_repo_root = tmp_path / "cloud_repo"
    expected_log_path = cloud_repo_root / "logs" / "vlm_service" / "vlm_serve.log"

    monkeypatch.setenv("VLM_SERVE_REPO_ROOT", str(cloud_repo_root))
    monkeypatch.delenv("VLM_SERVE_LOG_DIR", raising=False)

    logger = vlm_logger_module.get_vlm_logger("proxy")
    logger.info("cloud repo log smoke test")

    root_logger = logging.getLogger(vlm_logger_module.LOGGER_NAME)
    file_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, vlm_logger_module.FILE_HANDLER_MARKER, False)
    ]

    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == expected_log_path
    assert expected_log_path.parent.is_dir()

    file_handlers[0].flush()
    assert "cloud repo log smoke test" in expected_log_path.read_text(encoding="utf-8")


def test_get_vlm_logger_reuses_existing_file_handler(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_SERVE_LOG_DIR", str(tmp_path / "logs" / "vlm_service"))

    vlm_logger_module.get_vlm_logger("proxy")
    vlm_logger_module.get_vlm_logger("proxy")

    root_logger = logging.getLogger(vlm_logger_module.LOGGER_NAME)
    file_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, vlm_logger_module.FILE_HANDLER_MARKER, False)
    ]

    assert len(file_handlers) == 1
