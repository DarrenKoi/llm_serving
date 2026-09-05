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


def _fake_vlm_health_get(url: str, timeout: float, headers=None):
    # headers 는 업스트림 API_KEY 가 설정됐을 때만 채워진다 (프로브 인증).
    del timeout, headers
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


def test_chat_proxy_replaces_caller_authorization_with_upstream_key(monkeypatch):
    """호출자가 자기 api_key 를 들고 와도 업스트림에는 업스트림 키가 가야 한다.

    프록시를 쓰는 쪽(auto_recipe_creator/workflow_3)은 업스트림 키를 모른다 - 그게
    프록시를 두는 이유다. 호출자 헤더를 그대로 넘기면 vLLM 이 --api-key 를 켠 순간
    401 이 나고, 증상은 "프록시가 죽었다"로 보인다. VLM_SERVE_TOKEN 없이도 성립해야 한다.
    """
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return DummyResponse(status_code=200, body=b'{"ok": true}')

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)
    monkeypatch.setenv("VLM_SERVE_UPSTREAM_API_KEY", "internal-key")
    monkeypatch.setenv("VLM_SERVE_TOKEN", "")

    app = _create_test_app()
    client = app.test_client()

    response = client.post(
        "/api/vlm_serve/mai-ui/v1/chat/completions",
        json={"model": "mai-ui-8b", "messages": [{"role": "user", "content": "ping"}]},
        headers={"Authorization": "Bearer caller-placeholder"},
    )

    assert response.status_code == 200
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


def test_get_vlm_logger_creates_log_dir(monkeypatch, tmp_path):
    expected_log_path = tmp_path / "logs" / "vlm_service" / "vlm_serve.log"

    monkeypatch.setenv("VLM_SERVE_LOG_DIR", str(expected_log_path.parent))

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


# ── 공용 토큰 인증 (VLM_SERVE_TOKEN) ──────────────────────────────────


def _ok_response(*_args, **kwargs):
    """프록시가 upstream 까지 갔는지 보기 위한 더미."""
    _CAPTURED.update(kwargs)
    return DummyResponse(
        status_code=200,
        body=json.dumps({"data": [{"id": "mai-ui-8b"}]}).encode("utf-8"),
    )


_CAPTURED: dict[str, object] = {}


@pytest.fixture
def proxy_client(monkeypatch):
    _CAPTURED.clear()
    monkeypatch.setattr(
        "flask_api.vlm_serve.service_template.requests.request", _ok_response
    )
    return _create_test_app().test_client()


def test_proxy_rejects_call_without_token_when_configured(proxy_client, monkeypatch):
    monkeypatch.setenv("VLM_SERVE_TOKEN", "team-secret")

    response = proxy_client.get("/api/vlm_serve/mai-ui/v1/models")

    assert response.status_code == 401
    assert response.get_json()["code"] == "Unauthorized"
    # upstream 까지 가지 않아야 한다.
    assert _CAPTURED == {}


def test_proxy_rejects_wrong_token(proxy_client, monkeypatch):
    monkeypatch.setenv("VLM_SERVE_TOKEN", "team-secret")

    response = proxy_client.get(
        "/api/vlm_serve/mai-ui/v1/models", headers={"X-VLM-Token": "nope"}
    )

    assert response.status_code == 401


def test_proxy_accepts_x_vlm_token(proxy_client, monkeypatch):
    monkeypatch.setenv("VLM_SERVE_TOKEN", "team-secret")

    response = proxy_client.get(
        "/api/vlm_serve/mai-ui/v1/models", headers={"X-VLM-Token": "team-secret"}
    )

    assert response.status_code == 200
    assert _CAPTURED["url"] == "http://127.0.0.1:8002/v1/models"


def test_proxy_accepts_bearer_token_from_openai_style_client(proxy_client, monkeypatch):
    """OpenAI 클라이언트는 api_key 를 Authorization 으로 보낸다."""
    monkeypatch.setenv("VLM_SERVE_TOKEN", "team-secret")

    response = proxy_client.get(
        "/api/vlm_serve/mai-ui/v1/models",
        headers={"Authorization": "Bearer team-secret"},
    )

    assert response.status_code == 200


def test_client_token_is_not_forwarded_upstream(proxy_client, monkeypatch):
    """공용 토큰은 우리 것이다 - upstream 으로 새면 vLLM API_KEY 와 충돌한다."""
    monkeypatch.setenv("VLM_SERVE_TOKEN", "team-secret")
    monkeypatch.setenv("VLM_SERVE_UPSTREAM_API_KEY", "internal-key")

    response = proxy_client.get(
        "/api/vlm_serve/mai-ui/v1/models",
        headers={"Authorization": "Bearer team-secret"},
    )

    assert response.status_code == 200
    # 호출자의 토큰이 아니라 upstream 용 키가 실려 나가야 한다.
    assert _CAPTURED["headers"]["Authorization"] == "Bearer internal-key"


def test_health_and_home_stay_open_when_token_is_set(proxy_client, monkeypatch):
    monkeypatch.setenv("VLM_SERVE_TOKEN", "team-secret")

    assert proxy_client.get("/api/vlm_serve/mai-ui/").status_code == 200
    assert proxy_client.get("/api/vlm_serve/mai-ui/health").status_code == 200


def test_proxy_stays_open_when_no_token_configured(proxy_client, monkeypatch):
    """토큰을 안 걸면 기존 동작 그대로여야 한다."""
    monkeypatch.delenv("VLM_SERVE_TOKEN", raising=False)

    response = proxy_client.get("/api/vlm_serve/mai-ui/v1/models")

    assert response.status_code == 200


def test_client_authorization_still_passes_through_when_auth_is_off(
    proxy_client, monkeypatch
):
    """인증을 안 켠 배포에서는 종전처럼 호출자 Authorization 이 그대로 간다."""
    monkeypatch.delenv("VLM_SERVE_TOKEN", raising=False)

    proxy_client.get(
        "/api/vlm_serve/mai-ui/v1/models",
        headers={"Authorization": "Bearer caller-own-key"},
    )

    assert _CAPTURED["headers"]["Authorization"] == "Bearer caller-own-key"


def test_base_url_override_works_for_dotted_slug(monkeypatch):
    """점이 든 slug 도 env override 가 먹어야 한다.

    `qwen3.8-27b` 의 점을 `_` 로 접지 않으면 `VLM_SERVE_QWEN3.8_27B_BASE_URL`
    이라는 shell 로 export 불가능한 키가 만들어지고, override 가 조용히 무시된다.
    """
    monkeypatch.setenv("VLM_SERVE_QWEN3_8_27B_BASE_URL", "http://127.0.0.1:8106")

    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return DummyResponse(
            status_code=200,
            body=json.dumps({"data": [{"id": "qwen3.8-27b"}]}).encode("utf-8"),
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    client = _create_test_app().test_client()
    response = client.get("/api/vlm_serve/qwen3.8-27b/v1/models")

    assert response.status_code == 200
    assert captured["url"] == "http://127.0.0.1:8106/v1/models"


def test_health_reports_same_base_url_as_proxy_uses(monkeypatch):
    """health 가 보고하는 주소와 프록시가 실제로 나가는 주소가 갈리면 안 된다."""
    monkeypatch.setenv("VLM_SERVE_QWEN3_8_27B_BASE_URL", "http://127.0.0.1:8106")

    from flask_api.vlm_serve import VLM_SERVICE_CONFIGS, _base_url_for_service

    SERVICE_CONFIG = VLM_SERVICE_CONFIGS["qwen3.8-27b"]
    assert SERVICE_CONFIG.upstream_base_url == "http://127.0.0.1:8106"
    assert _base_url_for_service("qwen3.8-27b", 8006) == SERVICE_CONFIG.upstream_base_url
