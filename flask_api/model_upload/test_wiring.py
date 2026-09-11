"""flask_api 앱에 업로드 엔드포인트가 실제로 붙었는지 확인한다."""

from pathlib import Path

from flask import Flask

from flask_api import register_flask_api
from flask_api.model_upload.config import load_upload_config


def _client():
    """api blueprint 만 실은 앱의 test client 를 만든다."""
    app = Flask(__name__)
    register_flask_api(app)
    app.config["TESTING"] = True
    return app.test_client()


def test_model_upload_health_is_mounted():
    """/api/model_upload/health 가 열려 있다."""
    response = _client().get("/api/model_upload/health")

    assert response.status_code == 200
    assert response.get_json()["service"] == "model_upload"


def test_api_health_advertises_model_upload():
    """/api/health 가 업로드 엔드포인트를 알린다(오피스에서 도달성 확인용)."""
    payload = _client().get("/api/health").get_json()

    assert "model_upload" in payload


def test_upload_follows_model_root_and_the_one_key(monkeypatch):
    """업로드 목적지는 MODEL_ROOT, 토큰은 VLLM_API_KEY - 서빙과 따로 맞출 값이 없다.

    이름이 틀리면 증상이 조용하다: 토큰이 빈 문자열이 되어 업로드 인증이 그냥 꺼진다.
    """
    monkeypatch.setenv("MODEL_ROOT", "/srv/models")
    monkeypatch.setenv("VLLM_API_KEY", "team-key")

    config = load_upload_config()

    assert config.dest_root == Path("/srv/models")
    assert config.token == "team-key"
