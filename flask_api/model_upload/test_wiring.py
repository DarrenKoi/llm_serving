"""flask_api 앱에 업로드 엔드포인트가 실제로 붙었는지 확인한다."""

from flask import Flask

from flask_api import register_flask_api


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
