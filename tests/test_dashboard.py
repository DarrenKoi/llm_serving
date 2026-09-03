"""대시보드 라우트 테스트.

    pytest tests/test_dashboard.py
"""

from web_main import create_app


def _client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_dashboard_is_served_at_root():
    response = _client().get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["Content-Type"]
    assert "VLM 서빙 상태" in response.get_data(as_text=True)


def test_dashboard_has_no_external_resources():
    """오피스가 오프라인이다 - CDN 을 물면 페이지가 반쯤 죽은 채로 뜬다."""
    body = _client().get("/").get_data(as_text=True)

    # xmlns 같은 네임스페이스 URI 는 받아오지 않으므로 fetch 형태만 본다.
    for marker in ('src="http', "src='http", 'href="http', "href='http", "//cdn", "integrity="):
        assert marker not in body, f"외부 리소스로 보이는 것: {marker}"


def test_gpu_status_endpoint_is_mounted():
    payload = _client().get("/api/gpu_status").get_json()

    assert payload["service"] == "gpu_status"
    assert "available" in payload


def test_api_health_carries_gpu_status():
    payload = _client().get("/api/health").get_json()

    assert "gpu_status" in payload
    assert "available" in payload["gpu_status"]

