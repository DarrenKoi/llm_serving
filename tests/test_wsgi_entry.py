"""WSGI 진입점 배선 테스트.

index.py 가 WSGI 에 넘기는 이름(application)과 web_main 의 app 이 어긋나면
배포가 import 단계에서 죽는다. 이름 하나짜리 실수라 값싸게 고정해둔다.

    pytest tests/test_wsgi_entry.py
"""

import index
import web_main


def test_index_exposes_application_for_wsgi():
    """WSGI 서버가 찾는 이름은 application 이다."""
    assert index.application is web_main.app


def test_api_is_mounted_on_the_wsgi_app():
    """application 을 통해 /api/health 에 실제로 닿는다."""
    web_main.app.config["TESTING"] = True

    response = index.application.test_client().get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["service"] == "api"
