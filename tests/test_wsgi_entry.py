"""WSGI 진입점 배선 테스트.

index.py 가 WSGI 에 넘기는 이름(application)과 web_main 의 app 이 어긋나면
배포가 import 단계에서 죽는다. 이름 하나짜리 실수라 값싸게 고정해둔다.

    pytest tests/test_wsgi_entry.py
"""

import os

import pytest

import web_main


@pytest.fixture
def index(monkeypatch):
    """index 는 import 시점에 저장소 루트 .env 를 os.environ 에 얹는다. 이 PC 의 실제
    .env 가 다른 테스트로 새지 않도록 환경을 통째로 갈아끼운 채 import 한다."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    import index
    return index


def test_index_exposes_application_for_wsgi(index):
    """WSGI 서버가 찾는 이름은 application 이다."""
    assert index.application is web_main.app


def test_api_is_mounted_on_the_wsgi_app(index):
    """application 을 통해 /api/health 에 실제로 닿는다."""
    web_main.app.config["TESTING"] = True

    response = index.application.test_client().get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["service"] == "api"


def test_dotenv_fills_gaps_but_never_overrides_shell(tmp_path, index):
    """.env 는 빈 자리만 채운다 - WSGI/systemd 가 준 환경이 우선. ${VAR} 는 최종 환경으로 펼친다."""
    (tmp_path / ".env").write_text(
        'MODEL_ROOT=/file\nMODEL_UPLOAD_ROOT=${MODEL_ROOT}\nVLM_SERVE_TOKEN="t"\n'
    )
    os.environ["MODEL_ROOT"] = "/shell"
    os.environ.pop("MODEL_UPLOAD_ROOT", None)
    os.environ.pop("VLM_SERVE_TOKEN", None)

    index._load_dotenv(tmp_path / ".env")

    assert os.environ["MODEL_ROOT"] == "/shell"
    assert os.environ["MODEL_UPLOAD_ROOT"] == "/shell"
    assert os.environ["VLM_SERVE_TOKEN"] == "t"
