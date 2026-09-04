"""flask_api 의 site.env 로딩 테스트.

    pytest tests/test_site_env.py
"""

import os

import flask_api


def test_site_env_fills_gaps_but_never_overrides_shell(tmp_path, monkeypatch):
    """빈 자리만 채운다 - WSGI/systemd 가 준 환경이 우선. ${VAR} 는 최종 환경 기준, 따옴표는 벗긴다."""
    site_env = tmp_path / "site.env"
    site_env.write_text('MODEL_ROOT=/file\nMODEL_UPLOAD_ROOT=${MODEL_ROOT}\nVLM_SERVE_TOKEN="t"\n')
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ["MODEL_ROOT"] = "/shell"
    os.environ.pop("MODEL_UPLOAD_ROOT", None)
    os.environ.pop("VLM_SERVE_TOKEN", None)

    flask_api.load_site_env(site_env)

    assert os.environ["MODEL_ROOT"] == "/shell"
    assert os.environ["MODEL_UPLOAD_ROOT"] == "/shell"
    assert os.environ["VLM_SERVE_TOKEN"] == "t"


def test_site_env_default_path_sits_beside_model_envs(tmp_path, monkeypatch):
    """SITE_ENV 가 없으면 CONFIG_ROOT/site.env - models/*.env 와 같은 디렉터리라 폴더째 복사하면 따라온다."""
    (tmp_path / "site.env").write_text("SITE_ENV_PROBE=1\n")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop("SITE_ENV", None)
    os.environ["CONFIG_ROOT"] = str(tmp_path)

    flask_api.load_site_env()

    assert os.environ["SITE_ENV_PROBE"] == "1"
