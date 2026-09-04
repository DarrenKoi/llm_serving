"""Flask API 패키지.

api_blueprint 생성 및 앱 등록을 이 모듈에서 처리한다.
web_main.py 에서 register_flask_api(app) 만 호출하면 된다.
"""

import os
from pathlib import Path

from flask import Blueprint, Flask, jsonify

from .model_upload.config import (
    build_model_upload_health_payload,
    register_model_upload_routes,
)
from .dashboard import register_dashboard
from .gpu_status import build_gpu_status_payload, register_gpu_status_routes
from .vlm_serve import (
    _deploy_model_env_root,
    _load_env_file,
    build_vlm_health_payload,
    register_vlm_serve_routes,
)

DEFAULT_URL_PREFIX = "/api"


def load_site_env(path: Path | None = None) -> None:
    """deploy_vlms/config/site.env(사이트 경로·토큰)를 os.environ 의 빈 자리에 채운다.

    서버에는 git 없이 deploy_vlms/ 와 flask_api/ 만 복사해 올리므로 사이트 값은 그
    폴더 안에서 따라와야 한다. model_upload 가 import 시점에 설정을 굳히므로 아래
    라우트 등록보다 먼저 부른다. 이미 있는 키는 두므로 WSGI/systemd 가 준 환경이
    우선한다. SITE_ENV 로 경로를 바꿀 수 있다(conftest 가 os.devnull 로 끈다).
    serve_vlm.py 도 같은 파일을 같은 규칙으로 읽는다.
    """
    if path is None:
        override = os.environ.get("SITE_ENV", "").strip()
        path = Path(override) if override else _deploy_model_env_root().parent / "site.env"
    if not path.is_file():
        return
    for key, value in _load_env_file(path).items():
        os.environ.setdefault(key, os.path.expandvars(value))

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/health", methods=["GET"])
def health():
    """API root health endpoint."""
    return jsonify(
        {
            "service": "api",
            "status": "ok",
            "base_path": "/api",
            "vlm_serve": build_vlm_health_payload(),
            "model_upload": build_model_upload_health_payload(),
            "gpu_status": build_gpu_status_payload(),
        }
    )


load_site_env()
register_vlm_serve_routes(api_blueprint)
register_model_upload_routes(api_blueprint)
register_gpu_status_routes(api_blueprint)


def register_flask_api(app: Flask, url_prefix: str = DEFAULT_URL_PREFIX) -> None:
    """앱에 api blueprint 를 등록한다."""
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)


__all__ = [
    "api_blueprint",
    "load_site_env",
    "register_dashboard",
    "register_flask_api",
    "DEFAULT_URL_PREFIX",
]
