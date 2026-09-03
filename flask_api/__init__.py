"""Flask API 패키지.

api_blueprint 생성 및 앱 등록을 이 모듈에서 처리한다.
web_main.py 에서 register_flask_api(app) 만 호출하면 된다.
"""

from flask import Blueprint, Flask, jsonify

from .model_upload.config import (
    build_model_upload_health_payload,
    register_model_upload_routes,
)
from .vlm_serve import build_vlm_health_payload, register_vlm_serve_routes

DEFAULT_URL_PREFIX = "/api"

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
        }
    )


register_vlm_serve_routes(api_blueprint)
register_model_upload_routes(api_blueprint)


def register_flask_api(app: Flask, url_prefix: str = DEFAULT_URL_PREFIX) -> None:
    """앱에 api blueprint 를 등록한다."""
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)


__all__ = ["api_blueprint", "register_flask_api", "DEFAULT_URL_PREFIX"]
