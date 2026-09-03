"""VLM 서비스 blueprint template.

이미지 분석 전용 프록시이지만, upstream 이 `stream=true` 에서만
정상적인 assistant content 를 주는 경우가 있어 서비스별 최소 보정을 허용한다.
프록시 자체는 upstream 응답을 끝까지 버퍼링한 뒤 그대로 반환한다.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from hmac import compare_digest

import requests
from flask import Blueprint, Response, jsonify, request

from .logger import get_vlm_logger

logger = get_vlm_logger("proxy")


@dataclass(frozen=True)
class VLMServiceConfig:
    """VLM 서비스 route 설정."""

    blueprint_name: str
    route_slug: str
    display_name: str
    upstream_port: int
    force_stream: bool = False

    @property
    def env_prefix(self) -> str:
        """환경변수 prefix 를 반환한다."""
        return self.route_slug.replace("-", "_").upper()

    @property
    def api_base_path(self) -> str:
        """API base path 를 반환한다."""
        return f"/api/vlm_serve/{self.route_slug}"

    @property
    def health_path(self) -> str:
        """Health path 를 반환한다."""
        return f"{self.api_base_path}/health"

    @property
    def upstream_base_url(self) -> str:
        """예상 upstream base URL 을 반환한다."""
        service_key = f"VLM_SERVE_{self.env_prefix}_BASE_URL"
        service_url = os.environ.get(service_key, "").strip().rstrip("/")
        if service_url:
            return service_url

        upstream_host = os.environ.get("VLM_SERVE_UPSTREAM_HOST", "127.0.0.1").strip()
        if not upstream_host:
            upstream_host = "127.0.0.1"
        return f"http://{upstream_host}:{self.upstream_port}"

    def to_dict(self) -> dict[str, object]:
        """직렬화용 dict 를 반환한다."""
        return {
            "service": self.route_slug,
            "model_name": self.display_name,
            "mode": "proxy",
            "upstream_port": self.upstream_port,
            "upstream_base_url": self.upstream_base_url,
            "api_base_path": self.api_base_path,
            "health_url": self.health_path,
        }


def _upstream_timeout() -> tuple[float, float]:
    """Upstream 요청 timeout 을 반환한다."""
    connect_timeout = float(os.environ.get("VLM_SERVE_CONNECT_TIMEOUT_SEC", "5.0"))
    read_timeout = float(os.environ.get("VLM_SERVE_READ_TIMEOUT_SEC", "300.0"))
    return connect_timeout, read_timeout


def _shared_token() -> str:
    """팀 공용 토큰. 비어 있으면 인증 없이 열린다 (model_upload 와 같은 규약)."""
    return os.environ.get("VLM_SERVE_TOKEN", "").strip()


def _presented_token() -> str:
    """호출자가 제시한 토큰을 꺼낸다.

    X-VLM-Token 을 먼저 보고, 없으면 Authorization: Bearer 를 본다.
    OpenAI 클라이언트는 api_key 를 Authorization 으로 보내므로 둘 다 받아야
    팀원이 표준 클라이언트를 그대로 쓸 수 있다.
    """
    provided = request.headers.get("X-VLM-Token", "").strip()
    if provided:
        return provided
    authorization = request.headers.get("Authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _prepare_upstream_body(
    config: VLMServiceConfig,
    upstream_path: str,
    raw_body: bytes,
    content_type: str,
) -> bytes:
    """요청 body 를 upstream 으로 넘기기 전에 필요한 최소 보정만 수행한다."""
    if not raw_body:
        return raw_body

    if not config.force_stream:
        return raw_body

    normalized_path = f"/{upstream_path.lstrip('/')}"
    if normalized_path != "/v1/chat/completions":
        return raw_body

    if "json" not in (content_type or "").lower():
        return raw_body

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        logger.warning(
            "could not coerce stream flag service=%s path=%s content_type=%s",
            config.route_slug,
            normalized_path,
            content_type,
        )
        return raw_body

    if not isinstance(payload, dict):
        return raw_body

    if payload.get("stream") is True:
        return raw_body

    payload["stream"] = True
    logger.info(
        "forcing stream=true service=%s path=%s",
        config.route_slug,
        normalized_path,
    )
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _build_upstream_headers() -> dict[str, str]:
    """Hop-by-hop 헤더를 제외한 upstream 요청 헤더를 구성한다."""
    blocked_headers = {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "accept-encoding",
    }
    # 공용 토큰을 쓰는 중이면 클라이언트의 Authorization 은 **이 프록시에게** 온 것이다.
    # 그대로 넘기면 vLLM 이 API_KEY 를 켰을 때 엉뚱한 키가 도착해 401 이 난다.
    # 여기서 지우면 아래 주입 로직이 upstream 용 키를 제대로 채운다.
    if _shared_token():
        blocked_headers.add("authorization")
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in blocked_headers
    }

    default_api_key = os.environ.get("VLM_SERVE_UPSTREAM_API_KEY", "").strip()
    if default_api_key and "authorization" not in {key.lower() for key in headers}:
        headers["Authorization"] = f"Bearer {default_api_key}"

    return headers


def _build_response_headers(upstream_headers: requests.structures.CaseInsensitiveDict) -> list[tuple[str, str]]:
    """Flask 응답으로 넘길 헤더를 정리한다."""
    blocked_headers = {
        "content-length",
        "connection",
        "transfer-encoding",
        "content-encoding",
    }
    return [
        (key, value)
        for key, value in upstream_headers.items()
        if key.lower() not in blocked_headers
    ]


def _proxy_request(config: VLMServiceConfig, upstream_path: str):
    """현재 요청을 upstream vLLM 으로 프록시한다 (비스트리밍)."""
    upstream_url = f"{config.upstream_base_url.rstrip('/')}/{upstream_path.lstrip('/')}"
    start_time = time.monotonic()
    request_body = _prepare_upstream_body(
        config,
        upstream_path,
        request.get_data(cache=True),
        request.content_type or "",
    )
    request_headers = _build_upstream_headers()
    logger.info(
        "request service=%s method=%s upstream_url=%s",
        config.route_slug,
        request.method,
        upstream_url,
    )
    try:
        upstream_response = requests.request(
            method=request.method,
            url=upstream_url,
            params=request.args,
            headers=request_headers,
            data=request_body,
            cookies=request.cookies,
            timeout=_upstream_timeout(),
            stream=False,
        )
    except requests.RequestException as exc:
        logger.exception(
            "upstream failed service=%s upstream_url=%s elapsed_ms=%.1f error=%s",
            config.route_slug,
            upstream_url,
            (time.monotonic() - start_time) * 1000,
            exc,
        )
        return jsonify(
            {
                "service": config.route_slug,
                "status": "error",
                "message": str(exc),
                "upstream_url": upstream_url,
            }
        ), 502

    response_headers = _build_response_headers(upstream_response.headers)
    body = upstream_response.content
    level = logging.INFO
    if upstream_response.status_code >= 500:
        level = logging.ERROR
    elif upstream_response.status_code >= 400:
        level = logging.WARNING
    logger.log(
        level,
        "response service=%s upstream_url=%s status=%s elapsed_ms=%.1f",
        config.route_slug,
        upstream_url,
        upstream_response.status_code,
        (time.monotonic() - start_time) * 1000,
    )
    upstream_response.close()
    return Response(
        body,
        status=upstream_response.status_code,
        headers=response_headers,
    )


def create_vlm_service_blueprint(config: VLMServiceConfig) -> Blueprint:
    """VLM 서비스 proxy blueprint 를 생성한다."""
    service_blueprint = Blueprint(config.blueprint_name, __name__)

    @service_blueprint.before_request
    def _require_shared_token():
        """프록시 호출에 공용 토큰을 요구한다. home / health 는 열어 둔다.

        VLM_SERVE_TOKEN 이 비어 있으면 아무것도 하지 않는다 - 토큰을 설정하는
        순간에만 켜지므로 기존 배포가 조용히 막히지 않는다.
        """
        token = _shared_token()
        if not token:
            return None
        # endpoint 는 "api.vlm_serve.mai_ui_vlm.proxy_v1" 형태라 마지막 조각만 본다.
        if (request.endpoint or "").rsplit(".", 1)[-1] in {"home", "health"}:
            return None
        if not compare_digest(_presented_token(), token):
            return (
                jsonify({"error": "missing or invalid VLM token", "code": "Unauthorized"}),
                401,
            )
        return None

    @service_blueprint.route("/", methods=["GET"])
    def home():
        """서비스 안내 엔드포인트."""
        payload = config.to_dict()
        payload.update(
            {
                "status": "ok",
                "message": "VLM service proxy route is ready.",
            }
        )
        return jsonify(payload)

    @service_blueprint.route("/health", methods=["GET"])
    def health():
        """서비스 헬스 체크 엔드포인트."""
        return _proxy_request(config, "/v1/models")

    @service_blueprint.route("/v1/<path:subpath>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    def proxy_v1(subpath: str):
        """OpenAI-compatible `/v1/*` 요청을 upstream 으로 프록시한다."""
        return _proxy_request(config, f"/v1/{subpath}")

    return service_blueprint


__all__ = ["VLMServiceConfig", "create_vlm_service_blueprint"]
