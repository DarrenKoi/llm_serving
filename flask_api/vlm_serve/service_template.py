"""VLM 서비스 blueprint template.

프록시는 요청 body 를 손대지 않고 넘긴다. upstream 이 SSE(`text/event-stream`)를 주면
청크 단위로 그대로 흘리고(코딩 하네스의 stream=true), 그 외는 끝까지 버퍼링한 뒤 반환한다.

(역사: UI-TARS 가 non-stream 요청에 빈 content 를 주던 시절 force_stream 으로 body 의
stream 을 강제로 켜는 우회가 있었다. 그 모델과 함께 2026-09-05 에 제거 - 같은 증상이
다시 나오면 auto_recipe_creator 의 2b1186c/ba15f9c 를 보라.)
"""

import logging
import os
import time
from dataclasses import dataclass
from hmac import compare_digest

import requests
from flask import Blueprint, Response, jsonify, request, stream_with_context
from requests.structures import CaseInsensitiveDict

from .logger import get_vlm_logger

logger = get_vlm_logger("proxy")


def env_prefix_for(route_slug: str) -> str:
    """route slug 을 환경변수 이름 조각으로 바꾼다.

    slug 에는 `-` 와 `.` 이 둘 다 나온다 (`mai-ui`, `qwen3.8-27b`). 환경변수
    이름에 점은 쓸 수 없으므로 둘 다 `_` 로 접는다. 점을 빼먹으면
    `VLM_SERVE_QWEN3.8_27B_BASE_URL` 이라는, shell 로는 export 조차 못 하는
    키가 만들어져 override 가 조용히 무시된다.

    `__init__` 의 health 경로도 이 함수를 쓴다 - 프록시가 실제로 나가는 주소와
    health 가 보고하는 주소는 반드시 같은 키에서 나와야 한다.
    """
    return route_slug.replace("-", "_").replace(".", "_").upper()


@dataclass(frozen=True)
class VLMServiceConfig:
    """VLM 서비스 route 설정."""

    route_slug: str
    display_name: str
    upstream_port: int

    @property
    def env_prefix(self) -> str:
        """환경변수 prefix 를 반환한다."""
        return env_prefix_for(self.route_slug)

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


def _api_key() -> str:
    """팀 공용 키 하나(site.env 의 VLLM_API_KEY). 비어 있으면 인증 없이 열린다.

    같은 값이 호출자 검증과 업스트림 vLLM 의 --api-key 양쪽에 쓰인다 (model_upload 도 같은 키).
    """
    return os.environ.get("VLLM_API_KEY", "").strip()


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


def _build_upstream_headers() -> dict[str, str]:
    """Hop-by-hop 헤더를 제외한 upstream 요청 헤더를 구성한다."""
    blocked_headers = {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "accept-encoding",
    }
    api_key = _api_key()

    # 호출자의 Authorization 은 **이 프록시에게** 온 것이다 (Bearer 로 보낸 키, 또는 OpenAI
    # 클라이언트가 비워 둘 수 없어 채워 보내는 아무 값). 키를 쓰는 배포에서는 떼어내고 키를
    # 다시 싣는다 - X-VLM-Token 으로 인증한 호출자도 업스트림에는 인증된 채로 도착한다.
    if api_key:
        blocked_headers.add("authorization")

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in blocked_headers
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


def _build_response_headers(upstream_headers: CaseInsensitiveDict) -> list[tuple[str, str]]:
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
    """현재 요청을 upstream vLLM 으로 프록시한다. SSE 응답만 스트리밍한다."""
    upstream_url = f"{config.upstream_base_url.rstrip('/')}/{upstream_path.lstrip('/')}"
    start_time = time.monotonic()
    request_body = request.get_data(cache=True)
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
            stream=True,
        )
    except requests.RequestException as exc:
        # Timeout 은 504, 그 외는 502 로 가른다. 호출자(코딩 하네스)에게 이 구분이 전부다 -
        # 502 는 "업스트림이 죽었다"라 보통 재시도하지 않고, 504 는 "아직 일하는 중인데 내가
        # 못 기다렸다"라 재시도·설정 조정의 신호다. 둘을 502 로 뭉치면 느린 thinking 요청이
        # 모델이 죽은 것처럼 보고된다.
        is_timeout = isinstance(exc, requests.Timeout)
        logger.exception(
            "upstream failed service=%s upstream_url=%s elapsed_ms=%.1f timeout=%s error=%s",
            config.route_slug,
            upstream_url,
            (time.monotonic() - start_time) * 1000,
            is_timeout,
            exc,
        )
        payload = {
            "service": config.route_slug,
            "status": "error",
            "message": str(exc),
            "upstream_url": upstream_url,
        }
        if is_timeout:
            # non-stream 요청은 vLLM 이 완성 전까지 헤더조차 안 보내므로 read timeout 이
            # "첫 바이트까지"에 걸린다. 해결은 stream=true 다 (청크 사이 간격에만 걸린다).
            payload["hint"] = (
                "Send stream=true so the read timeout applies between chunks instead of to the "
                "whole completion, or lower chat_template_kwargs.reasoning_effort / set "
                f"thinking_token_budget. Current read timeout: {_upstream_timeout()[1]}s "
                "(VLM_SERVE_READ_TIMEOUT_SEC)."
            )
        return jsonify(payload), 504 if is_timeout else 502

    response_headers = _build_response_headers(upstream_response.headers)
    is_sse = upstream_response.headers.get("Content-Type", "").startswith("text/event-stream")
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
    if is_sse:
        # 청크가 올 때마다 바로 내보낸다. 읽기 timeout 은 청크 사이 간격에만 걸리므로 xhigh 로
        # 몇 분 생각하는 요청도 reasoning_content 델타가 흐르는 한 끊기지 않는다.
        def _relay():
            try:
                yield from upstream_response.iter_content(chunk_size=None)
            finally:
                upstream_response.close()

        response_headers.append(("X-Accel-Buffering", "no"))
        return Response(
            stream_with_context(_relay()),
            status=upstream_response.status_code,
            headers=response_headers,
        )

    body = upstream_response.content
    upstream_response.close()
    return Response(
        body,
        status=upstream_response.status_code,
        headers=response_headers,
    )


def create_vlm_service_blueprint(config: VLMServiceConfig) -> Blueprint:
    """VLM 서비스 proxy blueprint 를 생성한다."""
    # blueprint 이름은 slug 에서 유도한다 (mai-ui -> mai_ui, qwen3.8-27b -> qwen3_8_27b).
    service_blueprint = Blueprint(config.env_prefix.lower(), __name__)

    @service_blueprint.before_request
    def _require_api_key():
        """프록시 호출에 팀 공용 키(VLLM_API_KEY)를 요구한다. home / health 는 열어 둔다.

        키가 비어 있으면 아무것도 하지 않는다 - 서버 site.env 에 키를 채우는 순간
        vLLM · 프록시 · 업로드가 한꺼번에 켜진다.
        """
        token = _api_key()
        if not token:
            return None
        # endpoint 는 "api.vlm_serve.mai_ui.proxy_v1" 형태라 마지막 조각만 본다.
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
