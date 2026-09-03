"""VLM serve 라우트 패키지.

vlm_serve_blueprint 생성, 서비스 blueprint 등록, health payload 구성을
모두 이 모듈에서 처리한다.
"""

import os
from pathlib import Path
from typing import Any

import requests
from flask import Blueprint, jsonify

from .config import (
    ALL_VLM_SERVICES,
    ENABLED_VLM_SERVICES,
    VLMServiceEntry,
    get_enabled_slugs,
)
from .mai_ui import SERVICE_CONFIG as MAI_UI_CONFIG
from .mai_ui import service_blueprint as mai_ui_blueprint
from .mai_ui_2b import SERVICE_CONFIG as MAI_UI_2B_CONFIG
from .mai_ui_2b import service_blueprint as mai_ui_2b_blueprint
from .paddleocr_vl import SERVICE_CONFIG as PADDLEOCR_VL_CONFIG
from .paddleocr_vl import service_blueprint as paddleocr_vl_blueprint
from .qwen3_8_27b import SERVICE_CONFIG as QWEN3_8_27B_CONFIG
from .qwen3_8_27b import service_blueprint as qwen3_8_27b_blueprint

# ── 서비스 blueprint 등록 ─────────────────────────────────────────────

_ALL_SERVICE_BLUEPRINTS = [
    (MAI_UI_CONFIG, mai_ui_blueprint),
    (MAI_UI_2B_CONFIG, mai_ui_2b_blueprint),
    (PADDLEOCR_VL_CONFIG, paddleocr_vl_blueprint),
    (QWEN3_8_27B_CONFIG, qwen3_8_27b_blueprint),
]

# config.py 의 enabled 플래그에 따라 활성 서비스만 등록
_enabled_slugs = get_enabled_slugs()
VLM_SERVICE_BLUEPRINTS = [
    (cfg, bp) for cfg, bp in _ALL_SERVICE_BLUEPRINTS
    if cfg.route_slug in _enabled_slugs
]
VLM_SERVICE_CONFIGS = {
    service_config.route_slug: service_config
    for service_config, _ in VLM_SERVICE_BLUEPRINTS
}

vlm_serve_blueprint = Blueprint("vlm_serve", __name__)

# ── health probe 내부 함수 ────────────────────────────────────────────


def _deploy_model_env_root() -> Path:
    """배포 model env 디렉터리를 반환한다."""
    config_root = os.environ.get("CONFIG_ROOT", "").strip()
    if config_root:
        return Path(config_root).expanduser().resolve() / "models"

    deploy_vlms_root = os.environ.get("DEPLOY_VLMS_ROOT", "").strip()
    if deploy_vlms_root:
        return Path(deploy_vlms_root).expanduser().resolve() / "config" / "models"

    return Path(__file__).resolve().parents[2] / "deploy_vlms" / "config" / "models"


def _health_timeout_sec() -> float:
    """VLM health probe timeout 을 반환한다."""
    raw_value = os.environ.get("VLM_SERVE_HEALTH_TIMEOUT_SEC", "2.0").strip()
    try:
        return max(float(raw_value), 0.1)
    except ValueError:
        return 2.0


def _load_env_file(path: Path) -> dict[str, str]:
    """간단한 KEY=VALUE env 파일을 읽는다."""
    payload: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key:
                payload[key] = value
    return payload


def _parse_port(raw_value: str) -> int | None:
    """env 문자열에서 port 값을 파싱한다."""
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def _env_prefix(route_slug: str) -> str:
    """route slug 기반 env prefix 를 반환한다."""
    return route_slug.replace("-", "_").replace(".", "_").upper()


def _base_url_for_service(route_slug: str, upstream_port: int | None) -> str | None:
    """route slug 기준 upstream base URL 을 계산한다."""
    service_config = VLM_SERVICE_CONFIGS.get(route_slug)
    if service_config is not None:
        return service_config.upstream_base_url

    service_key = f"VLM_SERVE_{_env_prefix(route_slug)}_BASE_URL"
    service_url = os.environ.get(service_key, "").strip().rstrip("/")
    if service_url:
        return service_url

    if upstream_port is None:
        return None

    upstream_host = os.environ.get("VLM_SERVE_UPSTREAM_HOST", "127.0.0.1").strip()
    if not upstream_host:
        upstream_host = "127.0.0.1"
    return f"http://{upstream_host}:{upstream_port}"


def _display_name(route_slug: str, env_values: dict[str, str]) -> str:
    """health payload 에 표시할 model 이름을 계산한다."""
    service_config = VLM_SERVICE_CONFIGS.get(route_slug)
    if service_config is not None:
        return service_config.display_name

    model_id = env_values.get("MODEL_ID", "").strip()
    if model_id:
        return Path(model_id).name
    return route_slug


def _extract_model_ids(payload: Any) -> list[str]:
    """OpenAI `/v1/models` 응답에서 model id 목록을 추출한다."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]


def _probe_service(entry: dict[str, Any]) -> dict[str, Any]:
    """개별 VLM 상태를 프로브한다."""
    upstream_base_url = entry.get("upstream_base_url")
    if not upstream_base_url:
        entry["health_status"] = "not_proxy_managed"
        entry["reason"] = "No OpenAI-compatible PORT or BASE_URL is configured for live probing."
        entry["detected_models"] = []
        return entry

    health_url = f"{str(upstream_base_url).rstrip('/')}/v1/models"
    entry["probe_url"] = health_url
    try:
        response = requests.get(health_url, timeout=_health_timeout_sec())
    except requests.RequestException as exc:
        entry["health_status"] = "unreachable"
        entry["reason"] = str(exc)
        entry["detected_models"] = []
        return entry

    entry["http_status"] = response.status_code
    try:
        payload = response.json()
    except ValueError:
        entry["health_status"] = "invalid_response"
        entry["reason"] = "Health probe returned a non-JSON body."
        entry["detected_models"] = []
        return entry

    detected_models = _extract_model_ids(payload)
    entry["detected_models"] = detected_models
    expected_model = entry.get("served_model_name")
    if response.status_code >= 400:
        entry["health_status"] = "error"
        entry["reason"] = f"Upstream responded with HTTP {response.status_code}."
        return entry

    if expected_model and detected_models and expected_model not in detected_models:
        entry["health_status"] = "serving_mismatch"
        entry["reason"] = (
            f"Expected served model '{expected_model}' was not present in upstream /v1/models."
        )
        return entry

    entry["health_status"] = "serving"
    return entry


def _configured_vlm_entries() -> list[dict[str, Any]]:
    """배포 env 와 등록된 proxy 를 합쳐 VLM 상태 엔트리를 만든다."""
    env_root = _deploy_model_env_root()
    entries: list[dict[str, Any]] = []
    seen_services: set[str] = set()

    if env_root.is_dir():
        for env_path in sorted(env_root.glob("*.env")):
            env_values = _load_env_file(env_path)
            route_slug = env_path.stem
            service_config = VLM_SERVICE_CONFIGS.get(route_slug)
            upstream_port = _parse_port(env_values.get("PORT", "").strip())
            if upstream_port is None and service_config is not None:
                upstream_port = service_config.upstream_port

            entry = {
                "service": route_slug,
                "display_name": _display_name(route_slug, env_values),
                "served_model_name": env_values.get("SERVED_MODEL_NAME", "").strip() or None,
                "upstream_port": upstream_port,
                "upstream_base_url": _base_url_for_service(route_slug, upstream_port),
                "proxy_registered": service_config is not None,
                "runtime": "openai-compatible" if upstream_port is not None else "local-transformers",
                "api_base_path": service_config.api_base_path if service_config is not None else None,
                "health_path": service_config.health_path if service_config is not None else None,
                "source_env": str(env_path),
            }
            entries.append(_probe_service(entry))
            seen_services.add(route_slug)

    for route_slug, service_config in VLM_SERVICE_CONFIGS.items():
        if route_slug in seen_services:
            continue
        entry = {
            "service": route_slug,
            "display_name": service_config.display_name,
            "served_model_name": None,
            "upstream_port": service_config.upstream_port,
            "upstream_base_url": service_config.upstream_base_url,
            "proxy_registered": True,
            "runtime": "openai-compatible",
            "api_base_path": service_config.api_base_path,
            "health_path": service_config.health_path,
            "source_env": None,
        }
        entries.append(_probe_service(entry))

    return entries


# ── public API ────────────────────────────────────────────────────────


def build_vlm_health_payload() -> dict[str, Any]:
    """VLM proxy health payload 를 구성한다."""
    vlm_statuses = _configured_vlm_entries()
    serving_entries = [
        item
        for item in vlm_statuses
        if item.get("health_status") == "serving"
    ]
    return {
        "service": "vlm_serve",
        "status": "ok",
        "mode": "proxy",
        "base_path": "/api/vlm_serve",
        "registered_vlms": [
            service_config.route_slug
            for service_config, _ in VLM_SERVICE_BLUEPRINTS
        ],
        "serving_now": [
            item["display_name"]
            for item in serving_entries
        ],
        "serving_routes": [
            item["service"]
            for item in serving_entries
        ],
        "vlm_statuses": vlm_statuses,
    }


def register_vlm_serve_routes(api_blueprint: Blueprint) -> None:
    """API blueprint 에 VLM 서비스 blueprint 를 등록한다."""
    api_blueprint.register_blueprint(vlm_serve_blueprint, url_prefix="/vlm_serve")


# ── route 정의 ────────────────────────────────────────────────────────


@vlm_serve_blueprint.route("/", methods=["GET"], strict_slashes=False)
def home():
    """VLM 상태를 직접 반환하는 기본 엔드포인트."""
    return jsonify(build_vlm_health_payload())


@vlm_serve_blueprint.route("/health", methods=["GET"])
def health():
    """VLM 헬스 체크 엔드포인트."""
    return jsonify(build_vlm_health_payload())


# 각 서비스 blueprint 를 vlm_serve 하위에 등록
for _service_config, _service_blueprint in VLM_SERVICE_BLUEPRINTS:
    vlm_serve_blueprint.register_blueprint(
        _service_blueprint,
        url_prefix=f"/{_service_config.route_slug}",
    )


__all__ = [
    "ALL_VLM_SERVICES",
    "ENABLED_VLM_SERVICES",
    "VLM_SERVICE_BLUEPRINTS",
    "VLMServiceEntry",
    "build_vlm_health_payload",
    "register_vlm_serve_routes",
    "vlm_serve_blueprint",
]
