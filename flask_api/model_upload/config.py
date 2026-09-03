"""모델 업로드 엔드포인트 설정.

deploy_vlms/config/common.env 의 ALLOWED_MODEL_ROOT 와 같은 곳을 기본 목적지로 쓴다.
staging 은 반드시 목적지 루트 **안쪽**에 둔다 - 같은 파일시스템이어야
os.replace 가 원자적이고, 다 받은 뒤 파티션을 넘어 복사하는 일이 없다.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from flask import Blueprint

from .routes import create_model_upload_blueprint
from .store import UploadStore

DEFAULT_DEST_ROOT = "/path/to/models"
STAGING_DIRNAME = ".upload_staging"
DEFAULT_MAX_CHUNK_MB = 64
URL_PREFIX = "/model_upload"


@dataclass(frozen=True)
class ModelUploadConfig:
    """업로드 엔드포인트 런타임 설정."""

    dest_root: Path
    staging_root: Path
    token: str
    max_chunk_bytes: int
    enabled: bool


def load_upload_config() -> ModelUploadConfig:
    """환경변수에서 업로드 설정을 읽는다."""
    # deploy_vlms/config/common.env 의 ALLOWED_MODEL_ROOT 가 서빙 쪽 루트다.
    # 그것이 바뀌면 업로드 목적지도 따라가야 한다 - 하드코딩 기본값은 마지막 수단.
    dest_root = Path(
        os.environ.get("MODEL_UPLOAD_ROOT", "").strip()
        or os.environ.get("ALLOWED_MODEL_ROOT", "").strip()
        or DEFAULT_DEST_ROOT
    ).expanduser()

    staging_override = os.environ.get("MODEL_UPLOAD_STAGING_DIR", "").strip()
    staging_root = (
        Path(staging_override).expanduser()
        if staging_override
        else dest_root / STAGING_DIRNAME
    )

    max_chunk_mb = int(
        os.environ.get("MODEL_UPLOAD_MAX_CHUNK_MB", str(DEFAULT_MAX_CHUNK_MB)).strip()
        or DEFAULT_MAX_CHUNK_MB
    )
    enabled = os.environ.get("MODEL_UPLOAD_ENABLED", "1").strip() not in {"0", "false", "False"}

    return ModelUploadConfig(
        dest_root=dest_root,
        staging_root=staging_root,
        token=os.environ.get("MODEL_UPLOAD_TOKEN", "").strip(),
        max_chunk_bytes=max_chunk_mb * 1024 * 1024,
        enabled=enabled,
    )


def build_model_upload_blueprint() -> Blueprint:
    """설정을 읽어 업로드 blueprint 를 만든다."""
    config = load_upload_config()
    store = UploadStore(dest_root=config.dest_root, staging_root=config.staging_root)
    return create_model_upload_blueprint(
        store=store, token=config.token, max_chunk_bytes=config.max_chunk_bytes
    )


def build_model_upload_health_payload() -> dict[str, object]:
    """/api/health 에 실을 업로드 엔드포인트 요약을 만든다."""
    config = load_upload_config()
    return {
        "enabled": config.enabled,
        "base_path": f"/api{URL_PREFIX}",
        "dest_root": str(config.dest_root),
        "max_chunk_bytes": config.max_chunk_bytes,
        "auth_required": bool(config.token),
    }


def register_model_upload_routes(api_blueprint: Blueprint) -> None:
    """API blueprint 에 업로드 blueprint 를 등록한다."""
    if not load_upload_config().enabled:
        print("[WARNING] model_upload 비활성 (MODEL_UPLOAD_ENABLED=0)")
        return
    api_blueprint.register_blueprint(build_model_upload_blueprint(), url_prefix=URL_PREFIX)
