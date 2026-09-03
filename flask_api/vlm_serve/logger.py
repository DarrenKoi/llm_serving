"""VLM proxy logger helpers."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "flask_api.vlm_serve"
DEFAULT_LOG_DIRNAME = "logs"
DEFAULT_SERVICE_LOG_DIRNAME = "vlm_service"
DEFAULT_LOG_FILENAME = "vlm_serve.log"
DEFAULT_LOG_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 10
FILE_HANDLER_MARKER = "_vlm_serve_file_handler"


def _log_level() -> int:
    """환경변수 기반 logger level 을 반환한다."""
    level_name = os.environ.get("VLM_SERVE_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def _repo_root() -> Path:
    """현재 cloud repo root 경로를 반환한다."""
    override = os.environ.get("VLM_SERVE_REPO_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _log_dir() -> Path:
    """VLM service 로그 디렉터리를 반환한다."""
    override = os.environ.get("VLM_SERVE_LOG_DIR", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = _repo_root() / candidate
        return candidate.resolve()
    return _repo_root() / DEFAULT_LOG_DIRNAME / DEFAULT_SERVICE_LOG_DIRNAME


def _log_file_path() -> Path:
    """VLM service 로그 파일 경로를 반환한다."""
    return _log_dir() / DEFAULT_LOG_FILENAME


def _log_max_bytes() -> int:
    """로테이팅 로그 최대 파일 크기를 반환한다."""
    raw_value = os.environ.get("VLM_SERVE_LOG_MAX_BYTES", str(DEFAULT_LOG_MAX_BYTES)).strip()
    try:
        return max(int(raw_value), 1024)
    except ValueError:
        return DEFAULT_LOG_MAX_BYTES


def _log_backup_count() -> int:
    """로테이팅 로그 백업 개수를 반환한다."""
    raw_value = os.environ.get("VLM_SERVE_LOG_BACKUP_COUNT", str(DEFAULT_LOG_BACKUP_COUNT)).strip()
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return DEFAULT_LOG_BACKUP_COUNT


def _log_formatter() -> logging.Formatter:
    """파일 로그 formatter 를 반환한다."""
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )


def _configure_root_logger() -> logging.Logger:
    """루트 VLM logger 에 파일 핸들러를 보장한다."""
    root_logger = logging.getLogger(LOGGER_NAME)
    root_logger.setLevel(_log_level())

    desired_path = _log_file_path()
    existing_handler: RotatingFileHandler | None = None
    for handler in list(root_logger.handlers):
        if not getattr(handler, FILE_HANDLER_MARKER, False):
            continue
        if Path(getattr(handler, "baseFilename", "")).resolve() == desired_path.resolve():
            existing_handler = handler
            continue
        root_logger.removeHandler(handler)
        handler.close()

    if existing_handler is None:
        try:
            desired_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                desired_path,
                maxBytes=_log_max_bytes(),
                backupCount=_log_backup_count(),
                encoding="utf-8",
            )
            setattr(file_handler, FILE_HANDLER_MARKER, True)
            root_logger.addHandler(file_handler)
            existing_handler = file_handler
        except OSError as exc:
            print(f"[WARNING] Failed to initialize VLM service logger at {desired_path}: {exc}")
            return root_logger

    existing_handler.setLevel(_log_level())
    existing_handler.setFormatter(_log_formatter())
    return root_logger


def get_vlm_logger(name: str | None = None) -> logging.Logger:
    """`flask_api.vlm_serve` 하위 logger 를 반환한다."""
    root_logger = _configure_root_logger()
    if not name:
        return root_logger

    logger = logging.getLogger(f"{LOGGER_NAME}.{name}")
    logger.setLevel(root_logger.level)
    logger.propagate = True
    return logger


__all__ = [
    "get_vlm_logger",
]
