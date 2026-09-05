"""VLM proxy logger.

`flask_api.vlm_serve` 아래 로거 하나에 회전 파일 핸들러를 한 번만 단다. 핸들러에
FILE_HANDLER_MARKER 를 찍어 두는 이유: uWSGI/앱이 같은 로거에 다른 핸들러를 붙여도
"이미 있다" 로 오판해 파일 로그를 빼먹지 않기 위해서다.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "flask_api.vlm_serve"
FILE_HANDLER_MARKER = "_vlm_serve_file_handler"
LOG_MAX_BYTES = 20 * 1024 * 1024
LOG_BACKUP_COUNT = 10


def _log_dir() -> Path:
    """VLM_SERVE_LOG_DIR 이 없으면 저장소 루트의 logs/vlm_service."""
    override = os.environ.get("VLM_SERVE_LOG_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "logs" / "vlm_service"


def get_vlm_logger(name: str | None = None) -> logging.Logger:
    """`flask_api.vlm_serve` (또는 그 하위 `name`) 로거를 반환한다."""
    root_logger = logging.getLogger(LOGGER_NAME)
    root_logger.setLevel(os.environ.get("VLM_SERVE_LOG_LEVEL", "INFO").strip().upper() or "INFO")

    desired_path = _log_dir() / "vlm_serve.log"
    for handler in list(root_logger.handlers):
        if not isinstance(handler, RotatingFileHandler):
            continue
        if not getattr(handler, FILE_HANDLER_MARKER, False):
            continue
        if Path(handler.baseFilename) == desired_path:
            break
        root_logger.removeHandler(handler)  # 경로가 바뀌었다 - 새로 단다
        handler.close()
    else:
        try:
            desired_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                desired_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
            )
        except OSError as exc:
            print(f"[WARNING] Failed to initialize VLM service logger at {desired_path}: {exc}")
            return root_logger if not name else logging.getLogger(f"{LOGGER_NAME}.{name}")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        setattr(handler, FILE_HANDLER_MARKER, True)
        root_logger.addHandler(handler)

    return root_logger if not name else logging.getLogger(f"{LOGGER_NAME}.{name}")


__all__ = ["get_vlm_logger"]
