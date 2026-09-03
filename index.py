"""WSGI 진입점. WSGI 서버와 `python index.py` 가 여기서 application 을 가져간다.

저장소 루트 .env 를 셸의 `set -a; . ./.env; set +a` 대신 여기서 읽는다.
flask_api 는 import 시점에 환경을 읽으므로(model_upload 가 설정을 그때 굳힌다)
web_main import 보다 먼저여야 한다. 이미 있는 키는 두므로 WSGI/systemd 가 준
환경이 .env 보다 우선한다. serve_vlm.py 도 같은 규칙으로 같은 파일을 읽는다.
"""

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """KEY=VALUE 파일을 os.environ 의 빈 자리에만 채운다. ${VAR} 는 그 시점 환경으로 펼친다."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ.setdefault(key.strip(), os.path.expandvars(value))


_load_dotenv(Path(__file__).with_name(".env"))

from web_main import app as application  # .env 다음이어야 한다 - 위 독스트링 참고

if __name__ == "__main__":
    application.run(debug=True, host="0.0.0.0")
