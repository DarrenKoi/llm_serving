"""VLM 서버 상태 일괄 확인 스크립트.

config/models/*.env 파일에서 PORT와 SERVED_MODEL_NAME을 읽어
등록된 모든 vLLM 서버의 생존 여부를 한 번에 확인한다.

사용법:
  python check_vlm.py            # 모든 모델 확인 (localhost)
  python check_vlm.py 10.0.0.5   # 특정 호스트의 모든 모델 확인
"""

import sys
import json
import urllib.request
import urllib.error
from pathlib import Path


def read_env_value(path: Path, key: str) -> str:
    """env 파일에서 key에 해당하는 값을 읽는다."""
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, _, value = line.partition("=")
            if current_key.strip() != key:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            return value
    return ""


def discover_models(config_dir: Path) -> list[dict]:
    """config/models/*.env에서 PORT가 있는 모델 목록을 반환한다."""
    models = []
    for env_file in sorted(config_dir.glob("*.env")):
        port = read_env_value(env_file, "PORT")
        served_name = read_env_value(env_file, "SERVED_MODEL_NAME")
        if not port:
            continue
        models.append({
            "instance": env_file.stem,
            "port": int(port),
            "served_name": served_name or env_file.stem,
        })
    models.sort(key=lambda m: m["port"])
    return models


def check_model(host: str, port: int, expected_name: str) -> tuple[bool, str]:
    """vLLM /v1/models 엔드포인트를 호출해 모델 존재 여부를 확인한다."""
    url = f"http://{host}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return False, str(e)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False, "JSON 응답 파싱 실패"

    data = payload.get("data")
    if not isinstance(data, list):
        return False, "응답에 model data 목록이 없음"

    model_ids = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if model_id:
            model_ids.append(model_id)

    if expected_name in model_ids:
        return True, ""
    if model_ids:
        return False, f"기대 모델 '{expected_name}' 없음 (응답: {', '.join(model_ids)})"
    return False, f"기대 모델 '{expected_name}' 없음 (응답에 id 없음)"


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"

    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir.parent / "config" / "models"
    if not config_dir.is_dir():
        print(f"[ERROR] config 디렉토리를 찾을 수 없음: {config_dir}", file=sys.stderr)
        sys.exit(1)

    models = discover_models(config_dir)
    if not models:
        print("[WARNING] PORT가 설정된 모델이 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 호스트: {host} | 확인 대상: {len(models)}개 모델\n")

    alive, dead = 0, 0
    for m in models:
        ok, err = check_model(host, m["port"], m["served_name"])
        port_str = f":{m['port']}"
        if ok:
            alive += 1
            print(f"  [OK]   {port_str:<6} {m['served_name']:<25} ({m['instance']})")
        else:
            dead += 1
            print(f"  [FAIL] {port_str:<6} {m['served_name']:<25} ({m['instance']}) -- {err}")

    print(f"\n[INFO] 결과: {alive} alive / {dead} dead / {alive + dead} total")
    if dead:
        sys.exit(1)


if __name__ == "__main__":
    main()
