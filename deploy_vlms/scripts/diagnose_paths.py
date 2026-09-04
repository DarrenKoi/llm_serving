"""MODEL_ROOT / MODEL_ID 경로가 왜 안 잡히는지 보여주는 진단 스크립트.

serve_vlm.py 와 똑같은 순서로 config/site.env -> MODEL_ROOT_FALLBACK -> common.env ->
models/<instance>.env 를 읽은 뒤, 만들어진 MODEL_ID 를 커널에 직접 물어본다.
이름은 전부 repr 로 찍는다 - 눈에 안 보이는 문자(공백, 유니코드 유사 글자)를 잡기 위해.
심볼릭 링크는 대상과 대상의 존재 여부를 같이 찍는다.

아무것도 바꾸지 않는다. 출력만 붙여넣으면 된다.

사용법:
  python deploy_vlms/scripts/diagnose_paths.py
"""

import getpass
import os
import socket
import sys
import unicodedata
from pathlib import Path

from serve_vlm import MODEL_ROOT_FALLBACK, load_env_file


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def _norm(name: str) -> str:
    return unicodedata.normalize("NFC", name).strip().casefold()


def describe_dir(path: str) -> None:
    """path 의 존재 여부와, 없으면 부모 디렉터리 안의 실제 이름들을 찍는다."""
    log(f"  exists={os.path.exists(path)} isdir={os.path.isdir(path)} islink={os.path.islink(path)}")
    if os.path.islink(path):
        target = os.readlink(path)
        log(f"  symlink -> {target!r} (target exists={os.path.exists(path)})")
    if os.path.isdir(path):
        return

    parent, wanted = os.path.split(path)
    log(f"  parent {parent!r} isdir={os.path.isdir(parent)}")
    if not os.path.isdir(parent):
        warn("  부모 디렉터리부터 없다 - MODEL_ROOT 자체가 이 프로세스에서 안 보인다")
        return
    for name in sorted(os.listdir(parent)):
        full = os.path.join(parent, name)
        kind = "link" if os.path.islink(full) else ("dir" if os.path.isdir(full) else "file")
        extra = ""
        if os.path.islink(full):
            extra = f" -> {os.readlink(full)!r} (target exists={os.path.exists(full)})"
        log(f"    {name!r} [{kind}]{extra}")
        if name != wanted and _norm(name) == _norm(wanted):
            warn(f"  거의 같은 이름: {name!r} vs 설정 {wanted!r} (대소문자/공백/유니코드 차이)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    config_root = script_dir.parent / "config"
    log(f"host={socket.gethostname()} user={getpass.getuser()} uid={os.getuid()} python={sys.version.split()[0]}")
    log(f"config_root={str(config_root)!r}")

    shell_root = os.environ.get("MODEL_ROOT")
    site_env = config_root / "site.env"
    if site_env.is_file():
        load_env_file(str(site_env), override=False)
    if MODEL_ROOT_FALLBACK:
        os.environ.setdefault("MODEL_ROOT", MODEL_ROOT_FALLBACK)
    root = os.environ.get("MODEL_ROOT")
    source = "shell export" if shell_root else ("site.env" if site_env.is_file() and root else "MODEL_ROOT_FALLBACK")
    log(f"site.env={str(site_env)!r} present={site_env.is_file()}")
    log(f"MODEL_ROOT={root!r} (from {source})")
    if not root:
        warn("MODEL_ROOT 가 비어 있다")
        return
    log(f"MODEL_ROOT realpath={os.path.realpath(root)!r}")
    describe_dir(root)

    load_env_file(str(config_root / "common.env"))
    for env_path in sorted((config_root / "models").glob("*.env")):
        load_env_file(str(env_path))
        model_id = os.environ.get("MODEL_ID", "")
        log(f"{env_path.stem}: MODEL_ID={model_id!r}")
        describe_dir(model_id)
        config_json = os.path.join(model_id, "config.json")
        log(f"  config.json exists={os.path.isfile(config_json)}")


if __name__ == "__main__":
    main()
