"""MODEL_ROOT / MODEL_ID 경로가 왜 안 잡히는지 보여주는 진단 스크립트.

두 가지 눈으로 같은 것을 본다.
  1. 런처의 눈: start_all.py 와 똑같이 serve_vlm.py <instance> 를 자식 프로세스로 띄운다.
     SERVE_VLM_DRY_RUN=1 이라 검증과 명령 조립까지만 하고 vllm 은 띄우지 않는다.
     출력을 그대로 보여준다 - 어느 config 를 읽었고 MODEL_ROOT 가 어디서 왔는지 포함.
  2. 이 프로세스의 눈: 같은 순서(site.env -> MODEL_ROOT_FALLBACK -> common.env ->
     models/*.env)로 읽어 만든 MODEL_ID 를 커널에 직접 묻고, 부모 디렉터리의 실제 항목을
     repr 로 찍는다 - 숨은 문자, 대소문자 차이, 대상 없는 심볼릭 링크를 잡기 위해.
serve_vlm.py 가 따르는 DEPLOY_VLMS_ROOT / CONFIG_ROOT / SITE_ENV 오버라이드를 그대로 따른다.

아무것도 바꾸지 않는다. 출력 전체를 붙여넣으면 된다.

사용법:
  python deploy_vlms/scripts/diagnose_paths.py
"""

import getpass
import os
import socket
import subprocess
import sys
import unicodedata
from pathlib import Path

from serve_vlm import MODEL_ROOT_FALLBACK, load_env_file

OVERRIDE_KEYS = ("DEPLOY_VLMS_ROOT", "CONFIG_ROOT", "COMMON_ENV", "MODEL_ENV", "SITE_ENV", "MODEL_ROOT")


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
        log(f"  symlink -> {os.readlink(path)!r} (target exists={os.path.exists(path)})")
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
        extra = f" -> {os.readlink(full)!r} (target exists={os.path.exists(full)})" if os.path.islink(full) else ""
        log(f"    {name!r} [{kind}]{extra}")
        if name != wanted and _norm(name) == _norm(wanted):
            warn(f"  거의 같은 이름: {name!r} vs 설정 {wanted!r} (대소문자/공백/유니코드 차이)")


def launcher_view(script_dir: Path, config_root: Path, base_env: dict[str, str]) -> None:
    """start_all.py 와 같은 방식으로 serve_vlm.py 를 띄워 그 출력을 그대로 보여준다."""
    for env_path in sorted((config_root / "models").glob("*.env")):
        instance = env_path.stem
        log(f"=== 런처의 눈: {sys.executable} serve_vlm.py {instance} (SERVE_VLM_DRY_RUN=1) ===")
        result = subprocess.run(
            [sys.executable, str(script_dir / "serve_vlm.py"), instance],
            env={**base_env, "SERVE_VLM_DRY_RUN": "1", "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in result.stdout.splitlines():
            print("    " + line)
        log(f"    exit code={result.returncode}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base_env = os.environ.copy()  # 런처가 물려받을 환경 그대로, 아래에서 파일을 읽기 전에 떠 둔다
    deploy_vlms_root = Path(base_env.get("DEPLOY_VLMS_ROOT", "").strip() or script_dir.parent)
    config_root = Path(base_env.get("CONFIG_ROOT", "").strip() or deploy_vlms_root / "config")

    log(f"host={socket.gethostname()} user={getpass.getuser()} uid={os.getuid()} python={sys.executable} {sys.version.split()[0]}")
    log(f"script_dir={str(script_dir)!r}")
    for key in OVERRIDE_KEYS:
        log(f"env {key}={base_env.get(key)!r}")
    log(f"config_root={str(config_root)!r} exists={config_root.is_dir()}")

    launcher_view(script_dir, config_root, base_env)

    log("=== 이 프로세스의 눈 ===")
    site_env = Path(base_env.get("SITE_ENV", "").strip() or config_root / "site.env")
    if site_env.is_file():
        load_env_file(str(site_env), override=False)
    if MODEL_ROOT_FALLBACK:
        os.environ.setdefault("MODEL_ROOT", MODEL_ROOT_FALLBACK)
    root = os.environ.get("MODEL_ROOT")
    source = "shell export" if base_env.get("MODEL_ROOT") else ("site.env" if site_env.is_file() and root else "MODEL_ROOT_FALLBACK")
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
        log(f"{env_path.stem}: MODEL_ID={model_id!r} (from {str(env_path)!r})")
        describe_dir(model_id)
        log(f"  config.json exists={os.path.isfile(os.path.join(model_id, 'config.json'))}")


if __name__ == "__main__":
    main()
