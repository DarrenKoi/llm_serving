"""임의 instance 또는 family-size 조합으로 VLM 인스턴스를 시작한다.

기본 동작은 nohup과 비슷한 백그라운드 실행이다.
터미널을 닫아도 프로세스가 유지되며 로그/ PID 파일은 `deploy_vlms/runtime/` 아래에 남는다.
필요하면 이 파일 상단의 `RUN_IN_BACKGROUND` 값을 1/0으로 바꿔서 동작을 전환한다.

사용법:
  python start_model.py <instance>
  python start_model.py <family> <size>

예시:
  python start_model.py ui-venus
  python start_model.py ui-venus 30b
  python start_model.py mai-ui-7b

환경 변수:
  START_MODEL_LOG_DIR=/path/to/logs
  START_MODEL_PID_DIR=/path/to/pids
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from stop_model import (
    collect_targets,
    find_listening_pids_by_port,
    find_vllm_processes,
    resolve_port_for_instance,
    stop_instance,
)


STARTUP_POLL_SEC = 1.0
RUN_IN_BACKGROUND = 1


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def resolve_instance() -> str:
    if len(sys.argv) == 2:
        return normalize_token(sys.argv[1])
    if len(sys.argv) == 3:
        return f"{normalize_token(sys.argv[1])}-{normalize_token(sys.argv[2])}"

    print(__doc__, file=sys.stderr)
    sys.exit(1)


def resolve_runtime_paths(script_dir: Path, instance: str) -> tuple[Path, Path]:
    deploy_vlms_root = Path(os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or script_dir.parent)
    runtime_root = deploy_vlms_root / "runtime"
    log_dir = Path(os.environ.get("START_MODEL_LOG_DIR", "").strip() or (runtime_root / "logs"))
    pid_dir = Path(os.environ.get("START_MODEL_PID_DIR", "").strip() or (runtime_root / "pids"))
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{instance}.log", pid_dir / f"{instance}.pid"


def launch_foreground(script_dir: Path, instance: str) -> int:
    cmd = [sys.executable, str(script_dir / "serve_vlm.py"), instance]
    log(f"Starting instance={instance} in foreground")
    return subprocess.call(cmd)


def launch_detached(script_dir: Path, instance: str) -> int:
    cmd = [sys.executable, str(script_dir / "serve_vlm.py"), instance]
    log_path, pid_path = resolve_runtime_paths(script_dir, instance)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("a", encoding="utf-8") as log_file:
        launched_at = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n[INFO] Launch requested at {launched_at} for instance={instance}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(STARTUP_POLL_SEC)
    return_code = proc.poll()
    if return_code is not None:
        pid_path.unlink(missing_ok=True)
        fail(
            f"instance={instance} exited during startup (rc={return_code}). "
            f"Check log: {log_path}"
        )

    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    log(f"Started instance={instance} in background")
    log(f"PID={proc.pid}")
    log(f"LOG={log_path}")
    log(f"PID_FILE={pid_path}")
    log(f"To stop: python {script_dir / 'stop_model.py'} {instance}")
    return 0


def stop_if_already_running(instance: str) -> None:
    """이미 실행 중인 인스턴스가 있으면 종료한 뒤 포트가 해제될 때까지 대기한다."""
    port = resolve_port_for_instance(instance)
    processes = find_vllm_processes()
    targets = collect_targets(processes, instance, port)
    if not targets:
        return

    pids = ", ".join(str(t["pid"]) for t in targets)
    warn(
        f"Instance {instance} already running on port {port} "
        f"(PID {pids}). Stopping first..."
    )
    stop_instance(processes, instance, port)

    # 포트 해제 대기
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not find_listening_pids_by_port(port):
            break
        time.sleep(0.5)
    else:
        remaining = find_listening_pids_by_port(port)
        if remaining:
            fail(f"Port {port} still occupied after stop (PIDs: {remaining})")

    log(f"Previous instance stopped. Proceeding with fresh start.")


def main() -> None:
    instance = resolve_instance()
    script_dir = Path(__file__).resolve().parent
    stop_if_already_running(instance)
    if RUN_IN_BACKGROUND:
        sys.exit(launch_detached(script_dir, instance))
    sys.exit(launch_foreground(script_dir, instance))


if __name__ == "__main__":
    main()
