"""실행 중인 모델 서빙 프로세스를 종료한다.

사용법:
  python stop_model.py                # 실행 중인 vLLM 인스턴스 표시
  python stop_model.py <instance>     # 특정 인스턴스 종료 (포트 기준)
  python stop_model.py <port>         # 특정 포트의 인스턴스 종료
  python stop_model.py --port <port>  # 특정 포트의 인스턴스 종료
  python stop_model.py all            # 모든 등록 인스턴스 종료
  python stop_model.py <family> <size>

예시:
  python stop_model.py ui-venus
  python stop_model.py ui-venus 30b
  python stop_model.py 8005
  python stop_model.py --port 8005
  python stop_model.py mai-ui
  python stop_model.py all

동작:
  - 인스턴스 이름으로 config/models/<instance>.env의 PORT를 찾는다.
  - runtime/pids/<instance>.pid가 있으면 해당 PID를 우선 종료한다.
  - PID 파일이 없거나 stale이면 해당 포트를 사용 중인 프로세스 또는 vllm 프로세스를 찾는다.
  - SIGTERM 후 일정 시간 내 종료되지 않으면 SIGKILL을 보낸다.
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


VLLM_CMDLINE_MARKER = "vllm.entrypoints.openai.api_server"
SIGTERM_WAIT_SEC = 10


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def read_env_value(path: Path, key: str) -> str:
    """env 파일에서 특정 키 값을 읽는다."""
    if not path.is_file():
        return ""
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


def resolve_config_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or str(script_dir.parent)
    config_root = os.environ.get("CONFIG_ROOT", "").strip() or os.path.join(deploy_vlms_root, "config")
    return Path(config_root)


def resolve_runtime_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or str(script_dir.parent)
    runtime_root = os.environ.get("RUNTIME_ROOT", "").strip() or os.path.join(deploy_vlms_root, "runtime")
    return Path(runtime_root)


def resolve_pid_path(instance: str) -> Path:
    return resolve_runtime_root() / "pids" / f"{instance}.pid"


def resolve_port_for_instance(instance: str) -> str:
    """인스턴스 env 파일에서 PORT 값을 가져온다."""
    config_root = resolve_config_root()
    env_path = config_root / "models" / f"{instance}.env"
    if not env_path.is_file():
        fail(f"Instance config not found: {env_path}")
    port = read_env_value(env_path, "PORT")
    if not port:
        fail(f"PORT not defined in {env_path}")
    return port


def resolve_served_model_name(instance: str) -> str:
    config_root = resolve_config_root()
    env_path = config_root / "models" / f"{instance}.env"
    return read_env_value(env_path, "SERVED_MODEL_NAME")


def resolve_instance_for_port(port: str) -> str:
    config_root = resolve_config_root() / "models"
    if not config_root.is_dir():
        fail(f"Instance config directory not found: {config_root}")

    matched_instances = []
    for env_path in sorted(config_root.glob("*.env")):
        if read_env_value(env_path, "PORT") == port:
            matched_instances.append(env_path.stem)

    if not matched_instances:
        fail(f"No instance config found for PORT={port}")
    if len(matched_instances) > 1:
        fail(f"Multiple instances share PORT={port}: {', '.join(matched_instances)}")
    return matched_instances[0]


def discover_configured_instances() -> list[tuple[str, str]]:
    config_root = resolve_config_root() / "models"
    if not config_root.is_dir():
        return []

    instances = []
    for env_path in sorted(config_root.glob("*.env")):
        port = read_env_value(env_path, "PORT")
        if not port:
            continue
        instances.append((env_path.stem, port))
    return instances


def find_vllm_processes() -> list[dict]:
    """실행 중인 vLLM 프로세스를 /proc에서 찾는다."""
    results = []
    proc_path = Path("/proc")

    if not proc_path.is_dir():
        # /proc가 없는 경우 (macOS 등) ps 명령 사용
        return _find_vllm_processes_via_ps()

    for pid_dir in proc_path.iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        cmdline_path = pid_dir / "cmdline"
        try:
            cmdline_raw = cmdline_path.read_bytes()
        except (PermissionError, FileNotFoundError, OSError):
            continue
        if not cmdline_raw:
            continue
        cmdline_parts = cmdline_raw.decode("utf-8", errors="replace").split("\0")
        cmdline = " ".join(cmdline_parts)
        if VLLM_CMDLINE_MARKER in cmdline:
            port = _extract_port_from_cmdline(cmdline_parts)
            served_name = _extract_served_name_from_cmdline(cmdline_parts)
            results.append({
                "pid": pid,
                "port": port,
                "served_model_name": served_name,
                "cmdline": cmdline.strip(),
            })
    return results


def _find_vllm_processes_via_ps() -> list[dict]:
    """ps 명령으로 vLLM 프로세스를 찾는다 (/proc가 없는 환경용)."""
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    results = []
    for line in result.stdout.splitlines():
        if VLLM_CMDLINE_MARKER not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        cmdline_parts = parts[10:]  # ps aux: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND...
        port = _extract_port_from_cmdline(cmdline_parts)
        served_name = _extract_served_name_from_cmdline(cmdline_parts)
        results.append({
            "pid": pid,
            "port": port,
            "served_model_name": served_name,
            "cmdline": " ".join(cmdline_parts).strip(),
        })
    return results


def _extract_port_from_cmdline(parts: list[str]) -> str:
    for i, part in enumerate(parts):
        if part == "--port" and i + 1 < len(parts):
            return parts[i + 1].strip()
    return ""


def _extract_served_name_from_cmdline(parts: list[str]) -> str:
    for i, part in enumerate(parts):
        if part == "--served-model-name" and i + 1 < len(parts):
            return parts[i + 1].strip()
    return ""


def build_process_record(
    pid: int,
    port: str = "",
    served_model_name: str = "",
    cmdline: str = "",
) -> dict:
    return {
        "pid": pid,
        "port": port,
        "served_model_name": served_model_name,
        "cmdline": cmdline,
    }


def get_cmdline_for_pid(pid: int) -> str:
    proc_path = Path("/proc") / str(pid) / "cmdline"
    if proc_path.is_file():
        try:
            cmdline_raw = proc_path.read_bytes()
        except (PermissionError, FileNotFoundError, OSError):
            cmdline_raw = b""
        if cmdline_raw:
            return " ".join(part for part in cmdline_raw.decode("utf-8", errors="replace").split("\0") if part)

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def read_live_pid_from_file(instance: str) -> int | None:
    pid_path = resolve_pid_path(instance)
    if not pid_path.is_file():
        return None

    raw_value = pid_path.read_text(encoding="utf-8").strip()
    if not raw_value:
        warn(f"PID file is empty: {pid_path}")
        pid_path.unlink(missing_ok=True)
        return None

    try:
        pid = int(raw_value.splitlines()[0].strip())
    except ValueError:
        warn(f"Invalid PID file content: {pid_path}")
        pid_path.unlink(missing_ok=True)
        return None

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        warn(f"Removing stale PID file: {pid_path}")
        pid_path.unlink(missing_ok=True)
        return None
    except PermissionError:
        return pid
    return pid


def find_listening_pids_by_port(port: str) -> list[int]:
    if not port:
        return []

    commands = [
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        ["ss", "-ltnp", f"sport = :{port}"],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

        if cmd[0] == "lsof":
            if result.returncode not in {0, 1}:
                continue
            pids = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    pids.append(int(line))
                except ValueError:
                    continue
            if pids:
                return sorted(set(pids))
            continue

        if result.returncode != 0:
            continue

        pids = []
        for line in result.stdout.splitlines():
            matches = re.findall(r"pid=(\d+)", line)
            for match in matches:
                pids.append(int(match))
        if pids:
            return sorted(set(pids))

    return []


def kill_process(pid: int, sigterm_wait: float = SIGTERM_WAIT_SEC) -> bool:
    """프로세스를 SIGTERM으로 종료 시도 후, 실패 시 SIGKILL."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        warn(f"PID {pid} already exited")
        return True
    except PermissionError:
        fail(f"Permission denied for PID {pid}. Try running with sudo.")

    log(f"Sending SIGTERM to PID {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        log(f"PID {pid} exited before SIGTERM")
        return True
    except PermissionError:
        fail(f"Permission denied for PID {pid}. Try running with sudo.")

    # SIGTERM 후 대기
    deadline = time.monotonic() + sigterm_wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"PID {pid} exited after SIGTERM")
            return True
        time.sleep(0.5)

    # 아직 살아있으면 SIGKILL
    warn(f"PID {pid} did not exit after {sigterm_wait}s, sending SIGKILL...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        log(f"PID {pid} exited before SIGKILL")
        return True
    except PermissionError:
        fail(f"Permission denied for SIGKILL on PID {pid}.")

    time.sleep(1)
    try:
        os.kill(pid, 0)
        warn(f"PID {pid} still alive after SIGKILL")
        return False
    except ProcessLookupError:
        log(f"PID {pid} exited after SIGKILL")
        return True


def list_running(processes: list[dict]) -> None:
    """실행 중인 vLLM 인스턴스를 표시한다."""
    if not processes:
        log("No running vLLM instances found.")
        return

    log(f"Found {len(processes)} running vLLM instance(s):")
    for proc in processes:
        parts = [f"  PID={proc['pid']}"]
        if proc["port"]:
            parts.append(f"PORT={proc['port']}")
        if proc["served_model_name"]:
            parts.append(f"MODEL={proc['served_model_name']}")
        print(" ".join(parts))


def collect_targets(processes: list[dict], instance: str, port: str) -> list[dict]:
    """PID 파일, vLLM cmdline, 포트 리스너 기준으로 종료 대상을 수집한다."""
    served_name = resolve_served_model_name(instance)
    targets_by_pid: dict[int, dict] = {}

    pid = read_live_pid_from_file(instance)
    if pid is not None:
        targets_by_pid[pid] = build_process_record(
            pid=pid,
            port=port,
            served_model_name=served_name,
            cmdline=get_cmdline_for_pid(pid),
        )

    for proc in processes:
        if proc["port"] == port or (served_name and proc["served_model_name"] == served_name):
            targets_by_pid.setdefault(proc["pid"], proc)

    for pid in find_listening_pids_by_port(port):
        targets_by_pid.setdefault(
            pid,
            build_process_record(
                pid=pid,
                port=port,
                served_model_name=served_name,
                cmdline=get_cmdline_for_pid(pid),
            ),
        )

    return sorted(targets_by_pid.values(), key=lambda item: item["pid"])


def stop_instance(processes: list[dict], instance: str, port: str) -> None:
    """특정 인스턴스를 종료한다."""
    targets = collect_targets(processes, instance, port)
    if not targets:
        log(f"No running process found for instance={instance} (port={port})")
        list_running(processes)
        return

    all_stopped = True
    for proc in targets:
        log(
            f"Stopping instance={instance}: "
            f"PID={proc['pid']} PORT={proc['port']} MODEL={proc['served_model_name']}"
        )
        if not kill_process(proc["pid"]):
            all_stopped = False

    pid_path = resolve_pid_path(instance)
    if all_stopped:
        pid_path.unlink(missing_ok=True)
    elif pid_path.is_file():
        warn(f"Keeping PID file because some processes are still alive: {pid_path}")


def stop_all(processes: list[dict]) -> None:
    """config/models에 등록된 모든 인스턴스를 종료한다."""
    configured_instances = discover_configured_instances()
    if not configured_instances and not processes:
        log("No configured or running instances to stop.")
        return

    log(f"Stopping all configured instances ({len(configured_instances)})...")
    stopped_instances = 0
    for instance, port in configured_instances:
        targets = collect_targets(processes, instance, port)
        if not targets:
            continue
        stop_instance(processes, instance, port)
        stopped_instances += 1

    remaining_vllm = []
    for proc in processes:
        pid = proc["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        remaining_vllm.append(proc)

    if remaining_vllm:
        warn("Stopping extra vLLM processes that do not match configured instances.")
        for proc in remaining_vllm:
            parts = [f"PID={proc['pid']}"]
            if proc["port"]:
                parts.append(f"PORT={proc['port']}")
            if proc["served_model_name"]:
                parts.append(f"MODEL={proc['served_model_name']}")
            log(f"Stopping {' '.join(parts)}")
            kill_process(proc["pid"])

    if stopped_instances == 0 and not remaining_vllm:
        log("No running instances found to stop.")


def resolve_cli_target() -> tuple[str, str]:
    if len(sys.argv) == 2:
        token = normalize_token(sys.argv[1])
        if token == "all":
            return "all", ""
        if sys.argv[1].strip().isdigit():
            port = sys.argv[1].strip()
            return resolve_instance_for_port(port), port
        return token, resolve_port_for_instance(token)

    if len(sys.argv) == 3:
        first = normalize_token(sys.argv[1])
        if first in {"port", "--port"}:
            port = sys.argv[2].strip()
            if not port.isdigit():
                fail(f"Invalid port: {sys.argv[2]}")
            return resolve_instance_for_port(port), port
        instance = f"{first}-{normalize_token(sys.argv[2])}"
        return instance, resolve_port_for_instance(instance)

    print(__doc__, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    processes = find_vllm_processes()

    if len(sys.argv) < 2:
        list_running(processes)
        if processes:
            print()
            print("Usage:")
            print("  python stop_model.py <instance>   # stop specific instance")
            print("  python stop_model.py <port>       # stop specific port")
            print("  python stop_model.py --port 8005  # stop specific port")
            print("  python stop_model.py all           # stop all instances")
        return

    instance, port = resolve_cli_target()
    if instance == "all":
        stop_all(processes)
        return

    log(f"Instance={instance} -> PORT={port}")
    stop_instance(processes, instance, port)


if __name__ == "__main__":
    main()
