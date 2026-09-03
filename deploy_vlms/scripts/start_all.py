"""등록된 VLM 모델을 모두 시작한다.

GPU 배분 (H200 140GB × 2), 2026-09-03 재배치:
  GPU 0: mai-ui (8B, 8002, u=0.45) + paddleocr-vl-1.5 (0.9B, 8004, u=0.20)
         -> 둘 다 workflow_3 전용 소형 모델. 합쳐 0.65, 49GiB 여유.
  GPU 1: qwen3.8-27b (27B BF16, 8006, u=0.90) 단독
         -> 범용 추론/멀티모달. 27B dense 라 대역폭 바운드, 카드를 독점해야 의미가 있다.

시작 순서가 곧 이 리스트 순서다. 큰 모델을 먼저 띄우는 이유는 GPU 가 아니라
**호스트 RAM 16GB** 때문이다 - vLLM 인스턴스 3개(각 API server + EngineCore)가
같은 16GB 를 나눠 쓰므로, 가장 큰 로딩을 먼저 끝내고 나머지를 붙인다.
27B 가중치 로딩은 수 분 걸릴 수 있다. start_all 은 앞 모델이 /v1/models 에
응답할 때까지 기다린 뒤(wait_until_ready, 최대 READY_TIMEOUT_SEC) 다음으로
넘어가므로 로딩 최대 구간이 겹치지 않는다. 이미 떠 있는 인스턴스는 시작 전에
먼저 내린다(stop_if_already_running) - 살아 있는 스택 위에 그대로 실행하면
포트에 답하는 것이 옛 인스턴스라 "준비 완료" 가 거짓이 되기 때문이다.
최종 확인은 check_vlm.py 로 한다.

ui-venus(8001) / ui-tars(8003) / got-ocr(8005) 는 2026-09-03 에 **가중치를
서버에서 삭제**했다. 호스트 RAM 이 16GB 뿐이라 프로세스 수 자체가 제약이므로,
미사용 모델을 띄워두지 않는 것이 GPU 배분보다 큰 레버다.
env 파일 / 기동 스크립트 / flask route 등록도 같은 날 함께 지웠다 - 되살리려면
체크포인트 재반입부터 시작해야 하므로, 껍데기만 남겨 두면 "이름만 되돌리면
된다"는 잘못된 신호가 된다. 복원 근거는 git 이력이다.

vLLM 모델은 serve_vlm.py를 통해 백그라운드로 실행한다.

사용법:
  python start_all.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from check_vlm import check_model
from start_model import stop_if_already_running


STARTUP_POLL_SEC = 2.0
# 다음 모델을 띄우기 전에 앞 모델이 "떴다"고 답할 때까지 기다린다.
# 고정 sleep 이 아닌 이유는 GPU 가 아니라 **호스트 RAM 16GB** 때문이다 - 가중치 로딩과
# torch.compile/CUDA graph 캡처가 각 모델의 RSS 최대 구간인데, 고정 10초는 27B 로딩이
# 끝나기 한참 전이라 세 모델의 최대 구간이 겹친다. 겹치면 OOM killer 가 고른 하나가 죽는다.
READY_TIMEOUT_SEC = 900.0  # 27B BF16(~48GiB) 첫 기동(컴파일 캐시 없음)은 수 분 걸릴 수 있다
READY_POLL_SEC = 5.0

VLLM_MODELS = [
    "qwen3.8-27b",
    "mai-ui",
    "paddleocr-vl-1.5",
]


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


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


def print_gpu_plan(config_root: Path) -> None:
    """시작 전 GPU 배분 계획을 출력한다."""
    log("=" * 60)
    log("GPU 배분 계획 (H200 140GB × 2)")
    log("=" * 60)

    gpu_map: dict[str, list[str]] = {}

    for instance in VLLM_MODELS:
        env_path = config_root / "models" / f"{instance}.env"
        gpu_id = read_env_value(env_path, "GPU_ID") or "?"
        port = read_env_value(env_path, "PORT") or "?"
        served_name = read_env_value(env_path, "SERVED_MODEL_NAME") or instance
        gpu_mem = read_env_value(env_path, "GPU_MEMORY_UTILIZATION")
        auto_tune = read_env_value(env_path, "AUTO_TUNE_GPU_MEMORY_UTILIZATION")

        mem_info = f"u={gpu_mem}" if gpu_mem else ("auto-tune" if auto_tune else "default")
        entry = f"  {served_name:<25s} port={port:<5s} {mem_info} (vLLM)"
        gpu_map.setdefault(f"GPU {gpu_id}", []).append(entry)

    for gpu_label in sorted(gpu_map):
        log(f"{gpu_label}:")
        for line in gpu_map[gpu_label]:
            log(line)
    log("=" * 60)


def resolve_runtime_paths(deploy_vlms_root: Path, instance: str) -> tuple[Path, Path]:
    """로그/PID 경로를 반환한다."""
    runtime_root = deploy_vlms_root / "runtime"
    log_dir = runtime_root / "logs"
    pid_dir = runtime_root / "pids"
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{instance}.log", pid_dir / f"{instance}.pid"


def start_vllm_model(script_dir: Path, deploy_vlms_root: Path, instance: str) -> int:
    """vLLM 모델을 백그라운드로 시작한다. 성공 시 PID, 실패 시 0."""
    cmd = [sys.executable, str(script_dir / "serve_vlm.py"), instance]
    log_path, pid_path = resolve_runtime_paths(deploy_vlms_root, instance)

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
        warn(f"{instance} exited during startup (rc={return_code}). Check log: {log_path}")
        return 0

    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    log(f"Started {instance}: PID={proc.pid}, LOG={log_path}")
    return proc.pid


def process_alive(pid: int) -> bool:
    """PID 생존 확인. 죽은 프로세스를 기다리며 타임아웃을 다 태우지 않기 위한 것."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_until_ready(config_root: Path, instance: str, pid: int) -> bool:
    """모델이 /v1/models 에 응답할 때까지 기다린다. 죽으면 즉시 포기한다."""
    env_path = config_root / "models" / f"{instance}.env"
    port = read_env_value(env_path, "PORT")
    served_name = read_env_value(env_path, "SERVED_MODEL_NAME") or instance
    if not port:
        warn(f"{instance}: PORT 를 읽을 수 없어 준비 대기를 건너뛴다")
        return True

    deadline = time.monotonic() + READY_TIMEOUT_SEC
    log(f"{instance} 준비 대기 중 (port={port}, 최대 {READY_TIMEOUT_SEC:.0f}s)...")
    while time.monotonic() < deadline:
        if not process_alive(pid):
            warn(f"{instance}: 준비되기 전에 프로세스가 종료됐다 (PID={pid})")
            return False
        ok, _ = check_model("127.0.0.1", int(port), served_name)
        if ok:
            waited = READY_TIMEOUT_SEC - (deadline - time.monotonic())
            log(f"{instance} 준비 완료 ({waited:.0f}s)")
            return True
        time.sleep(READY_POLL_SEC)

    warn(f"{instance}: {READY_TIMEOUT_SEC:.0f}s 안에 준비되지 않았다. 로그를 확인할 것")
    return False


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = Path(os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or script_dir.parent)
    config_root = deploy_vlms_root / "config"

    print_gpu_plan(config_root)

    succeeded = []
    failed = []

    # 가장 큰 모델(GPU 1) 부터, 그리고 **한 번에 하나씩만** 로딩한다.
    # 호스트 RAM 16GB 를 3개 인스턴스가 나눠 쓰므로 로딩 최대 구간이 겹치면 안 된다.
    for instance in VLLM_MODELS:
        log(f"Starting {instance}...")
        # 이미 떠 있으면 먼저 내린다(start_model.py 와 같은 가드).
        # 없으면 살아 있는 스택 위에 다시 실행할 때 두 가지가 조용히 깨진다:
        #   (a) wait_until_ready 가 **옛 인스턴스**의 포트 응답을 보고 "준비 완료" 로 거짓 보고
        #   (b) 새 프로세스는 가중치를 다 올린 뒤에야 포트 바인딩에 실패한다
        # (b) 의 로딩 구간이 호스트 RAM 16GB 를 두 배로 밟는 자리다.
        try:
            stop_if_already_running(instance)
        except SystemExit:
            # 포트가 15s 안에 안 풀리는 경우(SIGKILL 후에도 D-state 로 남는 등).
            # 여기서 전체를 멈추면 무관한 production 모델까지 못 뜨므로 이 인스턴스만 실패 처리한다.
            warn(f"{instance}: 기존 인스턴스를 내리지 못해 건너뛴다 (포트 점유 지속)")
            failed.append(instance)
            continue
        pid = start_vllm_model(script_dir, deploy_vlms_root, instance)
        if pid and wait_until_ready(config_root, instance, pid):
            succeeded.append(instance)
        else:
            failed.append(instance)

    # 결과 요약
    log("=" * 60)
    log("시작 결과 요약")
    log("=" * 60)
    if succeeded:
        log(f"성공 ({len(succeeded)}): {', '.join(succeeded)}")
    if failed:
        warn(f"실패 ({len(failed)}): {', '.join(failed)}")
        log("실패한 모델 로그 확인:")
        for instance in failed:
            log_path = deploy_vlms_root / "runtime" / "logs" / f"{instance}.log"
            log(f"  {log_path}")
    log("=" * 60)

    if failed:
        log(f"개별 종료: python {script_dir / 'stop_model.py'} <instance>")
        log(f"전체 종료: python {script_dir / 'stop_model.py'} all")
        sys.exit(1)

    log(f"{len(succeeded)}개 모델 모두 시작 완료")
    log(f"전체 종료: python {script_dir / 'stop_model.py'} all")


if __name__ == "__main__":
    main()
