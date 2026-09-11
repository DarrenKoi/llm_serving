"""vLLM 서빙 스크립트 (Python 버전).

bash 실행 권한 문제가 있는 환경(사내 클라우드 등)에서
Python으로 동일한 기능을 수행한다.

사용법:
  python serve_vlm.py <instance>

예시:
  python serve_vlm.py qwen3.8-27b
  python serve_vlm.py mai-ui

환경변수 오버라이드:
  DEPLOY_VLMS_ROOT=/project/.../deploy_vlms
  CONFIG_ROOT=${DEPLOY_VLMS_ROOT}/config
  COMMON_ENV=${CONFIG_ROOT}/common.env
  MODEL_ENV=${CONFIG_ROOT}/models/<instance>.env
  SITE_ENV=${CONFIG_ROOT}/site.env

참고:
  - config/site.env 가 있으면 가장 먼저 읽는다 (MODEL_ROOT 등 사이트 값).
  - common.env를 먼저 읽고 model env를 나중에 읽는다.
  - 따라서 TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS,
    EXTRA_VLLM_ARGS 같은 키도 instance별로 override할 수 있다.
"""

import os
import shlex
import sys
from pathlib import Path


# 마지막 수단: site.env 도 셸 export 도 없을 때 쓸 MODEL_ROOT. 비우면 쓰지 않는다.
# 공개 저장소다 - 실제 경로를 채운 채 커밋하지 말 것(.git/hooks/pre-commit 이 막는다).
MODEL_ROOT_FALLBACK = ""


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def require_file(path: str) -> None:
    if not Path(path).is_file():
        fail(f"Required file not found: {path!r}")


def require_env_file(path: str) -> None:
    if not Path(path).is_file():
        fail(
            f"Required env file not found: {path} "
            "(create or restore config/common.env and config/models/<instance>.env)"
        )


def require_dir(path: str) -> None:
    if not Path(path).is_dir():
        fail(f"Required directory not found: {path!r}")


def load_env_file(path: str, override: bool = True) -> None:
    """단순 KEY=VALUE .env 파일을 os.environ에 로드한다. override=False 면 이미 있는 키는 둔다."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 따옴표 제거
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            # ${VAR} 를 이미 로드된 환경으로 펼친다. MODEL_ROOT 처럼 사이트마다 다른
            # 경로를 .env 한 곳에서만 채우고 config 는 공개본 그대로 두기 위한 것.
            # 못 펼치면 리터럴로 남고, MODEL_ID 의 isabs 검사가 그대로 잡아준다.
            value = os.path.expandvars(value)
            if override or key not in os.environ:
                os.environ[key] = value


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_flag(key: str, default: bool = False) -> bool:
    value = os.environ.get(key, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def env_required(key: str) -> str:
    value = os.environ.get(key, "")
    if not value:
        fail(f"{key} is required (set in common.env or model .env)")
    return value


def split_cuda_visible_devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    instance = sys.argv[1]

    # 경로 설정
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = env("DEPLOY_VLMS_ROOT") or str(script_dir.parent)
    config_root = env("CONFIG_ROOT") or os.path.join(deploy_vlms_root, "config")
    common_env = env("COMMON_ENV") or os.path.join(config_root, "common.env")
    model_env = env("MODEL_ENV") or os.path.join(config_root, "models", f"{instance}.env")

    os.environ["DEPLOY_VLMS_ROOT"] = deploy_vlms_root
    os.environ["CONFIG_ROOT"] = config_root

    # config/site.env (MODEL_ROOT 등 사이트 값) 를 common.env 보다 먼저 읽는다. 아래
    # common.env 의 ALLOWED_MODEL_ROOT=${MODEL_ROOT} 와 model env 의 MODEL_ID 가
    # 여기서 펼쳐진다. start_all / start_model 은 이 스크립트를 그대로 부르므로
    # 셸에서 export 하지 않아도 된다. 이미 export 된 키는 둔다(셸이 site.env 보다
    # 우선, flask_api 와 같은 규칙). 없으면 건너뛴다(공개 체크아웃).
    site_env = env("SITE_ENV") or os.path.join(config_root, "site.env")
    if os.path.isfile(site_env):
        load_env_file(site_env, override=False)
    if MODEL_ROOT_FALLBACK:
        os.environ.setdefault("MODEL_ROOT", MODEL_ROOT_FALLBACK)
    # 로그만 보고 어느 파일을 읽었고 값이 어디서 왔는지 알 수 있게 한다.
    log(f"config: root={config_root} common={common_env} model={model_env}")
    log(
        f"site env: {site_env} ({'loaded' if os.path.isfile(site_env) else 'missing'}), "
        f"MODEL_ROOT={os.environ.get('MODEL_ROOT', '<unset>')}"
    )

    # env 파일 로드
    require_env_file(common_env)
    require_env_file(model_env)
    load_env_file(common_env)
    load_env_file(model_env)

    # 필수 변수
    model_id = env_required("MODEL_ID")
    served_model_name = env_required("SERVED_MODEL_NAME")
    port = env_required("PORT")
    gpu_id = env_required("GPU_ID")

    # 기본값
    host = env("HOST") or "127.0.0.1"
    dtype = env("DTYPE") or "bfloat16"
    gpu_memory_utilization = env("GPU_MEMORY_UTILIZATION") or "0.80"
    max_model_len = env("MAX_MODEL_LEN") or "8192"
    max_num_seqs = env("MAX_NUM_SEQS") or "8"
    tensor_parallel_size = env("TENSOR_PARALLEL_SIZE") or "1"
    trust_remote_code = env("TRUST_REMOTE_CODE") or "1"
    limit_mm_per_prompt = env("LIMIT_MM_PER_PROMPT") or '{"image": 1}'
    allowed_model_root = env_required("ALLOWED_MODEL_ROOT")
    # site.env 의 팀 공용 키. common.env 에 API_KEY 줄을 따로 두지 않는다 - 두 벌은 어긋난다.
    api_key = env("VLLM_API_KEY")
    chat_template = env("CHAT_TEMPLATE")
    if chat_template and not os.path.isabs(chat_template):
        chat_template = os.path.join(config_root, chat_template)
    max_num_batched_tokens = env("MAX_NUM_BATCHED_TOKENS")
    extra_vllm_args = env("EXTRA_VLLM_ARGS")

    # MODEL_ID 검증: 절대경로 + 디렉토리 존재
    if not os.path.isabs(model_id):
        fail(f"MODEL_ID must be an absolute local path: {model_id}")
    require_dir(model_id)
    model_id_real = str(Path(model_id).resolve())
    model_dir = Path(model_id_real)

    require_file(str(model_dir / "config.json"))

    # 모델 경로가 허용된 루트 아래인지 검증 (오프라인 환경 - 항상 켠다)
    require_dir(allowed_model_root)
    allowed_model_root_real = Path(allowed_model_root).resolve()
    if not model_dir.is_relative_to(allowed_model_root_real):
        fail(f"MODEL_ID must stay under ALLOWED_MODEL_ROOT={allowed_model_root_real}: {model_id_real}")

    # 바깥으로 나가는 프록시는 항상 끊는다
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(var, None)

    # HF 토큰 제거
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(var, None)

    # 오프라인/텔레메트리 설정
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["VLLM_DO_NOT_TRACK"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    visible_devices = split_cuda_visible_devices(gpu_id)
    if visible_devices and len(visible_devices) < int(tensor_parallel_size):
        fail(
            "CUDA_VISIBLE_DEVICES count is smaller than TENSOR_PARALLEL_SIZE: "
            f"GPU_ID={gpu_id}, tensor_parallel_size={tensor_parallel_size}"
        )

    # HF_HOME 디렉토리 생성
    hf_home = env("HF_HOME")
    if hf_home:
        os.makedirs(hf_home, exist_ok=True)

    # vllm do_not_track 파일 생성
    vllm_config_dir = Path.home() / ".config" / "vllm"
    vllm_config_dir.mkdir(parents=True, exist_ok=True)
    (vllm_config_dir / "do_not_track").touch()

    # vllm serve 명령 구성
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", host,
        "--port", port,
        "--served-model-name", served_model_name,
        "--dtype", dtype,
        "--gpu-memory-utilization", gpu_memory_utilization,
        "--max-model-len", max_model_len,
        "--max-num-seqs", max_num_seqs,
        "--tensor-parallel-size", tensor_parallel_size,
    ]

    if trust_remote_code == "1":
        cmd.append("--trust-remote-code")

    if limit_mm_per_prompt:
        cmd.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])

    if max_num_batched_tokens:
        cmd.extend(["--max-num-batched-tokens", max_num_batched_tokens])

    if chat_template:
        require_file(chat_template)
        cmd.extend(["--chat-template", chat_template])

    if api_key:
        cmd.extend(["--api-key", api_key])

    if extra_vllm_args:
        cmd.extend(shlex.split(extra_vllm_args))

    # 로그 출력
    log(f"Starting instance={instance}")
    log(f"DEPLOY_VLMS_ROOT={deploy_vlms_root}")
    log(f"CONFIG_ROOT={config_root}")
    log(f"MODEL_ID={model_id_real}")
    log(f"SERVED_MODEL_NAME={served_model_name}")
    log(f"HOST={host} PORT={port} GPU_ID={gpu_id}")
    log(
        "DTYPE="
        f"{dtype} GPU_MEMORY_UTILIZATION={gpu_memory_utilization} "
        f"MAX_MODEL_LEN={max_model_len} MAX_NUM_SEQS={max_num_seqs} "
        f"TENSOR_PARALLEL_SIZE={tensor_parallel_size}"
    )
    if extra_vllm_args:
        log(f"EXTRA_VLLM_ARGS={extra_vllm_args}")
    log(f"HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1")
    log(f"VLLM_DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1")

    # 진단용: 검증과 argv 조립까지만 하고 vllm 은 띄우지 않는다 (diagnose_paths.py 가 쓴다).
    if env_flag("SERVE_VLM_DRY_RUN"):
        log(f"DRY RUN - 여기서 멈춘다. 실행했을 명령: {shlex.join(cmd)}")
        return

    # vllm 실행 (exec 대체: 현재 프로세스를 대체)
    os.execvpe(cmd[0], cmd, os.environ)


if __name__ == "__main__":
    main()
