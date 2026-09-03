"""vLLM 서빙 스크립트 (Python 버전).

bash 실행 권한 문제가 있는 환경(사내 클라우드 등)에서
Python으로 동일한 기능을 수행한다.

사용법:
  python serve_vlm.py <instance>

예시:
  python serve_vlm.py ui-venus
  python serve_vlm.py ui-venus-30b
  python serve_vlm.py mai-ui-7b

환경변수 오버라이드:
  DEPLOY_VLMS_ROOT=/project/.../deploy_vlms
  CONFIG_ROOT=${DEPLOY_VLMS_ROOT}/config
  COMMON_ENV=${CONFIG_ROOT}/common.env
  MODEL_ENV=${CONFIG_ROOT}/models/<instance>.env

참고:
  - common.env를 먼저 읽고 model env를 나중에 읽는다.
  - 따라서 TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS,
    EXTRA_VLLM_ARGS 같은 키도 instance별로 override할 수 있다.
"""

import importlib.metadata
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GIB = 1024 ** 3


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def require_file(path: str) -> None:
    if not Path(path).is_file():
        fail(f"Required file not found: {path}")


def require_env_file(path: str) -> None:
    if not Path(path).is_file():
        fail(
            f"Required env file not found: {path} "
            "(create or restore config/common.env and config/models/<instance>.env)"
        )


def require_dir(path: str) -> None:
    if not Path(path).is_dir():
        fail(f"Required directory not found: {path}")


def load_env_file(path: str) -> None:
    """단순 KEY=VALUE .env 파일을 os.environ에 로드한다."""
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
            os.environ[key] = value


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_flag(key: str, default: bool = False) -> bool:
    value = os.environ.get(key, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_limit_mm(value: str) -> str:
    """old key=val,... 형식을 vLLM 0.8+ JSON 형식으로 변환."""
    try:
        json.loads(value)
        return value
    except (json.JSONDecodeError, ValueError):
        pass
    # "image=1,video=2" → {"image": 1, "video": 2}
    pairs = {}
    for item in value.split(","):
        if "=" in item:
            k, _, v = item.partition("=")
            pairs[k.strip()] = int(v.strip())
    if pairs:
        return json.dumps(pairs)
    return value


def env_required(key: str) -> str:
    value = os.environ.get(key, "")
    if not value:
        fail(f"{key} is required (set in common.env or model .env)")
    return value


def split_cuda_visible_devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def bytes_to_gib(value: int | float) -> float:
    return float(value) / GIB


def require_any_file(paths: list[Path], description: str) -> None:
    if any(path.is_file() for path in paths):
        return
    fail(f"{description} not found. Checked: {', '.join(str(path) for path in paths)}")


def read_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    require_file(str(config_path))
    return read_json_file(config_path)


def detect_model_family(config: dict[str, Any]) -> tuple[list[str], str]:
    architectures = [str(item) for item in config.get("architectures") or []]
    model_type = str(config.get("model_type") or "")
    return architectures, model_type


def resolve_text_config(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("text_config", "language_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and nested.get("hidden_size"):
            return nested
    return config


def dtype_nbytes(dtype: str) -> int:
    normalized = dtype.strip().lower()
    if normalized in {"half", "float16", "fp16", "bfloat16", "bf16"}:
        return 2
    if normalized in {"float", "float32", "fp32"}:
        return 4
    if normalized.startswith("fp8") or normalized.startswith("float8"):
        return 1
    return 2


def estimate_kv_cache_bytes_per_token(config: dict[str, Any], dtype: str) -> int | None:
    text_config = resolve_text_config(config)
    num_hidden_layers = int(text_config.get("num_hidden_layers") or text_config.get("n_layer") or 0)
    hidden_size = int(text_config.get("hidden_size") or text_config.get("n_embd") or 0)
    num_attention_heads = int(text_config.get("num_attention_heads") or text_config.get("n_head") or 0)
    num_key_value_heads = int(
        text_config.get("num_key_value_heads")
        or text_config.get("multi_query_group_num")
        or num_attention_heads
        or 0
    )
    head_dim = int(text_config.get("head_dim") or 0)

    if not head_dim and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    if not (num_hidden_layers and num_attention_heads and num_key_value_heads and head_dim):
        return None

    return 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_nbytes(dtype)


def estimate_model_weight_bytes(model_dir: Path) -> int:
    index_candidates = [
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    ]
    for index_path in index_candidates:
        if not index_path.is_file():
            continue
        weight_map = read_json_file(index_path).get("weight_map") or {}
        if not isinstance(weight_map, dict):
            continue

        total_bytes = 0
        seen_files: set[Path] = set()
        for relative_name in weight_map.values():
            shard_path = model_dir / str(relative_name)
            if shard_path in seen_files:
                continue
            require_file(str(shard_path))
            total_bytes += shard_path.stat().st_size
            seen_files.add(shard_path)
        if total_bytes:
            return total_bytes

    weight_files = [*sorted(model_dir.glob("*.safetensors")), *sorted(model_dir.glob("*.bin"))]
    if not weight_files:
        fail(f"Could not estimate model weight size under {model_dir}")
    return sum(path.stat().st_size for path in weight_files)


def detect_gpu_total_memory_gib(visible_devices: list[str], override_gib: str) -> float:
    if override_gib.strip():
        return float(override_gib)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        warn("Could not query nvidia-smi; falling back to GPU_TOTAL_MEMORY_GIB=140")
        return 140.0

    totals_by_index: dict[str, float] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        index, _, memory_total_mib = line.partition(",")
        index = index.strip()
        memory_total_mib = memory_total_mib.strip()
        if not index or not memory_total_mib:
            continue
        totals_by_index[index] = float(memory_total_mib) / 1024.0

    if not totals_by_index:
        warn("nvidia-smi returned no GPU memory rows; falling back to GPU_TOTAL_MEMORY_GIB=140")
        return 140.0

    if visible_devices:
        selected = [totals_by_index[item] for item in visible_devices if item in totals_by_index]
        if selected:
            return min(selected)

    return min(totals_by_index.values())


@dataclass(frozen=True)
class MemorySizing:
    gpu_total_memory_gib: float
    colocated_models_per_gpu: int
    gpu_shared_reserve_gib: float
    gpu_process_reserve_gib: float
    weight_per_gpu_gib: float
    kv_cache_per_gpu_gib: float
    per_process_share_gib: float
    vllm_budget_gib: float
    min_required_total_gib: float
    min_required_utilization: float
    recommended_utilization: float
    suggested_max_num_seqs: int | None
    kv_cache_estimated: bool


def calculate_memory_sizing(
    *,
    weight_bytes: int,
    model_config: dict[str, Any],
    dtype: str,
    max_model_len: int,
    max_num_seqs: int,
    tensor_parallel_size: int,
    gpu_total_memory_gib: float,
    colocated_models_per_gpu: int,
    gpu_shared_reserve_gib: float,
    gpu_process_reserve_gib: float,
) -> MemorySizing:
    tp_size = max(1, tensor_parallel_size)
    colocated = max(1, colocated_models_per_gpu)
    weight_per_gpu_gib = bytes_to_gib(weight_bytes) / tp_size
    kv_bytes_per_token = estimate_kv_cache_bytes_per_token(model_config, dtype)

    kv_cache_per_gpu_gib = 0.0
    kv_cache_estimated = kv_bytes_per_token is not None
    if kv_bytes_per_token is not None and max_model_len > 0 and max_num_seqs > 0:
        total_tokens = max_model_len * max_num_seqs
        kv_cache_per_gpu_gib = bytes_to_gib(kv_bytes_per_token * total_tokens) / tp_size

    per_process_share_gib = max((gpu_total_memory_gib - gpu_shared_reserve_gib) / colocated, 0.0)
    vllm_budget_gib = max(per_process_share_gib - gpu_process_reserve_gib, 0.0)
    min_required_total_gib = weight_per_gpu_gib + kv_cache_per_gpu_gib + gpu_process_reserve_gib
    min_required_utilization = 0.0
    if gpu_total_memory_gib > 0:
        min_required_utilization = (weight_per_gpu_gib + kv_cache_per_gpu_gib) / gpu_total_memory_gib
    recommended_utilization = 0.0
    if gpu_total_memory_gib > 0:
        recommended_utilization = max(min(vllm_budget_gib / gpu_total_memory_gib, 0.95), 0.10)

    suggested_max_num_seqs = None
    if kv_cache_per_gpu_gib > 0 and max_num_seqs > 0:
        kv_per_seq_gib = kv_cache_per_gpu_gib / max_num_seqs
        allowed_kv_gib = max(vllm_budget_gib - weight_per_gpu_gib, 0.0)
        if kv_per_seq_gib > 0:
            suggested_max_num_seqs = max(int(allowed_kv_gib // kv_per_seq_gib), 1)

    return MemorySizing(
        gpu_total_memory_gib=gpu_total_memory_gib,
        colocated_models_per_gpu=colocated,
        gpu_shared_reserve_gib=gpu_shared_reserve_gib,
        gpu_process_reserve_gib=gpu_process_reserve_gib,
        weight_per_gpu_gib=weight_per_gpu_gib,
        kv_cache_per_gpu_gib=kv_cache_per_gpu_gib,
        per_process_share_gib=per_process_share_gib,
        vllm_budget_gib=vllm_budget_gib,
        min_required_total_gib=min_required_total_gib,
        min_required_utilization=min_required_utilization,
        recommended_utilization=recommended_utilization,
        suggested_max_num_seqs=suggested_max_num_seqs,
        kv_cache_estimated=kv_cache_estimated,
    )


def ensure_qwen25_vl_runtime_ready(model_dir: Path, instance: str) -> str:
    """Qwen2.5-VL 계열(UI-TARS 포함) 사전 점검."""
    if importlib.util.find_spec("transformers.models.qwen2_5_vl") is None:
        try:
            transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            transformers_version = "not-installed"
        fail(
            f"{instance} requires Qwen2.5-VL runtime support, but "
            f"transformers.models.qwen2_5_vl is unavailable (transformers={transformers_version})."
        )

    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "not-installed"
    log(f"Detected Qwen2.5-VL runtime (vllm={vllm_version})")

    require_file(str(model_dir / "preprocessor_config.json"))
    require_file(str(model_dir / "tokenizer_config.json"))
    require_any_file(
        [model_dir / "tokenizer.json", model_dir / "tokenizer.model"],
        "Tokenizer file",
    )
    require_any_file(
        [model_dir / "model.safetensors.index.json", *sorted(model_dir.glob("*.safetensors"))],
        "Model weights",
    )

    model_chat_template = model_dir / "chat_template.json"
    if model_chat_template.is_file():
        log(f"Using model-provided chat template: {model_chat_template}")
        return str(model_chat_template)

    log("Model chat_template.json not found; continuing without explicit chat template")
    return ""


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
    gpu_memory_utilization_raw = env("GPU_MEMORY_UTILIZATION") or "0.80"
    auto_tune_gpu_memory_utilization = (
        gpu_memory_utilization_raw.strip().lower() == "auto"
        or env_flag("AUTO_TUNE_GPU_MEMORY_UTILIZATION")
    )
    gpu_memory_utilization = "0.80" if gpu_memory_utilization_raw.strip().lower() == "auto" else gpu_memory_utilization_raw
    max_model_len = env("MAX_MODEL_LEN") or "8192"
    max_num_seqs = env("MAX_NUM_SEQS") or "8"
    tensor_parallel_size = env("TENSOR_PARALLEL_SIZE") or "1"
    data_parallel_size = env("DATA_PARALLEL_SIZE")
    trust_remote_code = env("TRUST_REMOTE_CODE") or "1"
    limit_mm_per_prompt = env("LIMIT_MM_PER_PROMPT") or '{"image": 1}'
    strict_offline = env("STRICT_OFFLINE") or "1"
    disable_outbound_proxies = env("DISABLE_OUTBOUND_PROXIES") or "1"
    allowed_model_root = env("ALLOWED_MODEL_ROOT") or "/path/to/models"
    create_vllm_do_not_track_file = env("CREATE_VLLM_DO_NOT_TRACK_FILE") or "1"
    api_key = env("API_KEY")
    chat_template = env("CHAT_TEMPLATE")
    if chat_template and not os.path.isabs(chat_template):
        chat_template = os.path.join(config_root, chat_template)
    mm_encoder_tp_mode = env("MM_ENCODER_TP_MODE")
    model_impl = env("MODEL_IMPL")
    max_num_batched_tokens = env("MAX_NUM_BATCHED_TOKENS")
    extra_vllm_args = env("EXTRA_VLLM_ARGS")
    colocated_models_per_gpu = int(env("COLOCATED_MODELS_PER_GPU") or "1")
    gpu_total_memory_gib_override = env("GPU_TOTAL_MEMORY_GIB")
    gpu_shared_reserve_gib = float(env("GPU_SHARED_RESERVE_GIB") or "8")
    gpu_process_reserve_gib = float(env("GPU_PROCESS_RESERVE_GIB") or "4")

    # MODEL_ID 검증: 절대경로 + 디렉토리 존재
    if not os.path.isabs(model_id):
        fail(f"MODEL_ID must be an absolute local path: {model_id}")
    require_dir(model_id)
    model_id_real = str(Path(model_id).resolve())
    model_dir = Path(model_id_real)

    model_config = read_model_config(model_dir)
    architectures, model_type = detect_model_family(model_config)
    is_qwen25_vl = model_type == "qwen2_5_vl" or any("Qwen2_5_VL" in item for item in architectures)

    # STRICT_OFFLINE: 모델 경로가 허용된 루트 아래인지 검증
    if strict_offline == "1":
        require_dir(allowed_model_root)
        allowed_model_root_real = str(Path(allowed_model_root).resolve())
        if not (model_id_real == allowed_model_root_real or model_id_real.startswith(allowed_model_root_real + "/")):
            fail(f"MODEL_ID must stay under ALLOWED_MODEL_ROOT={allowed_model_root_real}: {model_id_real}")

    # 프록시 비활성화
    if disable_outbound_proxies == "1":
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
    requested_parallelism = max(
        int(tensor_parallel_size or "1"),
        int(data_parallel_size or "1"),
    )
    if visible_devices and len(visible_devices) < requested_parallelism:
        fail(
            "CUDA_VISIBLE_DEVICES count is smaller than requested parallelism: "
            f"GPU_ID={gpu_id}, tensor_parallel_size={tensor_parallel_size}, "
            f"data_parallel_size={data_parallel_size or '1'}"
        )

    memory_sizing: MemorySizing | None = None
    if auto_tune_gpu_memory_utilization or colocated_models_per_gpu > 1:
        weight_bytes = estimate_model_weight_bytes(model_dir)
        memory_sizing = calculate_memory_sizing(
            weight_bytes=weight_bytes,
            model_config=model_config,
            dtype=dtype,
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_total_memory_gib=detect_gpu_total_memory_gib(visible_devices, gpu_total_memory_gib_override),
            colocated_models_per_gpu=colocated_models_per_gpu,
            gpu_shared_reserve_gib=gpu_shared_reserve_gib,
            gpu_process_reserve_gib=gpu_process_reserve_gib,
        )
        log(
            "Memory sizing: "
            f"GPU_TOTAL_MEMORY_GIB={memory_sizing.gpu_total_memory_gib:.1f} "
            f"COLOCATED_MODELS_PER_GPU={memory_sizing.colocated_models_per_gpu} "
            f"PER_PROCESS_SHARE_GIB={memory_sizing.per_process_share_gib:.1f} "
            f"VLLM_BUDGET_GIB={memory_sizing.vllm_budget_gib:.1f} "
            f"WEIGHTS_PER_GPU_GIB={memory_sizing.weight_per_gpu_gib:.1f} "
            f"KV_CACHE_PER_GPU_GIB={memory_sizing.kv_cache_per_gpu_gib:.1f}"
        )
        if not memory_sizing.kv_cache_estimated:
            warn(
                "Could not estimate KV cache from config.json; recommendation uses the per-GPU share rule "
                "without a KV feasibility check"
            )

        if memory_sizing.min_required_total_gib > memory_sizing.per_process_share_gib:
            parts = [
                "Estimated memory requirement does not fit the colocated plan: "
                f"required_per_process={memory_sizing.min_required_total_gib:.1f}GiB "
                f"> share_per_process={memory_sizing.per_process_share_gib:.1f}GiB.",
            ]
            if memory_sizing.suggested_max_num_seqs is not None:
                parts.append(
                    f"Try MAX_NUM_SEQS<={memory_sizing.suggested_max_num_seqs} at MAX_MODEL_LEN={max_model_len}, "
                    f"or reduce COLOCATED_MODELS_PER_GPU from {colocated_models_per_gpu}."
                )
            message = " ".join(parts)
            if auto_tune_gpu_memory_utilization:
                fail(message)
            warn(message)

        if auto_tune_gpu_memory_utilization:
            gpu_memory_utilization = f"{memory_sizing.recommended_utilization:.2f}"
            log(
                "Auto-tuned GPU_MEMORY_UTILIZATION="
                f"{gpu_memory_utilization} from the per-GPU share rule"
            )
        elif colocated_models_per_gpu > 1:
            warn(
                "When multiple models share one GPU, start near "
                f"GPU_MEMORY_UTILIZATION={memory_sizing.recommended_utilization:.2f} "
                f"(current={gpu_memory_utilization})"
            )

    # HF_HOME 디렉토리 생성
    hf_home = env("HF_HOME")
    if hf_home:
        os.makedirs(hf_home, exist_ok=True)

    # vllm do_not_track 파일 생성
    if create_vllm_do_not_track_file == "1":
        vllm_config_dir = Path.home() / ".config" / "vllm"
        vllm_config_dir.mkdir(parents=True, exist_ok=True)
        (vllm_config_dir / "do_not_track").touch()

    if is_qwen25_vl:
        log(f"Detected Qwen2.5-VL architecture: {architectures or [model_type]}")
        chat_template = chat_template or ensure_qwen25_vl_runtime_ready(model_dir, instance)
        if not limit_mm_per_prompt:
            limit_mm_per_prompt = '{"image": 1, "video": 0}'

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

    if data_parallel_size:
        cmd.extend(["--data-parallel-size", data_parallel_size])

    if trust_remote_code == "1":
        cmd.append("--trust-remote-code")

    if limit_mm_per_prompt:
        limit_mm_per_prompt = _normalize_limit_mm(limit_mm_per_prompt)
        cmd.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])

    if mm_encoder_tp_mode:
        cmd.extend(["--mm-encoder-tp-mode", mm_encoder_tp_mode])

    if model_impl:
        cmd.extend(["--model-impl", model_impl])

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
        f"TENSOR_PARALLEL_SIZE={tensor_parallel_size} "
        f"DATA_PARALLEL_SIZE={data_parallel_size or '1'}"
    )
    if memory_sizing is not None:
        log(
            "Memory sizing summary: "
            f"MIN_REQUIRED_UTILIZATION={memory_sizing.min_required_utilization:.2f} "
            f"RECOMMENDED_UTILIZATION={memory_sizing.recommended_utilization:.2f} "
            f"GPU_SHARED_RESERVE_GIB={memory_sizing.gpu_shared_reserve_gib:.1f} "
            f"GPU_PROCESS_RESERVE_GIB={memory_sizing.gpu_process_reserve_gib:.1f}"
        )
    if architectures:
        log(f"ARCHITECTURES={architectures}")
    if extra_vllm_args:
        log(f"EXTRA_VLLM_ARGS={extra_vllm_args}")
    log(f"STRICT_OFFLINE={strict_offline} DISABLE_OUTBOUND_PROXIES={disable_outbound_proxies}")
    log(f"HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1")
    log(f"VLLM_DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1")

    # vllm 실행 (exec 대체: 현재 프로세스를 대체)
    os.execvpe(cmd[0], cmd, os.environ)


if __name__ == "__main__":
    main()
