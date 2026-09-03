"""다중 사이즈 variant model env 파일을 생성한다.

사용법:
  python prepare_variant_envs.py
  python prepare_variant_envs.py mai-ui
  python prepare_variant_envs.py mai-ui-30b

기본 동작:
  - config/common.env가 없으면 기본값으로 생성한다.
  - config/models/<instance>.env를 생성한다.
  - 기존 파일은 기본적으로 덮어쓰지 않는다.

환경변수:
  DEPLOY_VLMS_ROOT=/project/.../deploy_vlms
  CONFIG_ROOT=${DEPLOY_VLMS_ROOT}/config
  MODEL_ROOT=/project/.../data/models
  OVERWRITE_EXISTING=1
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ROOT = "/path/to/models"
DEFAULT_COMMON_ENV_TEXT = """# Bind to localhost by default.
# If another internal machine must access this server, change this to the GPU server's company-internal IP
# and restrict inbound access with firewall/security-group rules.
HOST=127.0.0.1

DTYPE=bfloat16
GPU_MEMORY_UTILIZATION=0.80
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=8
TENSOR_PARALLEL_SIZE=1

# For 2-3 small models on one GPU, prefer the share rule below instead of a fixed value.
# AUTO_TUNE_GPU_MEMORY_UTILIZATION=1
# COLOCATED_MODELS_PER_GPU=2
# GPU_TOTAL_MEMORY_GIB=140
# GPU_SHARED_RESERVE_GIB=8
# GPU_PROCESS_RESERVE_GIB=4

# Keep empty unless you want internal API auth between company machines.
API_KEY=

# Security/offline controls
STRICT_OFFLINE=1
DISABLE_OUTBOUND_PROXIES=1
CREATE_VLLM_DO_NOT_TRACK_FILE=1
ALLOWED_MODEL_ROOT=/path/to/models
"""


@dataclass(frozen=True)
class ModelVariant:
    instance: str
    family: str
    size: str
    model_dir_placeholder: str
    served_model_name: str
    port: str
    gpu_id: str
    tensor_parallel_size: str
    gpu_memory_utilization: str
    max_model_len: str
    max_num_seqs: str
    extra_vllm_args_example: str = "--kv-cache-memory-bytes 40G"


MODEL_VARIANTS = [
    ModelVariant(
        instance="mai-ui-2b",
        family="mai-ui",
        size="2b",
        model_dir_placeholder="SET_ME_MAI_UI_2B",
        served_model_name="mai-ui-2b",
        port="8202",
        gpu_id="0",
        tensor_parallel_size="1",
        gpu_memory_utilization="0.65",
        max_model_len="4096",
        max_num_seqs="16",
    ),
    ModelVariant(
        instance="mai-ui-7b",
        family="mai-ui",
        size="7b",
        model_dir_placeholder="SET_ME_MAI_UI_7B",
        served_model_name="mai-ui-7b",
        port="8207",
        gpu_id="1",
        tensor_parallel_size="1",
        gpu_memory_utilization="0.75",
        max_model_len="8192",
        max_num_seqs="8",
    ),
    ModelVariant(
        instance="mai-ui-30b",
        family="mai-ui",
        size="30b",
        model_dir_placeholder="SET_ME_MAI_UI_30B",
        served_model_name="mai-ui-30b",
        port="8230",
        gpu_id="0,1",
        tensor_parallel_size="2",
        gpu_memory_utilization="0.88",
        max_model_len="8192",
        max_num_seqs="4",
    ),
]


def log(message: str) -> None:
    print(f"[INFO] {message}")


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def selected_variants() -> list[ModelVariant]:
    requested = {normalize_token(arg) for arg in sys.argv[1:] if arg.strip()}
    if not requested:
        return MODEL_VARIANTS

    selected = [
        variant
        for variant in MODEL_VARIANTS
        if variant.family in requested or variant.instance in requested or variant.size in requested
    ]
    if selected:
        return selected

    print(__doc__, file=sys.stderr)
    sys.exit(1)


def ensure_common_env(config_root: Path) -> None:
    common_env = config_root / "common.env"
    if common_env.is_file():
        return

    common_env.parent.mkdir(parents=True, exist_ok=True)
    common_env.write_text(DEFAULT_COMMON_ENV_TEXT, encoding="utf-8")
    log(f"Created default common env: {common_env}")


def build_env_text(variant: ModelVariant, model_root: str) -> str:
    model_id = f"{model_root.rstrip('/')}/{variant.model_dir_placeholder}"
    lines = [
        f"# Size variant: {variant.family} {variant.size}",
        "# Replace MODEL_ID with the actual local model directory before serving.",
        "# Keys duplicated from common.env override the shared defaults for this instance only.",
        f"MODEL_ID={model_id}",
        f"SERVED_MODEL_NAME={variant.served_model_name}",
        f"PORT={variant.port}",
        f"GPU_ID={variant.gpu_id}",
        f"TENSOR_PARALLEL_SIZE={variant.tensor_parallel_size}",
        "TRUST_REMOTE_CODE=1",
        'LIMIT_MM_PER_PROMPT={"image": 1}',
        "CHAT_TEMPLATE=",
        f"GPU_MEMORY_UTILIZATION={variant.gpu_memory_utilization}",
        f"MAX_MODEL_LEN={variant.max_model_len}",
        f"MAX_NUM_SEQS={variant.max_num_seqs}",
        "# To colocate 2-3 small models on one GPU, switch to the auto share rule below.",
        "# AUTO_TUNE_GPU_MEMORY_UTILIZATION=1",
        "# COLOCATED_MODELS_PER_GPU=2",
        "# GPU_TOTAL_MEMORY_GIB=140",
        "# GPU_SHARED_RESERVE_GIB=8",
        "# GPU_PROCESS_RESERVE_GIB=4",
    ]

    lines.append(f"# EXTRA_VLLM_ARGS={variant.extra_vllm_args_example}")
    lines.append("EXTRA_VLLM_ARGS=")

    lines.append("")
    return "\n".join(lines)


def write_variant_env(
    variant: ModelVariant,
    model_root: str,
    models_root: Path,
    overwrite_existing: bool,
) -> None:
    dest = models_root / f"{variant.instance}.env"
    if dest.exists() and not overwrite_existing:
        log(f"Skip existing env: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(build_env_text(variant, model_root), encoding="utf-8")
    log(f"Wrote {dest}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or str(script_dir.parent)
    config_root = Path(os.environ.get("CONFIG_ROOT", "").strip() or Path(deploy_vlms_root) / "config")
    model_root = os.environ.get("MODEL_ROOT", "").strip() or DEFAULT_MODEL_ROOT
    overwrite_existing = os.environ.get("OVERWRITE_EXISTING", "").strip() == "1"

    ensure_common_env(config_root)

    models_root = config_root / "models"
    variants = selected_variants()
    for variant in variants:
        write_variant_env(variant, model_root, models_root, overwrite_existing)

    log(f"Prepared {len(variants)} variant env file(s) under {models_root}")


if __name__ == "__main__":
    main()
