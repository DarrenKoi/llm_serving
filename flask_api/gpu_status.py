"""nvidia-smi 로 GPU 상태를 읽어 대시보드에 넘긴다.

serve_vlm.py 의 detect_gpu_total_memory_gib 와 같은 방식이다 - csv,noheader,nounits
로 물어보고 nvidia-smi 가 없으면 조용히 물러난다. 개발 PC 에는 GPU 가 없으므로
'없음' 이 에러가 아니라 정상 상태의 하나여야 한다.
"""

import subprocess
from typing import Any

from flask import Blueprint, jsonify

# nvidia-smi 가 답이 없으면 대시보드가 통째로 멈춘다. 폴링 주기보다 짧게 잡는다.
QUERY_TIMEOUT_SEC = 5.0

# 순서가 곧 아래 _parse_row 의 필드 순서다. 같이 고칠 것.
QUERY_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "power.limit",
]

gpu_status_blueprint = Blueprint("gpu_status", __name__)


def _number(raw: str) -> float | None:
    """nvidia-smi 는 못 읽는 값을 '[N/A]' 로 준다 - 숫자가 아니면 None."""
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return None


def _parse_row(line: str) -> dict[str, Any] | None:
    """csv 한 줄을 GPU 한 장으로 바꾼다."""
    cells = [cell.strip() for cell in line.split(",")]
    if len(cells) != len(QUERY_FIELDS):
        return None

    index, name, mem_total, mem_used, util, temp, power_draw, power_limit = cells
    total_mib = _number(mem_total)
    used_mib = _number(mem_used)
    percent = None
    if total_mib and used_mib is not None and total_mib > 0:
        percent = round(used_mib / total_mib * 100.0, 1)

    return {
        "index": index,
        "name": name,
        "memory_total_mib": total_mib,
        "memory_used_mib": used_mib,
        "memory_used_percent": percent,
        "utilization_percent": _number(util),
        "temperature_c": _number(temp),
        "power_draw_w": _number(power_draw),
        "power_limit_w": _number(power_limit),
    }


def query_gpus() -> dict[str, Any]:
    """GPU 목록을 읽는다. 실패는 예외가 아니라 available=False 로 돌려준다."""
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        # 개발 PC. 대시보드는 이걸 '에러' 가 아니라 '해당 없음' 으로 그린다.
        return {"available": False, "reason": "nvidia-smi not found", "gpus": []}
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "nvidia-smi timed out", "gpus": []}
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"exit {exc.returncode}"
        return {"available": False, "reason": f"nvidia-smi failed: {detail}", "gpus": []}

    gpus = [row for row in (_parse_row(line) for line in result.stdout.splitlines() if line.strip()) if row]
    if not gpus:
        return {"available": False, "reason": "nvidia-smi returned no GPU rows", "gpus": []}
    return {"available": True, "reason": None, "gpus": gpus}


def build_gpu_status_payload() -> dict[str, Any]:
    """/api/health 와 대시보드가 함께 쓰는 페이로드."""
    snapshot = query_gpus()
    return {
        "service": "gpu_status",
        "status": "ok",
        "base_path": "/api/gpu_status",
        **snapshot,
    }


@gpu_status_blueprint.route("/", methods=["GET"], strict_slashes=False)
def home():
    """GPU 스냅샷."""
    return jsonify(build_gpu_status_payload())


@gpu_status_blueprint.route("/health", methods=["GET"])
def health():
    """GPU 상태 헬스 체크."""
    return jsonify(build_gpu_status_payload())


def register_gpu_status_routes(api_blueprint: Blueprint) -> None:
    """API blueprint 에 GPU 상태 blueprint 를 등록한다."""
    api_blueprint.register_blueprint(gpu_status_blueprint, url_prefix="/gpu_status")


__all__ = [
    "build_gpu_status_payload",
    "query_gpus",
    "register_gpu_status_routes",
]
