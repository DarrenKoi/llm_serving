"""gpu_status 파싱/실패 처리 테스트.

개발 PC 에는 nvidia-smi 가 없으므로 subprocess 를 갈아끼워 돌린다.

    pytest flask_api/test_gpu_status.py
"""

import subprocess

import pytest

from flask_api import gpu_status

TWO_H200 = (
    "0, NVIDIA H200, 143771, 120448, 87, 61, 412.55, 700.00\n"
    "1, NVIDIA H200, 143771, 51200, 12, 44, 105.20, 700.00\n"
)


class _Done:
    def __init__(self, stdout: str):
        self.stdout = stdout


def _fake_run(stdout: str):
    def run(*_args, **_kwargs):
        return _Done(stdout)
    return run


def test_parses_two_gpus(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(TWO_H200))

    result = gpu_status.query_gpus()

    assert result["available"] is True
    assert [g["index"] for g in result["gpus"]] == ["0", "1"]
    first = result["gpus"][0]
    assert first["name"] == "NVIDIA H200"
    assert first["memory_used_mib"] == 120448
    assert first["utilization_percent"] == 87
    assert first["temperature_c"] == 61
    assert first["power_draw_w"] == 412.55


def test_computes_memory_percent(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(TWO_H200))

    gpus = gpu_status.query_gpus()["gpus"]

    assert gpus[0]["memory_used_percent"] == pytest.approx(83.8, abs=0.1)
    assert gpus[1]["memory_used_percent"] == pytest.approx(35.6, abs=0.1)


def test_na_values_become_none_not_zero(monkeypatch):
    """nvidia-smi 는 못 읽는 값을 [N/A] 로 준다. 0 으로 읽으면 그래프가 거짓말을 한다."""
    monkeypatch.setattr(
        subprocess, "run", _fake_run("0, NVIDIA H200, 143771, 120448, 87, [N/A], [N/A], 700.00\n")
    )

    gpu = gpu_status.query_gpus()["gpus"][0]

    assert gpu["temperature_c"] is None
    assert gpu["power_draw_w"] is None
    assert gpu["utilization_percent"] == 87


def test_missing_nvidia_smi_is_a_state_not_an_error(monkeypatch):
    """개발 PC 에서 대시보드가 500 으로 죽으면 안 된다."""
    def boom(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", boom)

    result = gpu_status.query_gpus()

    assert result["available"] is False
    assert result["gpus"] == []
    assert "not found" in result["reason"]


def test_timeout_is_reported(monkeypatch):
    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)
    monkeypatch.setattr(subprocess, "run", boom)

    result = gpu_status.query_gpus()

    assert result["available"] is False
    assert "timed out" in result["reason"]


def test_nonzero_exit_is_reported(monkeypatch):
    def boom(*_a, **_k):
        raise subprocess.CalledProcessError(returncode=9, cmd="nvidia-smi", stderr="driver error")
    monkeypatch.setattr(subprocess, "run", boom)

    result = gpu_status.query_gpus()

    assert result["available"] is False
    assert "driver error" in result["reason"]


def test_short_row_is_skipped(monkeypatch):
    """필드가 모자란 줄로 IndexError 를 내지 않는다."""
    monkeypatch.setattr(subprocess, "run", _fake_run("0, NVIDIA H200, 143771\n"))

    result = gpu_status.query_gpus()

    assert result["available"] is False
    assert result["gpus"] == []
