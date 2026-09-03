"""start_all 의 순차 기동 가드 테스트.

호스트 RAM 16GB / swap 없음이라 로딩 최대 구간이 겹치면 OOM killer 가 돈다.
그래서 "준비 안 됐는데 아직 살아 있는" 인스턴스는 다음 모델을 띄우기 전에
반드시 내려야 한다.

    uv run pytest deploy_vlms/scripts/test_start_all.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("start_all", _SCRIPTS / "start_all.py")
start_all = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(start_all)


@pytest.fixture
def calls(monkeypatch):
    """기동/정지 호출을 순서대로 기록한다."""
    log: list[tuple[str, str]] = []
    monkeypatch.setattr(start_all, "print_gpu_plan", lambda _root: None)
    monkeypatch.setattr(
        start_all,
        "stop_if_already_running",
        lambda instance: log.append(("stop", instance)),
    )
    monkeypatch.setattr(
        start_all,
        "start_vllm_model",
        lambda _s, _r, instance: log.append(("start", instance)) or 4242,
    )
    return log


def _stops_between_starts(log, first: str, second: str) -> list[tuple[str, str]]:
    """first 가 뜬 뒤 second 가 뜨기 전까지의 호출 구간을 돌려준다."""
    start_first = log.index(("start", first))
    start_second = log.index(("start", second))
    assert start_first < start_second, "리스트 순서대로 기동해야 한다"
    return log[start_first:start_second]


def test_stuck_instance_is_stopped_before_the_next_model_starts(calls, monkeypatch):
    """타임아웃했지만 살아 있으면 = 아직 로딩 중. 내리고 넘어가야 한다."""
    monkeypatch.setattr(start_all, "wait_until_ready", lambda *_a: False)
    monkeypatch.setattr(start_all, "process_alive", lambda _pid: True)

    with pytest.raises(SystemExit):  # 전부 실패했으므로 exit(1)
        start_all.main()

    first, second = start_all.VLLM_MODELS[0], start_all.VLLM_MODELS[1]
    window = _stops_between_starts(calls, first, second)
    assert ("stop", first) in window, (
        f"{first} 가 로딩 중인 채로 {second} 를 띄우면 RAM 구간이 겹친다"
    )


def test_dead_instance_is_not_stopped_again(calls, monkeypatch):
    """이미 죽었으면 정리할 것이 없다 - 불필요한 stop 을 부르지 않는다."""
    monkeypatch.setattr(start_all, "wait_until_ready", lambda *_a: False)
    monkeypatch.setattr(start_all, "process_alive", lambda _pid: False)

    with pytest.raises(SystemExit):  # 전부 실패했으므로 exit(1)
        start_all.main()

    first, second = start_all.VLLM_MODELS[0], start_all.VLLM_MODELS[1]
    window = _stops_between_starts(calls, first, second)
    # 남는 것은 second 의 기동 전 가드 하나뿐이어야 한다.
    assert window.count(("stop", first)) == 0


def test_healthy_startup_stops_nothing_extra(calls, monkeypatch):
    """정상 기동이면 기동 전 가드 외에 stop 이 더 붙지 않는다."""
    monkeypatch.setattr(start_all, "wait_until_ready", lambda *_a: True)
    monkeypatch.setattr(start_all, "process_alive", lambda _pid: True)

    start_all.main()

    for instance in start_all.VLLM_MODELS:
        assert calls.count(("stop", instance)) == 1
        assert calls.count(("start", instance)) == 1


def test_models_start_one_at_a_time_in_list_order(calls, monkeypatch):
    """가장 큰 모델(27B)이 먼저여야 한다 - 호스트 RAM 때문."""
    monkeypatch.setattr(start_all, "wait_until_ready", lambda *_a: True)
    monkeypatch.setattr(start_all, "process_alive", lambda _pid: True)

    start_all.main()

    started = [name for action, name in calls if action == "start"]
    assert started == start_all.VLLM_MODELS
    assert started[0] == "qwen3.8-27b"
