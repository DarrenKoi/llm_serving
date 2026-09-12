"""stop_model.find_pids_by_environ - EngineCore 자식까지 환경변수로 찾는다."""

from pathlib import Path

import stop_model
from stop_model import find_pids_by_environ


def _proc(root: Path, pid: int, environ: dict[str, str]) -> None:
    d = root / str(pid)
    d.mkdir()
    (d / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in environ.items()) + b"\0")


def test_finds_api_server_and_engine_core_but_not_other_instances(tmp_path: Path) -> None:
    base = {"VLLM_NO_USAGE_STATS": "1", "CUDA_VISIBLE_DEVICES": "1"}
    _proc(tmp_path, 100, {**base, "PORT": "8006"})  # api_server
    _proc(tmp_path, 101, {**base, "PORT": "8006"})  # VLLM::EngineCore (cmdline 에 marker 없음)
    _proc(tmp_path, 200, {**base, "PORT": "8002"})  # 다른 인스턴스
    _proc(tmp_path, 300, {"PORT": "8006"})  # vLLM 이 아닌 무관한 프로세스
    (tmp_path / "self").mkdir()

    assert find_pids_by_environ("8006", tmp_path) == [100, 101]
    assert find_pids_by_environ("", tmp_path) == []


def test_read_env_value_preserves_missing_and_quoted_values(tmp_path: Path) -> None:
    path = tmp_path / "model.env"
    assert stop_model.read_env_value(path, "PORT") == ""
    path.write_text("# comment\n\ninvalid\nOTHER=skip\n PORT = '8006' \nNAME=\"qwen\"\n")
    assert stop_model.read_env_value(path, "PORT") == "8006"
    assert stop_model.read_env_value(path, "NAME") == "qwen"
    assert stop_model.read_env_value(path, "MISSING") == ""


def test_collect_targets_preserves_sources_and_listing(monkeypatch, capsys) -> None:
    """PID 파일·cmdline·포트·자식 출처를 합치고 중복 PID 는 한 번만 표시한다."""
    monkeypatch.setattr(stop_model, "resolve_served_model_name", lambda _: "qwen")
    monkeypatch.setattr(stop_model, "read_live_pid_from_file", lambda _: 100)
    monkeypatch.setattr(stop_model, "find_listening_pids_by_port", lambda _: [100, 102])
    monkeypatch.setattr(stop_model, "find_pids_by_environ", lambda _: [103])
    processes = [
        stop_model.build_process_record(101, "8006", "qwen"),
        stop_model.build_process_record(200, "8002", "mai"),
    ]
    targets = stop_model.collect_targets(processes, "qwen", "8006")
    assert [proc["pid"] for proc in targets] == [100, 101, 102, 103]
    stop_model.list_running(targets)
    output = capsys.readouterr().out
    for pid in (100, 101, 102, 103):
        assert output.count(f"PID={pid} PORT=8006 MODEL=qwen") == 1
    assert "PID=200" not in output
