"""stop_model.find_pids_by_environ - EngineCore 자식까지 환경변수로 찾는다."""

from pathlib import Path

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
