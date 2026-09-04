"""start_model.rotate_log - 기동마다 이전 로그를 옮기고 오래된 것을 지운다."""

from pathlib import Path

from start_model import rotate_log


def test_rotate_log_moves_current_and_prunes_old(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    for i in range(3):
        (tmp_path / f"x.log.20260101-00000{i}").write_text("old")
    log.write_text("current run")

    rotate_log(log, keep=2)

    assert not log.exists()
    rotated = sorted(p.name for p in tmp_path.glob("x.log.*"))
    assert len(rotated) == 2
    assert (tmp_path / rotated[-1]).read_text() == "current run"
    assert "20260101-000000" not in rotated[0]


def test_rotate_log_skips_missing_or_empty(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    rotate_log(log)
    log.write_text("")
    rotate_log(log)
    assert log.exists() and not list(tmp_path.glob("x.log.*"))
