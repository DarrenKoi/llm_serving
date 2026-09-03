"""UploadStore 단위 테스트.

실서버/네트워크 없이 Mac 에서 실행된다:
    uv run pytest flask_api/model_upload/test_store.py
"""

import hashlib
import io

import pytest

from flask_api.model_upload.store import (
    ChecksumMismatch,
    OffsetMismatch,
    PathNotAllowed,
    UploadStore,
)


@pytest.fixture()
def store(tmp_path):
    """임시 dest/staging 루트를 가진 store 를 만든다."""
    return UploadStore(
        dest_root=tmp_path / "models",
        staging_root=tmp_path / "staging",
    )


def test_begin_creates_new_session_at_offset_zero(store):
    """새 파일 업로드를 시작하면 committed_offset 0 인 세션이 생긴다."""
    session = store.begin(
        rel_path="MAI-UI-8B/config.json",
        size=12,
        sha256="ab" * 32,
        chunk_size=1024,
    )

    assert session.upload_id
    assert session.committed_offset == 0
    assert session.completed is False


def _sha(data: bytes) -> str:
    """테스트용 sha256 헬퍼."""
    return hashlib.sha256(data).hexdigest()


def test_append_chunk_advances_committed_offset(store):
    """청크를 보내면 committed_offset 이 그만큼 전진한다."""
    data = b"hello world!"
    session = store.begin(
        rel_path="MAI-UI-8B/config.json",
        size=len(data),
        sha256=_sha(data),
        chunk_size=1024,
    )

    updated = store.append_chunk(
        upload_id=session.upload_id,
        offset=0,
        stream=io.BytesIO(data),
        length=len(data),
        chunk_sha256=_sha(data),
    )

    assert updated.committed_offset == len(data)


def _upload_all(store, rel_path: str, data: bytes, chunk_size: int):
    """파일 하나를 청크로 끝까지 올린다(테스트 헬퍼)."""
    session = store.begin(
        rel_path=rel_path, size=len(data), sha256=_sha(data), chunk_size=chunk_size
    )
    for start in range(0, len(data), chunk_size):
        piece = data[start : start + chunk_size]
        session = store.append_chunk(
            upload_id=session.upload_id,
            offset=start,
            stream=io.BytesIO(piece),
            length=len(piece),
            chunk_sha256=_sha(piece),
        )
    return store.finish(session.upload_id)


def test_finish_places_verified_file_at_destination(store, tmp_path):
    """모든 청크가 도착하면 목적지에 원본과 바이트 동일한 파일이 생긴다."""
    data = b"0123456789abcdefghij"

    session = _upload_all(store, "MAI-UI-8B/model.safetensors", data, chunk_size=7)

    placed = tmp_path / "models" / "MAI-UI-8B" / "model.safetensors"
    assert session.completed is True
    assert placed.read_bytes() == data


def test_begin_resumes_existing_session_at_committed_offset(store):
    """같은 파일로 다시 begin 하면 이미 받은 지점부터 이어받는다."""
    data = b"0123456789abcdefghij"
    first = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=7,
    )
    store.append_chunk(
        upload_id=first.upload_id,
        offset=0,
        stream=io.BytesIO(data[:7]),
        length=7,
        chunk_sha256=_sha(data[:7]),
    )

    resumed = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=7,
    )

    assert resumed.upload_id == first.upload_id
    assert resumed.committed_offset == 7


class _ShortStream:
    """선언된 length 보다 적게 주고 조용히 EOF 를 내는 스트림.

    소켓이 예외 없이 일찍 닫히는 경우를 흉내낸다 - 서버가 "다 받았다"고
    착각하면 파일에 구멍이 뚫린 채 offset 만 전진한다.
    """

    def __init__(self, data: bytes, deliver: int):
        self._data = data[:deliver]
        self._pos = 0

    def read(self, size: int) -> bytes:
        """일부만 흘린 뒤 EOF."""
        block = self._data[self._pos : self._pos + size]
        self._pos += len(block)
        return block


def test_short_chunk_body_is_rejected_and_offset_stays(store):
    """바디가 선언 길이보다 짧게 도착하면 거부되고 offset 이 전진하지 않는다."""
    data = b"0123456789abcdefghij"
    session = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )

    with pytest.raises(ChecksumMismatch):
        store.append_chunk(
            upload_id=session.upload_id,
            offset=0,
            stream=_ShortStream(data[:10], deliver=3),
            length=10,
            chunk_sha256=_sha(data[:10]),
        )

    assert store.status(session.upload_id).committed_offset == 0


def test_corrupted_chunk_is_rejected_and_offset_stays(store):
    """길이는 맞지만 내용이 손상된 청크는 거부되고 offset 이 전진하지 않는다."""
    data = b"0123456789abcdefghij"
    session = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )

    with pytest.raises(ChecksumMismatch):
        store.append_chunk(
            upload_id=session.upload_id,
            offset=0,
            stream=io.BytesIO(b"XXXXXXXXXX"),
            length=10,
            chunk_sha256=_sha(data[:10]),
        )

    assert store.status(session.upload_id).committed_offset == 0


def test_resend_after_rejected_chunk_still_produces_exact_file(store, tmp_path):
    """거부된 청크를 재전송하면 최종 파일은 원본과 바이트 동일하다."""
    data = b"0123456789abcdefghij"
    session = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )
    with pytest.raises(ChecksumMismatch):
        store.append_chunk(
            upload_id=session.upload_id,
            offset=0,
            stream=_ShortStream(data[:10], deliver=3),
            length=10,
            chunk_sha256=_sha(data[:10]),
        )

    for start in (0, 10):
        piece = data[start : start + 10]
        store.append_chunk(
            upload_id=session.upload_id,
            offset=start,
            stream=io.BytesIO(piece),
            length=10,
            chunk_sha256=_sha(piece),
        )
    store.finish(session.upload_id)

    placed = tmp_path / "models" / "MAI-UI-8B" / "model.safetensors"
    assert placed.read_bytes() == data


def test_wrong_offset_is_rejected_with_expected_offset(store):
    """클라이언트가 엉뚱한 offset 을 보내면 거부하고 기대 offset 을 알려준다."""
    data = b"0123456789abcdefghij"
    session = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )

    with pytest.raises(OffsetMismatch) as excinfo:
        store.append_chunk(
            upload_id=session.upload_id,
            offset=10,
            stream=io.BytesIO(data[10:]),
            length=10,
            chunk_sha256=_sha(data[10:]),
        )

    assert excinfo.value.expected_offset == 0
    assert store.status(session.upload_id).committed_offset == 0


def _stage_one_chunk(store, data: bytes):
    """첫 10바이트만 올린 세션을 만든다(테스트 헬퍼)."""
    session = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )
    store.append_chunk(
        upload_id=session.upload_id,
        offset=0,
        stream=io.BytesIO(data[:10]),
        length=10,
        chunk_sha256=_sha(data[:10]),
    )
    return session


def test_offset_is_reconciled_when_part_file_disappears(store):
    """.part 가 사라지면 offset 을 0 으로 되돌린다(구멍 뚫린 파일 방지)."""
    data = b"0123456789abcdefghij"
    session = _stage_one_chunk(store, data)

    for part in store.staging_root.glob("*.part"):
        part.unlink()

    resumed = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )
    assert resumed.committed_offset == 0
    assert store.status(session.upload_id).committed_offset == 0


def test_offset_is_reconciled_when_part_file_is_shorter(store, tmp_path):
    """.part 가 기록보다 짧으면 실제 크기까지만 인정한다."""
    data = b"0123456789abcdefghij"
    _stage_one_chunk(store, data)

    part = next(store.staging_root.glob("*.part"))
    with open(part, "r+b") as handle:
        handle.truncate(4)

    resumed = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )
    assert resumed.committed_offset == 4


@pytest.mark.parametrize(
    "rel_path",
    [
        "../escape.bin",
        "MAI-UI-8B/../../escape.bin",
        "/etc/passwd",
        "",
        "   ",
    ],
)
def test_paths_outside_dest_root_are_rejected(store, rel_path):
    """목적지 루트를 벗어나는 경로는 세션 생성 자체를 거부한다."""
    with pytest.raises(PathNotAllowed):
        store.begin(rel_path=rel_path, size=4, sha256="ab" * 32, chunk_size=10)


def test_nested_relative_path_inside_root_is_allowed(store):
    """루트 안쪽의 깊은 경로는 허용된다."""
    session = store.begin(
        rel_path="MAI-UI-8B/subdir/model-00001-of-00004.safetensors",
        size=4,
        sha256="ab" * 32,
        chunk_size=10,
    )
    assert session.rel_path == "MAI-UI-8B/subdir/model-00001-of-00004.safetensors"


def test_begin_reports_completed_for_already_uploaded_file(store):
    """이미 올라간 파일을 다시 begin 하면 completed 로 답해 재전송을 막는다."""
    data = b"0123456789abcdefghij"
    _upload_all(store, "MAI-UI-8B/model.safetensors", data, chunk_size=10)

    again = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )

    assert again.completed is True
    assert again.committed_offset == len(data)


def test_completed_state_is_invalidated_when_destination_file_is_gone(store, tmp_path):
    """목적지 파일이 지워졌으면 완료 기록을 믿지 않고 처음부터 다시 받는다."""
    data = b"0123456789abcdefghij"
    _upload_all(store, "MAI-UI-8B/model.safetensors", data, chunk_size=10)
    (tmp_path / "models" / "MAI-UI-8B" / "model.safetensors").unlink()

    again = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )

    assert again.completed is False
    assert again.committed_offset == 0


def test_completed_state_is_invalidated_when_destination_size_differs(store, tmp_path):
    """목적지 파일 크기가 다르면 완료로 보지 않는다."""
    data = b"0123456789abcdefghij"
    _upload_all(store, "MAI-UI-8B/model.safetensors", data, chunk_size=10)
    (tmp_path / "models" / "MAI-UI-8B" / "model.safetensors").write_bytes(b"short")

    again = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )

    assert again.completed is False
    assert again.committed_offset == 0


def test_finish_rejects_and_resets_when_whole_file_hash_differs(store, tmp_path):
    """조립된 파일 전체 해시가 다르면 목적지에 두지 않고 처음부터 다시 받는다."""
    data = b"0123456789abcdefghij"
    session = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )
    for start in (0, 10):
        piece = data[start : start + 10]
        store.append_chunk(
            upload_id=session.upload_id,
            offset=start,
            stream=io.BytesIO(piece),
            length=10,
            chunk_sha256=_sha(piece),
        )

    # 디스크 손상 흉내: 길이는 같고 내용만 뒤집힌다(청크 해시는 이미 통과한 뒤).
    part = next(store.staging_root.glob("*.part"))
    with open(part, "r+b") as handle:
        handle.seek(5)
        handle.write(b"X")

    with pytest.raises(ChecksumMismatch):
        store.finish(session.upload_id)

    assert not (tmp_path / "models" / "MAI-UI-8B" / "model.safetensors").exists()
    restarted = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )
    assert restarted.committed_offset == 0


def test_abort_clears_staging_and_frees_disk(store):
    """중단하면 .part 와 상태 파일이 사라지고 다음 begin 은 0 부터 시작한다."""
    data = b"0123456789abcdefghij"
    session = _stage_one_chunk(store, data)

    store.abort(session.upload_id)

    assert list(store.staging_root.glob("*.part")) == []
    restarted = store.begin(
        rel_path="MAI-UI-8B/model.safetensors",
        size=len(data),
        sha256=_sha(data),
        chunk_size=10,
    )
    assert restarted.committed_offset == 0


def test_zero_byte_file_completes(store, tmp_path):
    """0바이트 파일도 정상 완료된다(HF 리포에 흔히 섞여 있다)."""
    session = store.begin(
        rel_path="MAI-UI-8B/.gitkeep-like",
        size=0,
        sha256=_sha(b""),
        chunk_size=10,
    )

    finished = store.finish(session.upload_id)

    assert finished.completed is True
    assert (tmp_path / "models" / "MAI-UI-8B" / ".gitkeep-like").read_bytes() == b""
