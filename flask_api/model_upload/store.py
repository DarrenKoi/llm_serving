"""청크 업로드의 파일시스템/상태 계층.

HTTP 를 모른다 - Flask 계층은 이 store 를 감싸기만 한다.
그래서 이어받기/무결성 로직이 실서버 없이 검증된다.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

READ_BLOCK_BYTES = 1024 * 1024

UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")

STATE_SUFFIX = ".json"
PART_SUFFIX = ".part"


class UploadError(Exception):
    """업로드 계층의 기반 예외."""

    status_code = 400


class ChecksumMismatch(UploadError):
    """수신한 바이트의 해시가 선언된 해시와 다르다."""

    status_code = 422


class UploadNotFound(UploadError):
    """그런 업로드 세션이 없다."""

    status_code = 404


class ChunkTooLarge(UploadError):
    """청크가 서버가 허용한 상한을 넘는다."""

    status_code = 413


class LengthRequired(UploadError):
    """바디 길이를 알 수 없다."""

    status_code = 411


class UploadUnauthorized(UploadError):
    """업로드 토큰이 없거나 틀렸다."""

    status_code = 401


class PathNotAllowed(UploadError):
    """목적지 루트를 벗어나는 경로다."""

    status_code = 400


class OffsetMismatch(UploadError):
    """클라이언트가 보낸 offset 이 서버가 기대하는 지점과 다르다."""

    status_code = 409

    def __init__(self, expected_offset: int, received_offset: int):
        super().__init__(
            f"offset mismatch: expected={expected_offset} received={received_offset}"
        )
        self.expected_offset = expected_offset
        self.received_offset = received_offset


@dataclass(frozen=True)
class UploadSession:
    """진행 중인(또는 완료된) 업로드 한 건의 상태."""

    upload_id: str
    rel_path: str
    size: int
    sha256: str
    chunk_size: int
    committed_offset: int
    completed: bool


def _make_upload_id(rel_path: str, sha256: str) -> str:
    """rel_path + 파일 해시로 결정론적 upload_id 를 만든다.

    같은 파일이면 항상 같은 id 라서 클라이언트가 세션 id 를 보관하지 않아도
    이어받기가 된다. 내용이 바뀌면 id 가 달라져 낡은 .part 에 이어붙지 않는다.
    """
    digest = hashlib.sha256(f"{rel_path}\0{sha256}".encode("utf-8")).hexdigest()
    return digest[:32]


def _safe_upload_id(upload_id: str) -> str:
    """URL 세그먼트로 들어온 upload_id 가 우리가 발급한 모양인지 확인한다.

    _make_upload_id 는 항상 32자 hex 를 낸다. 검증 없이 경로에 붙이면
    Flask 기본 컨버터가 허용하는 역슬래시로 Windows 에서 staging 을 벗어난다
    (DELETE 는 unlink 까지 하므로 임의 파일 삭제가 된다). 경로를 만드는 곳이
    _state_path/_part_path 둘뿐이라 여기 한 곳에서 막는다.
    """
    if not UPLOAD_ID_RE.match(upload_id or ""):
        raise PathNotAllowed(f"malformed upload_id: {upload_id!r}")
    return upload_id


def _safe_rel_path(dest_root: Path, rel_path: str) -> str:
    """rel_path 를 정규화하고 dest_root 를 벗어나지 않는지 확인한다."""
    cleaned = (rel_path or "").strip().replace("\\", "/")
    if not cleaned.strip("/"):
        raise PathNotAllowed("rel_path is empty")
    if cleaned.startswith("/"):
        # 절대경로를 조용히 상대경로로 재해석하지 않는다 - 클라이언트 버그를 감춘다.
        raise PathNotAllowed(f"absolute path is not allowed: {rel_path!r}")
    if ":" in cleaned.split("/")[0]:
        raise PathNotAllowed(f"drive-qualified path is not allowed: {rel_path!r}")

    root = Path(dest_root).resolve()
    candidate = (root / cleaned).resolve()
    if candidate != root and root not in candidate.parents:
        raise PathNotAllowed(f"path escapes upload root: {rel_path!r}")
    if candidate == root:
        raise PathNotAllowed("rel_path must name a file, not the root")
    return candidate.relative_to(root).as_posix()


def _hash_file(path: Path) -> str:
    """파일 전체를 스트리밍하며 sha256 을 계산한다."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(READ_BLOCK_BYTES)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


class UploadStore:
    """업로드 세션을 디스크 위에서 관리한다."""

    def __init__(self, dest_root: Path, staging_root: Path):
        self.dest_root = Path(dest_root)
        self.staging_root = Path(staging_root)

    def _state_path(self, upload_id: str) -> Path:
        """세션 상태 JSON 경로를 반환한다."""
        return self.staging_root / f"{_safe_upload_id(upload_id)}{STATE_SUFFIX}"

    def _part_path(self, upload_id: str) -> Path:
        """부분 수신 파일(.part) 경로를 반환한다."""
        return self.staging_root / f"{_safe_upload_id(upload_id)}{PART_SUFFIX}"

    def _write_state(self, session: UploadSession) -> None:
        """세션 상태를 디스크에 기록한다."""
        self.staging_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "upload_id": session.upload_id,
            "rel_path": session.rel_path,
            "size": session.size,
            "sha256": session.sha256,
            "chunk_size": session.chunk_size,
            "committed_offset": session.committed_offset,
            "completed": session.completed,
        }
        self._state_path(session.upload_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def begin(self, rel_path: str, size: int, sha256: str, chunk_size: int) -> UploadSession:
        """업로드 세션을 새로 만들거나 기존 세션을 이어받는다."""
        rel_path = _safe_rel_path(self.dest_root, rel_path)
        upload_id = _make_upload_id(rel_path, sha256)
        if self._state_path(upload_id).exists():
            return self._read_state(upload_id)

        session = UploadSession(
            upload_id=upload_id,
            rel_path=rel_path,
            size=size,
            sha256=sha256,
            chunk_size=chunk_size,
            committed_offset=0,
            completed=False,
        )
        self._write_state(session)
        return session

    def _read_state(self, upload_id: str) -> UploadSession:
        """디스크에서 세션 상태를 읽고 .part 실제 크기와 대조한다.

        상태 파일이 "10바이트 받았다"는데 .part 가 사라졌거나 짧으면,
        그 지점부터 이어붙일 때 앞쪽에 0 으로 채워진 구멍이 생긴다.
        기록보다 실제가 항상 우선이다.
        """
        state_path = self._state_path(upload_id)
        if not state_path.exists():
            raise UploadNotFound(f"unknown upload_id: {upload_id}")
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        session = UploadSession(**payload)
        if session.completed:
            # 완료 기록이 있어도 목적지 파일이 실제로 그 크기로 있어야 완료다.
            # (누가 models 디렉터리를 정리하면 기록만 남고 파일은 없다)
            destination = self.dest_root / session.rel_path
            if destination.exists() and destination.stat().st_size == session.size:
                return session
            return replace(session, completed=False, committed_offset=0)

        part_path = self._part_path(upload_id)
        part_size = part_path.stat().st_size if part_path.exists() else 0
        if part_size >= session.committed_offset:
            return session

        reconciled = replace(session, committed_offset=part_size)
        self._write_state(reconciled)
        return reconciled

    def status(self, upload_id: str) -> UploadSession:
        """현재 세션 상태를 반환한다."""
        return self._read_state(upload_id)

    def append_chunk(
        self,
        upload_id: str,
        offset: int,
        stream: BinaryIO,
        length: int,
        chunk_sha256: str,
    ) -> UploadSession:
        """청크 하나를 .part 끝에 흘려 쓰고 committed_offset 을 전진시킨다."""
        session = self._read_state(upload_id)
        if offset != session.committed_offset:
            raise OffsetMismatch(
                expected_offset=session.committed_offset, received_offset=offset
            )

        part_path = self._part_path(upload_id)
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.touch(exist_ok=True)

        written = 0
        hasher = hashlib.sha256()
        with open(part_path, "r+b") as handle:
            # 지난번 중단으로 남은 미커밋 꼬리를 먼저 잘라낸다.
            handle.truncate(session.committed_offset)
            handle.seek(session.committed_offset)
            remaining = length
            while remaining > 0:
                block = stream.read(min(READ_BLOCK_BYTES, remaining))
                if not block:
                    break
                handle.write(block)
                hasher.update(block)
                written += len(block)
                remaining -= len(block)

            digest = hasher.hexdigest()
            if written != length or digest != chunk_sha256:
                # 짧게 도착했거나 손상됐다. 커밋 지점까지 되돌리고 offset 은 그대로 둔다.
                handle.truncate(session.committed_offset)
                handle.flush()
                os.fsync(handle.fileno())
                raise ChecksumMismatch(
                    f"chunk rejected at offset={session.committed_offset}: "
                    f"expected {length} bytes sha256={chunk_sha256}, "
                    f"got {written} bytes sha256={digest}"
                )

            handle.flush()
            os.fsync(handle.fileno())

        updated = replace(session, committed_offset=session.committed_offset + written)
        self._write_state(updated)
        return updated

    def finish(self, upload_id: str) -> UploadSession:
        """.part 전체를 재해싱해 검증한 뒤 목적지로 원자적으로 옮긴다.

        청크별 해시와 별개로 여기서 한 번 더 전체를 읽는다 - 네트워크가 아니라
        디스크 쓰기/조립 단계에서 생긴 손상은 이 단계에서만 잡힌다.
        """
        session = self._read_state(upload_id)
        if session.completed:
            return session

        part_path = self._part_path(upload_id)
        # 0바이트 파일은 청크가 한 번도 안 와서 .part 가 없다 (HF 리포에 흔하다).
        # 없는 채로 두면 여기서 FileNotFoundError 가 난다.
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.touch(exist_ok=True)
        actual = _hash_file(part_path)
        if actual != session.sha256:
            # 청크는 다 통과했는데 전체가 어긋났다 = 디스크/조립 단계 손상.
            # 이어받을 지점을 신뢰할 수 없으므로 통째로 버리고 처음부터 받는다.
            part_path.unlink(missing_ok=True)
            self._write_state(replace(session, committed_offset=0, completed=False))
            raise ChecksumMismatch(
                f"file sha256 mismatch: expected={session.sha256} actual={actual}"
            )

        destination = self.dest_root / session.rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part_path, destination)

        completed = replace(session, completed=True)
        self._write_state(completed)
        return completed

    def abort(self, upload_id: str) -> None:
        """진행 중인 세션을 버리고 staging 을 정리한다."""
        self._part_path(upload_id).unlink(missing_ok=True)
        self._state_path(upload_id).unlink(missing_ok=True)
