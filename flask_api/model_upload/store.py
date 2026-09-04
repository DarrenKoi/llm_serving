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


class DestinationUnverified(UploadError):
    """목적지로 옮겼는데 파일시스템을 통해 되읽히지 않는다."""

    status_code = 500


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


def describe_mount(path: Path) -> dict[str, object]:
    """path 가 실제로 어느 마운트 위에 있는지 알아낸다.

    업로드는 성공했는데 나중에 런처가 "디렉토리 없음"으로 죽는 사고의 원인은
    거의 항상 여기다 - 업로드를 받은 쪽과 모델을 읽는 쪽이 **서로 다른 마운트**
    를 본다. 원인을 가르는 값은 셋이다:

    - `fs_type` 이 `overlay` 계열  -> PVC 가 안 붙었고 컨테이너 쓰기 레이어에
      썼다. pod 이 재생성되면 가중치가 통째로 사라진다. 가장 위험한 상태다.
    - `fs_type` 이 `nfs`/`nfs4`    -> 정상. `mount_point` 가 기대한 경로인지 본다.
    - `mount_point` 가 `/`         -> 볼륨이 아예 마운트되지 않았다.

    `/proc/self/mountinfo` 를 쓰는 이유는 pod 안에서 특권 없이 읽을 수 있는
    유일한 실마운트 증거이기 때문이다. `mount` 명령은 컨테이너에 없을 수 있고
    `/etc/mtab` 은 호스트 것이 섞인다.

    /proc 이 없는 환경(개발 노트북)에서는 조용히 unavailable 을 준다 - 이것은
    진단 정보이지 업로드의 성공 조건이 아니다.
    """
    result: dict[str, object] = {"path": str(path)}
    try:
        resolved = path.resolve()
    except OSError as exc:  # 끊어진 심링크, ESTALE 등
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["resolved"] = str(resolved)

    try:
        raw = Path("/proc/self/mountinfo").read_text()
    except OSError:
        result["available"] = False
        result["reason"] = "/proc/self/mountinfo not readable (non-Linux or restricted)"
        return result

    # mountinfo 한 줄: id parent major:minor root MOUNT_POINT opts... - FSTYPE source superopts
    # 옵션 개수가 가변이라 " - " 구분자를 기준으로 앞뒤를 나눈다.
    best: tuple[int, dict[str, str]] | None = None
    for line in raw.splitlines():
        head, sep, tail = line.partition(" - ")
        if not sep:
            continue
        head_fields = head.split()
        tail_fields = tail.split()
        if len(head_fields) < 5 or len(tail_fields) < 2:
            continue
        # mountinfo 는 공백/탭 등을 8진 이스케이프(\040)로 쓴다.
        mount_point = head_fields[4].replace("\\040", " ")
        if resolved != Path(mount_point) and mount_point not in ("/",) and not str(resolved).startswith(
            mount_point.rstrip("/") + "/"
        ):
            continue
        entry = {
            "mount_point": mount_point,
            "fs_type": tail_fields[0],
            "source": tail_fields[1],
            # propagation: shared/master 가 있으면 호스트 마운트 변화가 전파된다.
            # 없으면(private) pod 시작 후 호스트에 새로 붙은 마운트는 영영 안 보인다.
            "propagation": " ".join(f for f in head_fields[6:] if f.startswith(("shared:", "master:"))) or "private",
        }
        # 가장 긴 mount_point 가 실제로 이 경로를 덮는 마운트다.
        if best is None or len(mount_point) > best[0]:
            best = (len(mount_point), entry)

    result["available"] = True
    if best is None:
        result["reason"] = "no mountinfo entry covers this path"
        return result
    result.update(best[1])
    result["container_layer"] = str(best[1]["fs_type"]).startswith("overlay")
    return result


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

    def verify_destination(self, session: UploadSession) -> dict[str, object]:
        """옮겨놓은 파일을 **파일시스템을 통해 되읽어** 확인한다.

        `finish()` 의 해시 검증은 `.part` 를 읽은 것이고, 이 함수는 `os.replace`
        **이후의 목적지**를 읽는다. 둘은 다른 것을 잡는다:

        - 해시 검증  -> 전송/조립 중 손상
        - 되읽기     -> 마운트가 어긋났거나(ESTALE), 옮긴 결과가 이 프로세스에
                        보이지 않는 상태. 업로드는 200 인데 런처는 파일을 못 찾는
                        그 사고가 여기서 잡힌다.

        크기 불일치나 읽기 실패는 **업로드 실패로 올린다** - 200 을 받고 안심한
        뒤 몇 시간 있다 기동에서 발견하는 것보다 지금 아는 편이 싸다.
        마운트 정보는 판단 재료일 뿐이라 실패 사유로 쓰지 않는다.
        """
        destination = self.dest_root / session.rel_path
        report: dict[str, object] = {
            "path": str(destination),
            "expected_size": session.size,
            "mount": describe_mount(destination.parent),
        }
        try:
            stat_result = destination.stat()
        except OSError as exc:
            raise DestinationUnverified(
                f"destination not readable after commit: {destination} "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        report["actual_size"] = stat_result.st_size
        if stat_result.st_size != session.size:
            raise DestinationUnverified(
                f"destination size mismatch: {destination} "
                f"expected={session.size} actual={stat_result.st_size}"
            )
        if session.size:
            # stat 은 캐시된 속성으로 답할 수 있다. 실제 read 는 서버까지 간다.
            try:
                with open(destination, "rb") as handle:
                    handle.read(1)
            except OSError as exc:
                raise DestinationUnverified(
                    f"destination stat ok but read failed: {destination} "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
        report["readable"] = True
        return report

    def abort(self, upload_id: str) -> None:
        """진행 중인 세션을 버리고 staging 을 정리한다."""
        self._part_path(upload_id).unlink(missing_ok=True)
        self._state_path(upload_id).unlink(missing_ok=True)
