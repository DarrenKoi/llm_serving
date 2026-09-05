"""로컬 PC -> 사내 Flask 서버로 모델 가중치를 청크 업로드한다.

브라우저 드래그앤드롭이 1GB 근처에서 깨지는 것을 대체한다:
  1) 스트리밍  - 파일을 통째로 메모리에 올리지 않고 청크 단위로 흘린다.
  2) 이어받기  - 끊기면 같은 명령을 다시 실행하면 받은 지점부터 이어간다.
  3) 무결성    - 청크마다 sha256, 완료 시 파일 전체 sha256 을 서버가 재검증한다.

의존성은 requests 뿐이다(서버 코드/Flask 를 import 하지 않는다).
"""

# ---------------------------------------------------------------------------
# 실행 인자 (여기를 고쳐 쓴다 - 셸 env 는 1회성 override 일 뿐이다)
#
# 우선순위: 실제 셸 env > 아래 상수 > 코드 기본값.
# env 에 밀려 무시된 상수는 실행 시 콘솔에 찍힌다.
# None / "" 은 "미설정"이고, 0 은 유효한 값이다.
# ---------------------------------------------------------------------------

BASE_URL = "http://vlm-host.internal"
SRC = ""            # 올릴 로컬 폴더 또는 파일. 예) r"C:/models/Qwen3.8-27B"
DEST = ""           # 서버 루트 아래 목적지. 비우면 소스 폴더명 = <model_name>
TOKEN = ""          # 서버가 토큰을 안 쓰면 빈 문자열
CHUNK_MB = None     # None = 코드 기본값(32). 프록시 상한이 작으면 낮춘다
MAX_RETRIES = None  # None = 코드 기본값(5)


import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

READ_BLOCK_BYTES = 1024 * 1024
MIN_CHUNK_BYTES = 256 * 1024
RETRY_BACKOFF_BASE_SEC = 1.0
RETRY_BACKOFF_CAP_SEC = 30.0


class UploadFailed(Exception):
    """재시도해도 소용없는 실패(인증/권한/경로 오류 등)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


class TransientError(Exception):
    """재시도하면 될 수 있는 실패(네트워크/5xx)."""


class RemoteOffsetMismatch(Exception):
    """서버가 기대하는 offset 이 다르다 - 그 지점으로 다시 맞춘다."""

    def __init__(self, expected_offset: int):
        super().__init__(f"server expects offset {expected_offset}")
        self.expected_offset = expected_offset


class RemoteChecksumMismatch(Exception):
    """서버가 청크 해시 불일치로 거부했다."""


class UploadTransport(Protocol):
    """업로드 서버와 말하는 최소 인터페이스."""

    def open_session(
        self, rel_path: str, size: int, sha256: str, chunk_size: int
    ) -> dict: ...

    def put_chunk(
        self, upload_id: str, offset: int, data: bytes, chunk_sha256: str
    ) -> dict: ...

    def complete(self, upload_id: str) -> dict: ...

    def get_session(self, upload_id: str) -> dict: ...


@dataclass(frozen=True)
class FileJob:
    """올릴 파일 한 건."""

    local_path: Path
    rel_path: str
    size: int


@dataclass(frozen=True)
class UploadResult:
    """파일 한 건의 업로드 결과."""

    rel_path: str
    sent_bytes: int
    skipped: bool
    chunk_size: int = 0


def sha256_of_file(path: Path) -> str:
    """파일 전체를 스트리밍하며 sha256 을 계산한다."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(READ_BLOCK_BYTES)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def plan_jobs(source: Path, dest_prefix: str = "") -> list[FileJob]:
    """올릴 파일 목록을 만든다(폴더면 재귀, 숨김 항목은 건너뛴다)."""
    source = Path(source)
    prefix = dest_prefix.strip("/")

    if source.is_file():
        rel = f"{prefix}/{source.name}" if prefix else source.name
        return [FileJob(source, rel, source.stat().st_size)]

    jobs: list[FileJob] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts):
            continue
        rel = f"{prefix}/{relative.as_posix()}" if prefix else relative.as_posix()
        jobs.append(FileJob(path, rel, path.stat().st_size))
    return jobs


def upload_file(
    transport: UploadTransport,
    job: FileJob,
    chunk_size: int,
    max_retries: int = 5,
    min_chunk_size: int = MIN_CHUNK_BYTES,
    sleep_fn: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] | None = None,
) -> UploadResult:
    """파일 하나를 청크로 올린다."""
    digest = sha256_of_file(job.local_path)
    session = transport.open_session(job.rel_path, job.size, digest, chunk_size)
    if session.get("completed"):
        return UploadResult(job.rel_path, 0, skipped=True, chunk_size=chunk_size)

    upload_id = session["upload_id"]
    offset = int(session["committed_offset"])
    sent = 0

    with open(job.local_path, "rb") as handle:
        attempts = 0
        while offset < job.size:
            handle.seek(offset)
            block = handle.read(chunk_size)
            try:
                response = transport.put_chunk(
                    upload_id, offset, block, hashlib.sha256(block).hexdigest()
                )
            except RemoteOffsetMismatch as error:
                # 서버가 다른 지점을 기대한다. 대개 청크는 반영됐는데 응답만 유실된 경우다.
                log(f"[WARNING] offset 재동기화 {offset} -> {error.expected_offset}")
                offset = error.expected_offset
                attempts += 1
                if attempts > max_retries:
                    raise
                continue
            except UploadFailed as error:
                if error.status_code != 413 or chunk_size <= min_chunk_size:
                    raise
                # 앞단 프록시(nginx client_max_body_size)가 Flask 에 닿기 전에 잘랐다.
                # 서버 /health 는 프록시 상한을 모르므로 여기서 직접 줄여 본다.
                chunk_size = max(min_chunk_size, chunk_size // 2)
                log(
                    f"[WARNING] 프록시가 청크를 413 으로 거부했습니다. "
                    f"청크를 {_human_bytes(chunk_size)} 로 줄여 재시도합니다 "
                    "(nginx client_max_body_size 를 올리는 것이 근본 해결입니다)"
                )
                continue
            except (TransientError, RemoteChecksumMismatch) as error:
                attempts += 1
                if attempts > max_retries:
                    raise
                backoff = min(RETRY_BACKOFF_CAP_SEC, RETRY_BACKOFF_BASE_SEC * 2 ** (attempts - 1))
                log(
                    f"[WARNING] {job.rel_path} offset={offset} 재시도 "
                    f"{attempts}/{max_retries} ({type(error).__name__}: {error})"
                )
                sleep_fn(backoff)
                # 서버가 실제로 어디까지 받았는지 다시 물어 어긋남을 없앤다.
                offset = int(
                    transport.open_session(
                        job.rel_path, job.size, digest, chunk_size
                    )["committed_offset"]
                )
                continue

            attempts = 0
            offset = int(response["committed_offset"])
            sent += len(block)
            if progress_fn is not None:
                progress_fn(offset, job.size)

    _complete_with_proxy_tolerance(
        transport, upload_id, job, max_retries=max_retries, sleep_fn=sleep_fn, log=log
    )
    return UploadResult(job.rel_path, sent, skipped=False, chunk_size=chunk_size)


# ── HTTP transport ────────────────────────────────────────────────────

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class HttpTransport:
    """사내 Flask 업로드 엔드포인트와 말하는 transport."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        connect_timeout_sec: float = 10.0,
        read_timeout_sec: float = 600.0,
    ):
        import requests  # 로컬 PC 에는 requests 만 있으면 된다.

        self._requests = requests
        self._session = requests.Session()
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = (connect_timeout_sec, read_timeout_sec)

    def _headers(self, extra: dict | None = None) -> dict:
        """공통 헤더에 토큰을 얹는다."""
        headers = {}
        if self.token:
            headers["X-Upload-Token"] = self.token
        headers.update(extra or {})
        return headers

    def _request(self, method: str, path: str, **kwargs):
        """요청을 보내고 네트워크 오류를 TransientError 로 바꾼다."""
        url = f"{self.base_url}/api/model_upload{path}"
        try:
            response = self._session.request(
                method, url, timeout=self.timeout, **kwargs
            )
        except (
            self._requests.exceptions.ConnectionError,
            self._requests.exceptions.Timeout,
            self._requests.exceptions.ChunkedEncodingError,
        ) as error:
            raise TransientError(f"{method} {url}: {error}") from error
        return self._interpret(response)

    def _interpret(self, response) -> dict:
        """상태 코드를 클라이언트 예외로 매핑한다."""
        if response.status_code == 200:
            return response.json()

        try:
            payload = response.json()
        except ValueError:
            payload = {"error": response.text[:500]}
        message = str(payload.get("error", response.reason))

        if response.status_code == 409:
            return_offset = payload.get("expected_offset")
            if return_offset is not None:
                raise RemoteOffsetMismatch(int(return_offset))
        if response.status_code == 422:
            raise RemoteChecksumMismatch(message)
        if response.status_code in RETRYABLE_STATUS:
            raise TransientError(f"HTTP {response.status_code}: {message}")
        raise UploadFailed(response.status_code, message)

    def open_session(
        self, rel_path: str, size: int, sha256: str, chunk_size: int
    ) -> dict:
        """세션을 만들거나 이어받는다."""
        return self._request(
            "POST",
            "/sessions",
            json={
                "rel_path": rel_path,
                "size": size,
                "sha256": sha256,
                "chunk_size": chunk_size,
            },
            headers=self._headers(),
        )

    def put_chunk(
        self, upload_id: str, offset: int, data: bytes, chunk_sha256: str
    ) -> dict:
        """청크 하나를 보낸다."""
        return self._request(
            "PUT",
            f"/sessions/{upload_id}/chunk",
            data=data,
            headers=self._headers(
                {
                    "X-Upload-Offset": str(offset),
                    "X-Chunk-Sha256": chunk_sha256,
                    "Content-Type": "application/octet-stream",
                }
            ),
        )

    def complete(self, upload_id: str) -> dict:
        """전체 해시 검증과 배치를 요청한다."""
        return self._request(
            "POST", f"/sessions/{upload_id}/complete", headers=self._headers()
        )

    def get_session(self, upload_id: str) -> dict:
        """서버가 보는 세션 상태를 조회한다."""
        return self._request(
            "GET", f"/sessions/{upload_id}", headers=self._headers()
        )

    def health(self) -> dict:
        """엔드포인트 도달성을 확인한다."""
        return self._request("GET", "/health")


# ── 실행 진입점 ────────────────────────────────────────────────────────

DEFAULT_CHUNK_MB = 32
# 완료 재해싱이 프록시 타임아웃보다 오래 걸릴 때 폴링 창이 되기도 한다.
# 5회면 백오프 합계가 ~31s 뿐이라 수십 GB 샤드에서 모자란다(백오프 상한 30s).
DEFAULT_MAX_RETRIES = 12


def _human_bytes(value: float) -> str:
    """사람이 읽는 크기 문자열."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def _env(name: str, constant, default: object = "") -> str:
    """셸 env > 파일 상수 > 코드 기본값. 무시된 상수는 콘솔에 남긴다."""
    import os

    raw = os.environ.get(name, "").strip()
    if raw:
        if constant not in (None, ""):
            print(f"[INFO] {name}={raw} (셸 env) - 파일 상수 {constant!r} 는 무시됩니다")
        return raw
    if constant in (None, ""):
        return str(default)
    return str(constant)


def _load_settings() -> dict:
    """파일 상단 상수 + 환경변수에서 실행 설정을 읽는다(CLI 인자는 쓰지 않는다)."""
    source = _env("MODEL_UPLOAD_SRC", SRC)
    base_url = _env("MODEL_UPLOAD_URL", BASE_URL)
    if not source or not base_url:
        raise SystemExit(
            "[ERROR] SRC 와 BASE_URL 이 필요합니다.\n"
            f"  {__file__} 상단의 SRC / BASE_URL 상수를 채우세요."
        )

    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise SystemExit(f"[ERROR] 소스를 찾을 수 없습니다: {source_path}")

    dest_prefix = _env("MODEL_UPLOAD_DEST", DEST)
    if not dest_prefix:
        dest_prefix = source_path.name if source_path.is_dir() else ""

    return {
        "source": source_path,
        "base_url": base_url,
        "dest_prefix": dest_prefix,
        "token": _env("MODEL_UPLOAD_TOKEN", TOKEN),
        "chunk_size": int(_env("MODEL_UPLOAD_CHUNK_MB", CHUNK_MB, DEFAULT_CHUNK_MB))
        * 1024
        * 1024,
        "max_retries": int(
            _env("MODEL_UPLOAD_MAX_RETRIES", MAX_RETRIES, DEFAULT_MAX_RETRIES)
        ),
    }


def _complete_with_proxy_tolerance(
    transport: UploadTransport,
    upload_id: str,
    job: FileJob,
    max_retries: int,
    sleep_fn: Callable[[float], None],
    log: Callable[[str], None],
) -> None:
    """완료 요청이 프록시 타임아웃에 잘리는 경우까지 감안해 마무리한다.

    서버는 여기서 파일 전체를 재해싱하므로 큰 샤드는 nginx proxy_read_timeout
    (기본 60s)을 넘길 수 있다. 그때 응답만 못 받았을 뿐 서버는 성공했을 수 있어,
    무작정 complete 를 다시 부르면 같은 재해싱을 반복하게 된다.
    그래서 먼저 **상태를 물어보고**, 정말 안 끝났을 때만 다시 부른다.
    """
    for attempt in range(1, max_retries + 1):
        try:
            transport.complete(upload_id)
            return
        except TransientError as error:
            log(
                f"[WARNING] {job.rel_path} 완료 응답을 못 받았습니다 "
                f"({attempt}/{max_retries}): {error}"
            )
            sleep_fn(min(RETRY_BACKOFF_CAP_SEC, RETRY_BACKOFF_BASE_SEC * 2 ** (attempt - 1)))
            try:
                if transport.get_session(upload_id).get("completed"):
                    log(f"[INFO] {job.rel_path} 서버에서는 이미 완료돼 있습니다")
                    return
            except TransientError:
                continue
    raise TransientError(f"{job.rel_path}: 완료 확인 실패 ({max_retries}회 시도)")


def resolve_chunk_size(requested: int, server_limit: int | None) -> int:
    """서버가 알려준 상한에 맞춰 청크 크기를 정한다.

    프록시(nginx client_max_body_size)가 앞에 있으면 큰 청크는 Flask 에
    닿기도 전에 413 으로 잘린다 - 재시도해도 영원히 안 되는 실패라
    아예 상한 아래로 맞춰서 보낸다.
    """
    if not server_limit or server_limit <= 0:
        return requested
    return min(requested, int(server_limit))


def _make_progress(rel_path: str, started_at: float) -> Callable[[int, int], None]:
    """한 줄짜리 진행 표시기를 만든다."""

    def _report(done: int, total: int) -> None:
        elapsed = max(time.time() - started_at, 1e-6)
        rate = done / elapsed
        remaining = (total - done) / rate if rate > 0 else 0
        percent = (done / total * 100) if total else 100.0
        print(
            f"\r[INFO] {rel_path} {percent:5.1f}% "
            f"({_human_bytes(done)}/{_human_bytes(total)}) "
            f"{_human_bytes(rate)}/s ETA {remaining/60:.1f}min",
            end="",
            flush=True,
        )

    return _report


def main() -> int:
    """소스(폴더 또는 파일)를 서버로 올린다."""
    settings = _load_settings()
    transport = HttpTransport(base_url=settings["base_url"], token=settings["token"])

    try:
        health = transport.health()
    except Exception as error:  # 도달성 문제를 업로드 실패로 오해하지 않게 먼저 끊는다
        print(f"[ERROR] 업로드 엔드포인트에 닿지 못했습니다: {error}")
        return 2
    print(f"[INFO] 서버 목적지 루트: {health.get('dest_root')}")

    chunk_size = resolve_chunk_size(settings["chunk_size"], health.get("max_chunk_bytes"))
    if chunk_size != settings["chunk_size"]:
        print(
            f"[WARNING] 청크를 서버 상한에 맞춰 줄입니다: "
            f"{_human_bytes(settings['chunk_size'])} -> {_human_bytes(chunk_size)}"
        )
    settings["chunk_size"] = chunk_size

    jobs = plan_jobs(settings["source"], settings["dest_prefix"])
    if not jobs:
        print(f"[WARNING] 올릴 파일이 없습니다: {settings['source']}")
        return 1

    total_bytes = sum(job.size for job in jobs)
    print(
        f"[INFO] 대상 {len(jobs)}개 파일, 합계 {_human_bytes(total_bytes)}, "
        f"청크 {_human_bytes(settings['chunk_size'])}"
    )

    sent_total = 0
    skipped = 0
    for index, job in enumerate(jobs, start=1):
        print(f"[INFO] ({index}/{len(jobs)}) {job.rel_path} - sha256 계산 중...")
        started_at = time.time()
        try:
            result = upload_file(
                transport,
                job,
                chunk_size=settings["chunk_size"],
                max_retries=settings["max_retries"],
                progress_fn=_make_progress(job.rel_path, started_at),
            )
        except UploadFailed as error:
            print(f"\n[ERROR] {job.rel_path} 실패(재시도 불가): {error}")
            if error.status_code == 413:
                print(
                    "        프록시(nginx client_max_body_size)가 청크를 자르는 중일 수 있습니다. "
                    "MODEL_UPLOAD_CHUNK_MB 를 낮춰 보세요."
                )
            if error.status_code == 401:
                print("        MODEL_UPLOAD_TOKEN 을 확인하세요.")
            return 3
        except (TransientError, RemoteChecksumMismatch) as error:
            print(
                f"\n[ERROR] {job.rel_path} 재시도 한도 초과: {error}\n"
                "        같은 명령을 다시 실행하면 받은 지점부터 이어갑니다."
            )
            return 4

        if result.chunk_size and result.chunk_size < settings["chunk_size"]:
            # 프록시가 알려준 상한은 파일마다 다시 알아낼 필요가 없다.
            settings["chunk_size"] = result.chunk_size
            print(
                f"[INFO] 이후 파일은 청크 {_human_bytes(result.chunk_size)} 로 진행합니다"
            )

        if result.skipped:
            skipped += 1
            print(f"[INFO] ({index}/{len(jobs)}) {job.rel_path} - 이미 완료, 건너뜀")
        else:
            sent_total += result.sent_bytes
            elapsed = max(time.time() - started_at, 1e-6)
            print(
                f"\r[INFO] ({index}/{len(jobs)}) {job.rel_path} - 완료 "
                f"{_human_bytes(result.sent_bytes)} / {elapsed:.1f}s "
                f"({_human_bytes(result.sent_bytes / elapsed)}/s)"
                + " " * 20
            )

    print(
        f"[INFO] 끝. 전송 {_human_bytes(sent_total)}, 건너뜀 {skipped}개, "
        f"총 {len(jobs)}개 파일"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
