"""모델 가중치 청크 업로드 HTTP 계층.

바디는 절대 통째로 읽지 않는다 - request.stream 을 블록 단위로 흘려서
store 가 곧바로 디스크에 쓴다. 그래야 32MB 청크가 32MB 버퍼로 끝난다.
"""

from hmac import compare_digest

from flask import Blueprint, jsonify, request

from .store import (
    ChunkTooLarge,
    LengthRequired,
    UploadError,
    UploadSession,
    UploadStore,
    UploadUnauthorized,
)


def _int_field(value: object, name: str) -> int:
    """정수 필드를 읽는다 - 쓰레기 입력은 500 이 아니라 400 으로 돌려준다."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise UploadError(f"{name} must be an integer, got {value!r}")


def _session_payload(session: UploadSession) -> dict[str, object]:
    """세션 상태를 JSON 응답 형태로 만든다."""
    return {
        "upload_id": session.upload_id,
        "rel_path": session.rel_path,
        "size": session.size,
        "sha256": session.sha256,
        "chunk_size": session.chunk_size,
        "committed_offset": session.committed_offset,
        "completed": session.completed,
    }


def create_model_upload_blueprint(
    store: UploadStore, token: str = "", max_chunk_bytes: int = 64 * 1024 * 1024
) -> Blueprint:
    """업로드 blueprint 를 만든다(store 주입식이라 테스트에서 격리된다)."""
    blueprint = Blueprint("model_upload", __name__)

    def _require_token() -> None:
        """토큰이 설정돼 있으면 헤더를 확인한다."""
        if not token:
            return
        provided = request.headers.get("X-Upload-Token", "").strip()
        if not provided:
            authorization = request.headers.get("Authorization", "").strip()
            if authorization.lower().startswith("bearer "):
                provided = authorization[7:].strip()
        if not compare_digest(provided, token):
            raise UploadUnauthorized("missing or invalid upload token")

    @blueprint.errorhandler(UploadError)
    def _handle_upload_error(error: UploadError):
        """업로드 계층 예외를 상태코드 있는 JSON 으로 바꾼다."""
        payload: dict[str, object] = {
            "error": str(error),
            "code": type(error).__name__,
        }
        expected = getattr(error, "expected_offset", None)
        if expected is not None:
            payload["expected_offset"] = expected
        return jsonify(payload), error.status_code

    @blueprint.route("/health", methods=["GET"])
    def health():
        """업로드 엔드포인트 헬스 체크."""
        return jsonify(
            {
                "service": "model_upload",
                "status": "ok",
                "dest_root": str(store.dest_root),
                "max_chunk_bytes": max_chunk_bytes,
                "auth_required": bool(token),
            }
        )

    @blueprint.route("/sessions", methods=["POST"])
    def open_session():
        """업로드 세션을 만들거나 이어받는다."""
        _require_token()
        body = request.get_json(silent=True) or {}
        session = store.begin(
            rel_path=str(body.get("rel_path", "")),
            size=_int_field(body.get("size", 0), "size"),
            sha256=str(body.get("sha256", "")),
            chunk_size=_int_field(body.get("chunk_size", 0), "chunk_size"),
        )
        return jsonify(_session_payload(session))

    @blueprint.route("/sessions/<upload_id>", methods=["GET"])
    def get_session(upload_id: str):
        """세션 상태를 조회한다."""
        _require_token()
        return jsonify(_session_payload(store.status(upload_id)))

    @blueprint.route("/sessions/<upload_id>", methods=["DELETE"])
    def delete_session(upload_id: str):
        """세션을 버리고 staging 을 정리한다."""
        _require_token()
        store.abort(upload_id)
        return jsonify({"upload_id": upload_id, "aborted": True})

    @blueprint.route("/sessions/<upload_id>/chunk", methods=["PUT"])
    def put_chunk(upload_id: str):
        """청크 바디를 스트리밍으로 받아 .part 에 이어붙인다."""
        _require_token()
        length = request.content_length
        if length is None:
            raise LengthRequired("Content-Length header is required for chunk upload")
        if length > max_chunk_bytes:
            raise ChunkTooLarge(
                f"chunk of {length} bytes exceeds limit {max_chunk_bytes}"
            )

        session = store.append_chunk(
            upload_id=upload_id,
            offset=_int_field(
                request.headers.get("X-Upload-Offset", "-1"), "X-Upload-Offset"
            ),
            stream=request.stream,
            length=int(length),
            chunk_sha256=request.headers.get("X-Chunk-Sha256", ""),
        )
        return jsonify(_session_payload(session))

    @blueprint.route("/sessions/<upload_id>/complete", methods=["POST"])
    def complete_session(upload_id: str):
        """전체 해시를 검증하고 목적지로 옮긴다."""
        _require_token()
        session = store.finish(upload_id)
        payload = _session_payload(session)
        payload["path"] = str(store.dest_root / session.rel_path)
        return jsonify(payload)

    return blueprint
