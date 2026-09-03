"""model_upload HTTP 계층 테스트.

Flask test client 로 왕복시킨다 - 실서버 불필요:
    uv run pytest flask_api/model_upload/test_routes.py
"""

import hashlib
import io

import pytest
from flask import Flask

from flask_api.model_upload.routes import create_model_upload_blueprint
from flask_api.model_upload.store import UploadStore

DATA = b"0123456789abcdefghij"
CHUNK = 10


def _sha(data: bytes) -> str:
    """테스트용 sha256 헬퍼."""
    return hashlib.sha256(data).hexdigest()


def _make_app(tmp_path, token: str = "", max_chunk_bytes: int = 1024):
    """업로드 blueprint 만 실은 최소 Flask 앱을 만든다."""
    store = UploadStore(dest_root=tmp_path / "models", staging_root=tmp_path / "staging")
    app = Flask(__name__)
    app.register_blueprint(
        create_model_upload_blueprint(
            store=store, token=token, max_chunk_bytes=max_chunk_bytes
        ),
        url_prefix="/api/model_upload",
    )
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(tmp_path):
    """토큰 없는 기본 앱의 test client."""
    return _make_app(tmp_path).test_client()


def _open_session(client, rel_path="MAI-UI-8B/model.safetensors", data=DATA, headers=None):
    """세션 생성 요청을 보낸다."""
    return client.post(
        "/api/model_upload/sessions",
        json={
            "rel_path": rel_path,
            "size": len(data),
            "sha256": _sha(data),
            "chunk_size": CHUNK,
        },
        headers=headers or {},
    )


def _put_chunk(client, upload_id, offset, piece, chunk_sha=None, headers=None):
    """청크 하나를 전송한다."""
    base = {
        "X-Upload-Offset": str(offset),
        "X-Chunk-Sha256": chunk_sha or _sha(piece),
        "Content-Type": "application/octet-stream",
    }
    base.update(headers or {})
    return client.put(
        f"/api/model_upload/sessions/{upload_id}/chunk", data=piece, headers=base
    )


def test_full_roundtrip_places_file_on_disk(client, tmp_path):
    """세션 생성 -> 청크 2개 -> 완료 순서로 목적지에 원본이 그대로 생긴다."""
    opened = _open_session(client)
    assert opened.status_code == 200
    upload_id = opened.get_json()["upload_id"]
    assert opened.get_json()["committed_offset"] == 0

    for start in (0, 10):
        response = _put_chunk(client, upload_id, start, DATA[start : start + CHUNK])
        assert response.status_code == 200, response.get_json()

    done = client.post(f"/api/model_upload/sessions/{upload_id}/complete")

    assert done.status_code == 200
    assert done.get_json()["completed"] is True
    placed = tmp_path / "models" / "MAI-UI-8B" / "model.safetensors"
    assert placed.read_bytes() == DATA


def test_reopening_session_reports_resume_offset(client):
    """끊긴 뒤 세션을 다시 열면 이미 받은 지점을 알려준다."""
    upload_id = _open_session(client).get_json()["upload_id"]
    _put_chunk(client, upload_id, 0, DATA[:CHUNK])

    reopened = _open_session(client)

    assert reopened.get_json()["upload_id"] == upload_id
    assert reopened.get_json()["committed_offset"] == CHUNK


def test_wrong_offset_returns_409_with_expected_offset(client):
    """offset 이 어긋나면 409 와 함께 기대 offset 을 돌려준다."""
    upload_id = _open_session(client).get_json()["upload_id"]

    response = _put_chunk(client, upload_id, 10, DATA[10:])

    assert response.status_code == 409
    assert response.get_json()["expected_offset"] == 0


def test_corrupted_chunk_returns_422_and_keeps_offset(client):
    """손상된 청크는 422 로 거부되고 offset 이 전진하지 않는다."""
    upload_id = _open_session(client).get_json()["upload_id"]

    response = _put_chunk(client, upload_id, 0, b"XXXXXXXXXX", chunk_sha=_sha(DATA[:CHUNK]))

    assert response.status_code == 422
    status = client.get(f"/api/model_upload/sessions/{upload_id}")
    assert status.get_json()["committed_offset"] == 0


def test_path_escaping_root_returns_400(client):
    """루트를 벗어나는 rel_path 는 400 으로 막는다."""
    response = _open_session(client, rel_path="../../etc/passwd")

    assert response.status_code == 400
    assert response.get_json()["code"] == "PathNotAllowed"


def test_unknown_upload_id_returns_404(client):
    """모양은 맞지만 존재하지 않는 세션 조회는 404 다."""
    response = client.get("/api/model_upload/sessions/" + "de" * 16)

    assert response.status_code == 404
    assert response.get_json()["code"] == "UploadNotFound"


def test_malformed_upload_id_is_rejected_before_path_construction(client):
    """우리가 발급하지 않은 모양의 upload_id 는 경로가 되기 전에 막힌다.

    Flask 기본 컨버터는 역슬래시를 허용하므로, 검증이 없으면 Windows 에서
    staging 밖의 파일을 가리킬 수 있다(DELETE 는 unlink 까지 한다).
    """
    for bad in ("deadbeef", "..%5C..%5Cx", "DE" * 16, "z" * 32):
        response = client.get("/api/model_upload/sessions/" + bad)

        assert response.status_code == 400, bad
        assert response.get_json()["code"] == "PathNotAllowed", bad

    response = client.delete("/api/model_upload/sessions/" + "..%5C..%5Cevil")
    assert response.status_code == 400


def test_chunk_larger_than_limit_returns_413(tmp_path):
    """설정된 청크 상한을 넘는 바디는 받지 않는다."""
    client = _make_app(tmp_path, max_chunk_bytes=8).test_client()
    upload_id = _open_session(client).get_json()["upload_id"]

    response = _put_chunk(client, upload_id, 0, DATA[:CHUNK])

    assert response.status_code == 413


def test_token_is_required_when_configured(tmp_path):
    """토큰이 설정되면 헤더 없이는 어떤 업로드 요청도 통과하지 못한다."""
    client = _make_app(tmp_path, token="s3cret").test_client()

    assert _open_session(client).status_code == 401
    assert _open_session(client, headers={"X-Upload-Token": "wrong"}).status_code == 401

    opened = _open_session(client, headers={"X-Upload-Token": "s3cret"})
    assert opened.status_code == 200
    upload_id = opened.get_json()["upload_id"]
    assert (
        _put_chunk(client, upload_id, 0, DATA[:CHUNK]).status_code == 401
    )
    assert (
        _put_chunk(
            client,
            upload_id,
            0,
            DATA[:CHUNK],
            headers={"X-Upload-Token": "s3cret"},
        ).status_code
        == 200
    )


def test_health_needs_no_token(tmp_path):
    """헬스 체크는 토큰 없이도 열려 있다(도달성 확인용)."""
    client = _make_app(tmp_path, token="s3cret").test_client()

    assert client.get("/api/model_upload/health").status_code == 200


def test_chunk_without_content_length_returns_411(client):
    """길이를 알 수 없는 바디는 받지 않는다(구멍 뚫린 .part 방지)."""
    upload_id = _open_session(client).get_json()["upload_id"]

    response = client.put(
        f"/api/model_upload/sessions/{upload_id}/chunk",
        input_stream=io.BytesIO(DATA[:CHUNK]),
        headers={
            "X-Upload-Offset": "0",
            "X-Chunk-Sha256": _sha(DATA[:CHUNK]),
            "Transfer-Encoding": "chunked",
        },
    )

    assert response.status_code == 411


def test_non_integer_fields_return_400_not_500(client, tmp_path):
    """쓰레기 정수 필드는 500 이 아니라 400 으로 돌아온다."""
    response = client.post(
        "/api/model_upload/sessions",
        json={"rel_path": "a.bin", "size": "abc", "sha256": "0" * 64},
    )
    assert response.status_code == 400
    assert response.get_json()["code"] == "UploadError"

    response = client.put(
        "/api/model_upload/sessions/" + "de" * 16 + "/chunk",
        data=b"x",
        headers={"X-Upload-Offset": "not-a-number", "X-Chunk-Sha256": "0" * 64},
    )
    assert response.status_code == 400
