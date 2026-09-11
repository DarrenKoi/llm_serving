"""check_model 이 업스트림 인증 키를 붙이는지 확인한다.

vLLM 을 `--api-key` 로 띄우면 `/v1/*` 가 전부 401 이 된다(`/health` 만 열려 있다).
start_all.py 는 이 함수의 결과를 "떴는가"로 읽고, 안 떴다고 판단하면 해당 인스턴스를
**중지시킨다**. 즉 헤더 하나가 빠지면 증상은 401 이 아니라 "전체 기동 실패"다.

    pytest deploy_vlms/scripts/test_check_vlm_auth.py
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("check_vlm", _SCRIPTS / "check_vlm.py")
assert _SPEC and _SPEC.loader  # spec_from_file_location 은 Optional 을 낸다
check_vlm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_vlm)


@pytest.fixture
def captured(monkeypatch):
    """urlopen 을 가로채 Request 를 기록하고 정상 /v1/models 응답을 돌려준다."""
    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(request, timeout=None):
        seen["request"] = request
        body = json.dumps({"data": [{"id": "qwen3.8-27b"}]}).encode("utf-8")
        return _Resp(body)

    monkeypatch.setattr(check_vlm.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_sends_bearer_token_when_key_is_configured(captured, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "team-token")

    ok, reason = check_vlm.check_model("127.0.0.1", 8006, "qwen3.8-27b")

    assert ok, reason
    assert captured["request"].get_header("Authorization") == "Bearer team-token"


def test_sends_no_auth_header_when_key_is_absent(captured, monkeypatch):
    """키를 안 쓰는 배포에서 헤더를 만들어 보내면 안 된다."""
    monkeypatch.setenv("VLLM_API_KEY", "")
    monkeypatch.setenv("SITE_ENV", str(_SCRIPTS / "does-not-exist.env"))

    ok, reason = check_vlm.check_model("127.0.0.1", 8006, "qwen3.8-27b")

    assert ok, reason
    assert captured["request"].get_header("Authorization") is None


def test_reads_key_from_site_env_when_shell_is_empty(captured, monkeypatch, tmp_path):
    """서버에는 셸 export 가 없다 - site.env 에서 읽어야 한다."""
    site_env = tmp_path / "site.env"
    site_env.write_text("VLLM_API_KEY=from-site-env\n", encoding="utf-8")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("SITE_ENV", str(site_env))

    check_vlm.check_model("127.0.0.1", 8006, "qwen3.8-27b")

    assert captured["request"].get_header("Authorization") == "Bearer from-site-env"
