"""qwen_client 의 요청 조립과 응답 해석 - 서버 없이 돈다."""

import pytest

from qwen_client import EFFORT_LEVELS, build_request, image_part, iter_sse, parse_reply, user_message


def test_effort_goes_into_chat_template_kwargs_not_top_level():
    body = build_request([{"role": "user", "content": "x"}], thinking=True, effort="xhigh", budget=300)
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    assert "reasoning_effort" not in body  # 0.19.1 top-level 은 xhigh 를 400 으로 거부한다
    assert body["thinking_token_budget"] == 300
    assert body["temperature"] == 1.0 and body["presence_penalty"] == 0.0


def test_thinking_off_drops_effort_and_budget_and_uses_instruct_sampling():
    body = build_request([], thinking=False, effort="low", budget=300, temperature=0.2)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_token_budget" not in body
    assert body["presence_penalty"] == 1.5
    assert body["temperature"] == 0.2  # 명시한 값이 권장값을 덮는다


def test_high_is_not_a_level():
    assert "high" not in EFFORT_LEVELS
    with pytest.raises(ValueError):
        build_request([], effort="high")


def test_parse_reply_accepts_both_reasoning_field_names():
    body = {"choices": [{"finish_reason": "stop", "message": {"content": "답", "reasoning": "생각"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    reply = parse_reply(body, 1.5)
    assert (reply.reasoning, reply.content, reply.completion_tokens) == ("생각", "답", 20)
    assert reply.as_assistant_message() == {"role": "assistant", "content": "답", "reasoning_content": "생각"}

    body["choices"][0]["message"] = {"content": "답", "reasoning_content": "옛 필드"}
    assert parse_reply(body).reasoning == "옛 필드"


def test_iter_sse_splits_reasoning_and_content_and_stops_at_done():
    lines = [
        b'data: {"choices":[{"delta":{"reasoning":"a"}}]}',
        b"",
        b'data: {"choices":[{"delta":{"content":"b"}}]}',
        b"data: [DONE]",
        b'data: {"choices":[{"delta":{"content":"IGNORED"}}]}',
    ]
    assert list(iter_sse(lines)) == [("reasoning", "a"), ("content", "b")]


def test_user_message_embeds_images_as_base64_data_urls(tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG")
    assert user_message("x") == {"role": "user", "content": "x"}
    msg = user_message("설명", png, png)
    assert msg["content"][-1] == {"type": "text", "text": "설명"}
    assert msg["content"][0] == image_part(png)
    assert msg["content"][0]["image_url"]["url"] == "data:image/png;base64,iVBORw=="
    odd = tmp_path / "b.unknownext"
    odd.write_bytes(b"")
    assert image_part(odd)["image_url"]["url"].startswith("data:image/png;base64,")  # 확장자 모르면 png


def test_proxy_example_rebinds_base_url_so_requests_go_through_the_proxy(monkeypatch):
    """proxy_example 의 "한 줄만 바꾼다" 주장이 실제로 _open 까지 닿는지.

    qwen_client.BASE_URL 을 모듈 속성으로 덮어쓰는 방식이라, _open 이 그 값을 호출 시점에
    읽지 않으면(예: from-import 로 값을 복사해 두면) 조용히 8006 직결로 나간다.
    """
    import qwen_client
    import proxy_example

    seen: dict[str, str] = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        # step 1 이 잡는 예외로 끊는다 - main() 이 뒤 step 으로 넘어가지 않고 돌아온다.
        raise qwen_client.urllib.error.URLError("stop here - 주소만 확인한다")

    monkeypatch.setattr(qwen_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(proxy_example, "PROXY_BASE_URL", "http://flask.example/api/vlm_serve/qwen3.8-27b")
    monkeypatch.setattr(qwen_client, "BASE_URL", "http://127.0.0.1:8006")

    proxy_example.main()

    assert seen["url"] == "http://flask.example/api/vlm_serve/qwen3.8-27b/v1/models"
