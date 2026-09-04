"""qwen_client 의 요청 조립과 응답 해석 - 서버 없이 돈다."""

import pytest

from qwen_client import EFFORT_LEVELS, build_request, iter_sse, parse_reply


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
