"""도구 호출(function calling)이 서버에서 실제로 파싱되는지 한 번에 확인한다.

두 가지를 찍는다:
  1. 서버가 chat template 로 렌더링한 프롬프트의 끝부분 - template 이 모델에게 **어떤 형식으로**
     도구를 부르라고 시키는지 그대로 보인다 (JSON `{"name": ...}` 인지 `<function=...>` XML 인지).
     /tokenize -> /detokenize 로 우회한다. 가중치를 건드리지 않으니 몇 ms 면 끝난다.
  2. tools 를 붙인 chat completion 하나. 500 이면 파서가 죽은 것이고, 200 이어도 tool_calls 가
     비어 있고 content 에 `<tool_call>` 이 그대로 남아 있으면 파서가 형식을 못 알아본 것이다.

RED/GREEN 을 마지막 줄에 찍는다. GPU 서버에서 돈다 (BASE_URL 은 qwen_client.py).

사용법:
  python scripts/check_tool_call.py
"""

import json
import sys
import urllib.error
from typing import Any

from qwen_client import MODEL, _open

# ── 인자는 여기 있다 ───────────────────────────────────────────────────────
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "도시의 현재 날씨를 돌려준다.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "도시 이름"}},
            "required": ["city"],
        },
    },
}]
MESSAGES = [{"role": "user", "content": "서울 날씨 알려줘. 반드시 get_weather 도구를 써라."}]
PROMPT_TAIL_CHARS = 1500  # 렌더링된 프롬프트에서 뒤에서부터 보여줄 글자 수


def _post(path, payload) -> tuple[int, Any]:
    try:
        with _open(path, payload) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def show_rendered_prompt():
    status, body = _post("/tokenize", {
        "model": MODEL, "messages": MESSAGES, "tools": TOOLS, "add_generation_prompt": True,
    })
    if status != 200:
        print(f"[WARNING] /tokenize {status}: {str(body)[:300]}")
        return None
    status, body = _post("/detokenize", {"model": MODEL, "tokens": body["tokens"]})
    if status != 200:
        print(f"[WARNING] /detokenize {status}: {str(body)[:300]}")
        return None
    prompt = body["prompt"]
    print("[INFO] 렌더링된 프롬프트 끝부분 (template 이 요구하는 도구 호출 형식):")
    print("─" * 70)
    print(prompt[-PROMPT_TAIL_CHARS:])
    print("─" * 70)
    fmt = "xml(<function=...>)" if "<function=" in prompt else "json({\"name\": ...})" if '"name"' in prompt else "unknown"
    print(f"[INFO] template 도구 형식 = {fmt}\n")
    return fmt


def probe_tool_call():
    status, body = _post("/v1/chat/completions", {
        "model": MODEL, "messages": MESSAGES, "tools": TOOLS, "tool_choice": "auto",
        "temperature": 0.0, "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    print(f"[INFO] /v1/chat/completions -> HTTP {status}")
    if status != 200:
        print(str(body)[:2000])
        return False, "서버가 5xx/4xx 를 냈다 - 위 본문의 JSONDecodeError 가 파서 실패다"
    choice = body["choices"][0]
    message = choice["message"]
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    print(f"[INFO] finish_reason={choice.get('finish_reason')} tool_calls={len(tool_calls)}")
    print(f"[INFO] content={content[:600]!r}")
    for call in tool_calls:
        print(f"[INFO] tool_call: {json.dumps(call.get('function'), ensure_ascii=False)}")
    if tool_calls:
        return True, "tool_calls 가 파싱되어 돌아왔다"
    if "<tool_call>" in content or "<function=" in content:
        return False, "모델은 도구를 불렀지만 파서가 형식을 못 알아봐 본문 텍스트로 새어 나왔다"
    return False, "모델이 도구를 부르지 않았다 (프롬프트/모델 문제, 파서 문제 아님)"


def main():
    show_rendered_prompt()
    ok, why = probe_tool_call()
    print(f"\n{'GREEN' if ok else 'RED'}: {why}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
