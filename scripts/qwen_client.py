"""qwen3.8-27b 호출 도우미 - thinking / effort / budget 을 한 곳에서 다룬다.

이 폴더의 다른 스크립트가 전부 이 파일을 import 한다. 서버(vLLM 0.19.1)에서 검증된
규칙만 담는다:

  - thinking on/off       : chat_template_kwargs.enable_thinking (기본 on)
  - 추론 강도(effort)      : chat_template_kwargs.reasoning_effort = low | medium | xhigh
                            **top-level reasoning_effort 필드에 넣지 말 것.** 0.19.1 의
                            top-level 필드는 none/low/medium/high 만 받고(xhigh -> 400),
                            high 는 Qwen3.8 chat template 이 raise_exception 으로 거부한다(500).
                            chat_template_kwargs 쪽은 merge 에서 살아남는다(None 만 덮어쓴다).
  - 추론 토큰 상한(budget)  : top-level thinking_token_budget (SamplingParams 로 간다).
                            서버 EXTRA_VLLM_ARGS 에 --reasoning-config 가 있어야 동작한다.
                            없으면 조용히 무시된다 - thinking_budget.py 가 그걸 잡아낸다.
  - 샘플링                 : 모델 카드 권장값. thinking 과 instruct 가 다르다(아래 상수).
  - 이미지                 : image_part(path) / user_message(text, *paths). base64 data URL 로 넣는다.
                            사무실은 오프라인이라 http URL 은 서버가 못 가져오고, file:// 는
                            --allowed-local-media-path 가 없어 거절된다. 서버 한도는 요청당 2장
                            (qwen3.8-27b.env 의 LIMIT_MM_PER_PROMPT).

프록시(/api/vlm_serve/qwen3.8-27b) 대신 vLLM 에 직접 붙는 이유는 check_kv_longctx.py 와
같다 - 프록시는 응답을 끝까지 버퍼링하고 read timeout 300s 라, xhigh 로 몇 분씩
생각하는 요청이 HTTP 경로 때문에 끊긴다. 프록시로 가려면 BASE_URL 을
http://<flask>/api/vlm_serve/qwen3.8-27b 로 바꾸고 TOKEN 을 채운다.
"""

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from typing import Iterator, NamedTuple

# ── 인자는 여기 있다 (이 저장소는 CLI 플래그를 쓰지 않는다) ────────────────
BASE_URL = "http://127.0.0.1:8006"
MODEL = "qwen3.8-27b"
TOKEN = ""  # 프록시 경유 + VLM_SERVE_TOKEN 설정 시에만 채운다
# vLLM 이 --api-key 로 떠 있으면 8006 직결도 인증을 요구한다 (/v1/* 전부, /health 만 열림).
# site.env 의 VLM_SERVE_UPSTREAM_API_KEY 와 같은 값이다. 셸에 export 하거나 여기 직접 채운다.
API_KEY = os.environ.get("VLM_SERVE_UPSTREAM_API_KEY", "").strip()
REQUEST_TIMEOUT_SEC = 1800.0  # xhigh 는 한 요청이 몇 분씩 간다

# 모델 카드(Qwen/Qwen3.8-27B) 권장 샘플링. thinking 이면 전자, 끄면 후자.
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
INSTRUCT_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}

EFFORT_LEVELS = ("low", "medium", "xhigh")  # chat template 이 받는 값 전부. 'high' 는 없다.


class Reply(NamedTuple):
    reasoning: str
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_sec: float

    def as_assistant_message(self):
        """다음 턴에 되돌려 줄 assistant 메시지.

        reasoning_content 를 같이 보내면 chat template(preserve_thinking 기본 true)이
        이전 턴의 사고 과정을 프롬프트에 다시 넣는다. 에이전트 루프에서 권장되는 방식이다.
        """
        message = {"role": "assistant", "content": self.content}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        return message


def image_part(path):
    """이미지 파일 하나 -> OpenAI 식 image_url 콘텐츠 파트 (base64 data URL)."""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def user_message(text, *image_paths):
    """텍스트 + 이미지 0~N 장을 담은 user 메시지. 이미지가 없으면 content 는 문자열 그대로다."""
    if not image_paths:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [*map(image_part, image_paths), {"type": "text", "text": text}]}


def build_request(messages, *, thinking=True, effort="xhigh", budget=None,
                  max_tokens=16384, stream=False, **sampling_overrides):
    """/v1/chat/completions body 를 만든다. 검증된 위치에만 값을 넣는다."""
    if effort not in EFFORT_LEVELS:
        raise ValueError(f"effort 는 {EFFORT_LEVELS} 중 하나여야 한다: {effort!r}")
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        **(THINKING_SAMPLING if thinking else INSTRUCT_SAMPLING),
        **sampling_overrides,
    }
    if thinking:
        body["chat_template_kwargs"]["reasoning_effort"] = effort
        if budget is not None:
            body["thinking_token_budget"] = int(budget)
    return body


def _headers():
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["X-VLM-Token"] = TOKEN
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _open(path, payload):
    request = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC)


def _reasoning_of(message):
    # 0.19.1 은 'reasoning', 그 이전 vLLM 은 'reasoning_content'. 둘 다 받는다.
    return message.get("reasoning") or message.get("reasoning_content") or ""


def parse_reply(body, elapsed_sec=0.0):
    """non-stream 응답 JSON -> Reply."""
    choice = body["choices"][0]
    message = choice["message"]
    usage = body.get("usage") or {}
    return Reply(
        reasoning=_reasoning_of(message),
        content=message.get("content") or "",
        finish_reason=choice.get("finish_reason") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        elapsed_sec=elapsed_sec,
    )


def chat(messages, **kw):
    """한 번 묻고 Reply 를 돌려준다. HTTPError 는 본문을 붙여 다시 던진다(400/500 원인이 거기 있다)."""
    started = time.monotonic()
    try:
        with _open("/v1/chat/completions", build_request(messages, stream=False, **kw)) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from None
    return parse_reply(body, time.monotonic() - started)


def iter_sse(lines) -> Iterator[tuple[str, str]]:
    """SSE 줄들 -> ('reasoning'|'content', 조각). [DONE] 에서 멈춘다."""
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        chunk = json.loads(data)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning = _reasoning_of(delta)
            if reasoning:
                yield "reasoning", reasoning
            if delta.get("content"):
                yield "content", delta["content"]


def stream_chat(messages, **kw) -> Iterator[tuple[str, str]]:
    """조각 단위로 흘려준다. thinking 이 끝나고 답이 시작되는 순간을 볼 수 있다."""
    with _open("/v1/chat/completions", build_request(messages, stream=True, **kw)) as response:
        yield from iter_sse(response)


def count_tokens(text):
    """vLLM /tokenize 로 실제 토큰 수를 센다. 실패하면 None. usage 에는 reasoning 분리 집계가 없다."""
    try:
        with _open("/tokenize", {"model": MODEL, "prompt": text}) as response:
            return json.loads(response.read().decode("utf-8")).get("count")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
