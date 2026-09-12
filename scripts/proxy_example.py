"""프록시(`/api/vlm_serve/<slug>`) 경유로 thinking · effort · budget 을 조절하는 실행 예제.

**결론부터: 8006 직결과 body 가 완전히 같다.** 프록시는 요청 body 를 파싱하지도 재직렬화하지도
않고 그대로 업스트림에 넘긴다(`service_template.py:_proxy_request`). 그래서 knob 의 위치·이름·
함정이 직결과 1:1 이다 - 바꿀 것은 주소 하나뿐이다. 이 계약은
`tests/test_vlm_serve.py::test_reasoning_knobs_reach_upstream_untouched` 가 지킨다.

    qwen_client.BASE_URL = "http://<flask>/api/vlm_serve/qwen3.8-27b"   # 이게 전부다

직결과 **다른 점은 하나뿐이다: 긴 요청은 `stream=True` 로 보내라.**
SSE 는 청크 단위로 relay 되므로 read timeout 이 청크 사이 간격에만 걸린다. non-stream 은
프록시가 끝까지 버퍼링하니 `VLM_SERVE_READ_TIMEOUT_SEC`(300s) 안에 못 끝내는 xhigh 요청이
답과 무관하게 HTTP 경로에서 끊긴다. 그래서 아래 step 3/4 는 스트리밍이다.

모델별 차이: **reasoning knob 이 있는 모델은 qwen3.8-27b 하나뿐이다.**

    qwen3.8-27b (8006)        enable_thinking · reasoning_effort · thinking_token_budget
    mai-ui (8002)             없음 - grounding 모델, --reasoning-parser 가 안 붙어 있다
    paddleocr-vl-1.5 (8004)   없음 - OCR 모델

프록시는 모델별 분기를 하지 않는다(할 게 없다). 없는 모델에 knob 을 보내면 chat template 이
모르는 kwarg 로 무시하거나 400 을 준다 - 프록시가 삼키지 않고 그대로 돌려준다.

사용법:
  python scripts/proxy_example.py        # PROXY_BASE_URL 을 자기 Flask 주소로 고친 뒤
"""

import json
import time
import urllib.error
import urllib.request

import qwen_client
from qwen_client import EFFORT_LEVELS, chat, stream_chat

# ── 인자는 여기 있다 (이 저장소는 CLI 플래그를 쓰지 않는다) ────────────────
# 로컬 개발이면 5000 (python index.py), 서버 배포면 nginx 주소.
PROXY_BASE_URL = "http://127.0.0.1:5000/api/vlm_serve/qwen3.8-27b"
QUESTION = "347 × 829 를 계산하라. 마지막 줄에 '정답: <값>' 형식으로만 답하라."
MAX_TOKENS = 4096
BUDGET = 512  # step 4 의 사고 토큰 상한


def _get(path):
    """프록시에 GET. 키는 qwen_client 와 같은 VLLM_API_KEY 하나다."""
    headers = {"Authorization": f"Bearer {qwen_client.API_KEY}"} if qwen_client.API_KEY else {}
    request = urllib.request.Request(f"{PROXY_BASE_URL.rstrip('/')}{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def step1_reachable():
    """도달 + 인증 확인. 키가 틀리면 프록시가 401 을 준다(업스트림까지 가지 않는다)."""
    print(f"\n== step 1: GET {PROXY_BASE_URL}/v1/models")
    try:
        payload = _get("/v1/models")
    except urllib.error.HTTPError as error:
        # 401 은 프록시의 before_request, 그 외는 업스트림 vLLM 이 준 것이다.
        print(f"[ERROR] HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:200]}")
        print("[INFO] 401 이면 VLLM_API_KEY 를 export 하거나 site.env 를 확인하라.")
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"[ERROR] 프록시에 닿지 않는다: {error}")
        print(f"[INFO] PROXY_BASE_URL 을 고쳐라 (현재 {PROXY_BASE_URL}).")
        return False
    print(f"[INFO] served models = {[item['id'] for item in payload.get('data', [])]}")
    return True


def step2_instruct():
    """thinking off - instruct 모드. 짧으니 non-stream 으로 충분하다."""
    print("\n== step 2: enable_thinking=False (non-stream)")
    reply = chat([{"role": "user", "content": QUESTION}], thinking=False, max_tokens=MAX_TOKENS)
    print(f"[INFO] {reply.elapsed_sec:.1f}s  completion={reply.completion_tokens}  "
          f"reasoning={'있음' if reply.reasoning else '없음(기대값)'}")
    print(f"[INFO] {reply.content.strip().splitlines()[-1][:60]}")


def step3_effort():
    """effort 단계별 스트리밍. 사고 첫 조각까지의 시간(TTFT)이 단계마다 갈린다."""
    for effort in EFFORT_LEVELS:
        print(f"\n== step 3: reasoning_effort={effort} (stream)")
        started = time.monotonic()
        first_reasoning = None
        think_chars, answer = 0, []
        for kind, piece in stream_chat([{"role": "user", "content": QUESTION}],
                                       thinking=True, effort=effort, max_tokens=MAX_TOKENS):
            if kind == "reasoning":
                if first_reasoning is None:
                    first_reasoning = time.monotonic() - started
                think_chars += len(piece)
            else:
                answer.append(piece)
        print(f"[INFO] ttft={first_reasoning or 0:.1f}s  total={time.monotonic() - started:.1f}s  "
              f"think_chars={think_chars}")
        print(f"[INFO] {''.join(answer).strip().splitlines()[-1][:60] if answer else '(답 없음)'}")


def step4_budget():
    """thinking_token_budget - 사고를 강제로 끊는다. 서버 --reasoning-config 가 있어야 먹는다."""
    print(f"\n== step 4: thinking_token_budget={BUDGET} (stream)")
    think_chars, answer = 0, []
    for kind, piece in stream_chat([{"role": "user", "content": QUESTION}],
                                   thinking=True, effort="xhigh", budget=BUDGET,
                                   max_tokens=MAX_TOKENS):
        if kind == "reasoning":
            think_chars += len(piece)
        else:
            answer.append(piece)
    print(f"[INFO] think_chars={think_chars} (step 3 의 xhigh 보다 작아야 budget 이 먹은 것)")
    print(f"[INFO] {''.join(answer).strip().splitlines()[-1][:60] if answer else '(답 없음)'}")


def step5_the_trap():
    """top-level reasoning_effort 는 넣지 말 것. 프록시는 이 400 도 그대로 돌려준다.

    프록시가 에러를 삼키지 않는다는 확인이기도 하다 - 실패 원인이 body 에 그대로 온다.
    """
    print("\n== step 5: top-level reasoning_effort='xhigh' -> 400 이 정상이다")
    payload = {
        "model": qwen_client.MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 32,
        "reasoning_effort": "xhigh",  # 여기가 함정. chat_template_kwargs 안에 넣어야 한다.
    }
    try:
        with qwen_client._open("/v1/chat/completions", payload) as response:
            body = json.loads(response.read().decode("utf-8"))
        print(f"[WARNING] 400 을 기대했는데 통과했다 - vLLM 버전이 바뀐 것이다: "
              f"{body['choices'][0]['message'].get('content', '')[:60]}")
    except urllib.error.HTTPError as error:
        print(f"[INFO] 기대대로 HTTP {error.code}: "
              f"{error.read().decode('utf-8', 'replace')[:120]}")


def main():
    qwen_client.BASE_URL = PROXY_BASE_URL  # ← 직결을 프록시로 바꾸는 유일한 한 줄
    print(f"[INFO] BASE_URL = {qwen_client.BASE_URL}")
    print(f"[INFO] API_KEY = {'설정됨' if qwen_client.API_KEY else '비어 있음(인증 off 배포)'}")
    if not step1_reachable():
        return
    step2_instruct()
    step3_effort()
    step4_budget()
    step5_the_trap()


if __name__ == "__main__":
    main()
