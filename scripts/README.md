# scripts/ - qwen3.8-27b 를 끝까지 써 보기 (thinking · effort · budget)

서버(vLLM 0.19.1, `--reasoning-parser qwen3`)에서 **검증된 조작법**만 적는다. 근거는
vLLM v0.19.1 소스와 `Qwen/Qwen3.8-27B` 의 `chat_template.jinja` 다. 스크립트는 전부
stdlib 만 쓰고 `qwen_client.py` 하나를 공유한다. CLI 인자는 없다 - 상수 블록을 고친다.

| 스크립트 | 무엇을 재나 | 소요 |
|---|---|---|
| `effort_ladder.py` | 과제 7개 × (off / low / medium / xhigh) - 정확도·사고 토큰·시간 | 5-15분 |
| `stream_thinking.py` | 프롬프트 하나를 스트리밍 - 사고와 답이 갈리는 순간, TTFT | 1분 |
| `thinking_budget.py` | `thinking_token_budget` 이 먹는지, 어디서 품질이 꺾이는지 | 5분 |
| `check_tool_call.py` | template 의 도구 호출 형식 + tools 요청 하나가 tool_calls 로 파싱되는지 (RED/GREEN) | 10초 |

```bash
python scripts/effort_ladder.py     # 먼저 이걸로 "어느 단계가 기본값이면 되는지" 정한다
python scripts/stream_thinking.py
python scripts/thinking_budget.py
python scripts/check_tool_call.py    # 도구 호출이 파서에서 새면 여기서 RED
pytest scripts/                     # 서버 없이 요청 조립·응답 해석만 검증
```

## 조작 레버 세 개

### 1. thinking on/off - `chat_template_kwargs.enable_thinking`

기본 **on**. 끄면 template 이 `<think>\n\n</think>` 를 프롬프트에 박아 모델이 바로 답한다.
분류·추출·짧은 Q&A 처럼 `effort_ladder.py` 의 off 행에서도 맞는 과제는 끄는 것이 정답이다.
샘플링도 달라진다(모델 카드 권장값, `qwen_client.py` 상수):

| 모드 | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| thinking | 1.0 | 0.95 | 20 | 0.0 |
| instruct(off) | 0.7 | 0.80 | 20 | 1.5 |

### 2. 추론 강도 - `chat_template_kwargs.reasoning_effort` = `low` · `medium` · `xhigh`

기본 **medium**. 원래 template 기본값은 xhigh 인데, `qwen3.8-27b.env` 가
`--default-chat-template-kwargs '{"reasoning_effort": "medium"}'` 로 medium 을 깔아 둔다
(코딩 하네스가 이 필드를 안 보내기 때문 - 아래 "코딩 하네스" 절). 요청이 보낸
`chat_template_kwargs` 는 이 기본값을 덮는다. 이건 상한이 아니라 **프롬프트 문구**다 - template 이 시스템 프롬프트에
"Reasoning effort is set to xhigh. Please think carefully…" 또는 "…low. Keep your thinking
brief…" 를 넣을 뿐이다(medium 은 아무 문구도 안 넣는다). 그래서 low 라도 어려운 문제엔
길게 생각할 수 있다. 강제 상한은 3번이다.

**함정 - top-level `reasoning_effort` 필드에 넣지 말 것.** 0.19.1 의 top-level 필드는
`none|low|medium|high` 만 받는다. `xhigh` 는 400, `high` 는 template 의 `raise_exception`
으로 500 이 난다(QwenLM/Qwen3.8#217 이 그 현상이다). `chat_template_kwargs` 안의 값은
merge 에서 살아남는다(top-level 이 None 이면 덮지 않는다). Claude Code 나 OpenAI SDK 의
`reasoning_effort=` 파라미터가 바로 그 top-level 필드로 나간다 - 쓰지 말고 `extra_body` 로:

```python
client.chat.completions.create(
    model="qwen3.8-27b", messages=..., temperature=1.0, top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}},
)
```

### 3. 추론 토큰 상한 - `thinking_token_budget` (top-level)

budget 에 닿으면 vLLM 이 `</think>` 를 강제로 넣어 답으로 넘긴다. Qwen3.5+ template 처럼
`<think>` 가 프롬프트 쪽에 있어도 잡는다(0.19.1 의 `ThinkingTokenBudgetLogitsProcessor` 가
프롬프트 토큰을 뒤진다). **서버 인자가 필요하다** - `qwen3.8-27b.env` 의 `EXTRA_VLLM_ARGS` 끝에:

```
--reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}'
```

없으면 200 으로 정상 응답하면서 budget 만 조용히 무시된다. `thinking_budget.py` 가
"사고 토큰 > budget" 으로 그걸 잡아 `[WARNING]` 을 찍는다.

### 4. 이미지 - `user_message(text, *paths)` / `image_part(path)`

base64 data URL 로 넣는다. 사무실은 오프라인이라 http URL 은 서버가 못 가져오고, `file://` 는
`--allowed-local-media-path` 가 없어 거절된다. 서버 한도는 요청당 **2장**
(`qwen3.8-27b.env` 의 `LIMIT_MM_PER_PROMPT`). 도구 호출과 한 요청에 섞어도 된다.

```python
from qwen_client import chat, user_message
reply = chat([user_message("두 화면의 차이를 설명해라.", "before.png", "after.png")], effort="medium")
```

## 그 밖에 알아둘 것

- **`max_tokens` 는 사고 + 답의 합이다.** xhigh 에서 `finish_reason=length` 가 나오면 답이
  아니라 사고가 잘린 것이다. 모델 카드는 에이전트 과제에 사고 최대 262144 / 답 131072 를
  권하지만 그건 1M 컨텍스트 기준이다. 스크립트 기본값은 16384 다.
- **응답 필드는 `message.reasoning`** 이다(0.19.1). 이전 vLLM 의 `reasoning_content` 도
  `qwen_client.py` 가 같이 받는다. 스트리밍은 `delta.reasoning` 조각이 먼저 오고
  `delta.content` 가 뒤따른다. `include_reasoning: false` 면 사고를 안 보내준다(모델 품질은
  그대로, 트래픽만 준다).
- **멀티턴에서 사고를 이어가려면** assistant 메시지에 `reasoning_content` 를 되돌려 보낸다
  (`Reply.as_assistant_message()`). template 의 `preserve_thinking` 기본값이 true 라 이전
  턴 사고가 프롬프트에 다시 들어간다. 마지막 턴 것만 남기려면
  `chat_template_kwargs.preserve_thinking=false`.
- **`usage` 에 사고 토큰이 따로 없다.** 스크립트는 `/tokenize` 로 `reasoning` 문자열을
  다시 세서 사고/답을 나눈다.
- **프록시 대신 vLLM 에 직접 붙는다**(`127.0.0.1:8006`, 다른 장비에선 `<사내IP>:8006`).
  `common.env` 의 `HOST=0.0.0.0` + `API_KEY` 라 사내에서 바로 닿고 키가 필요하다
  (`site.env` 의 `VLM_SERVE_UPSTREAM_API_KEY`). `/api/vlm_serve/...` 는 응답을
  끝까지 버퍼링하고 read timeout 이 300s 라, xhigh 로 몇 분씩 생각하는 요청이 HTTP 경로
  때문에 끊긴다. 프록시로 가려면 `BASE_URL` 을 `http://<flask>/api/vlm_serve/qwen3.8-27b`
  로, 토큰이 있으면 `TOKEN` 을 채운다.
- **tool calling** 은 이미 켜져 있다 - `EXTRA_VLLM_ARGS` 의 `--enable-auto-tool-choice
  --tool-call-parser qwen3_xml`. template 의 tool 포맷이 `<tool_call><function=…><parameter=…>`
  라 XML 파서여야 한다(`hermes` 는 JSON 전용이라 못 쓴다). 코딩 하네스는 전부 이걸 쓴다.
- **1M 컨텍스트**는 `--max-model-len 1010000 --hf-overrides '{"text_config":
  {"max_position_embeddings": 1010000}}'` 로 열린다. fp8 KV 기준 시퀀스당 32GiB 라
  `MAX_NUM_SEQS` 를 2 이하로 내려야 하고, `check_kv_longctx.py` 를 그 길이로 다시 돌려야
  한다. 필요가 생기기 전엔 262144 로 둔다.

## 코딩 하네스 붙이기 (opencode · pi)

`stream: true` 는 서버 설정이 아니라 요청 필드다 - vLLM 은 이미 SSE 를 준다. 붙는 주소는 둘 중 하나:

| 주소 | 언제 | 키 |
|---|---|---|
| `http://<사내IP>:8006/v1` | 8006 포트가 클라이언트에서 열릴 때 | `VLM_SERVE_UPSTREAM_API_KEY` |
| `http://<webapp 호스트>/api/vlm_serve/qwen3.8-27b/v1` | 웹앱 호스트(ingress)만 닿을 때 | `VLM_SERVE_TOKEN` (비어 있으면 아무 값) |

프록시는 SSE 응답을 청크 단위로 흘리므로(2026-09-10) 둘 다 스트리밍이 된다. 프록시 경로는
호출자의 Authorization 을 떼고 업스트림 키를 대신 넣으므로 하네스가 vLLM 키를 알 필요가 없다.
단 프록시 경로는 nginx 600s > harakiri 870 > 앱 read timeout 300s 가 걸린다 - 읽기 timeout 은
청크 사이 간격이라 사고가 흐르는 동안은 안 끊기지만, 한 요청이 870s 를 넘기면 uWSGI 가 자른다.

**효 강도는 반드시 `chat_template_kwargs` 로 보낸다.** top-level `reasoning_effort` 는
0.19.1 에서 `xhigh` → 400, `high` → template `raise_exception` → 500 이다(위 §2 의 함정).
하네스의 "reasoning effort" UI 가 기본적으로 내보내는 곳이 바로 그 top-level 필드라,
아래 설정의 핵심은 **그 경로를 끄고 chat_template_kwargs 로 우회시키는 것**이다.

서버가 medium 을 깔아 두므로, 아무것도 설정하지 않아도 하네스는 medium 으로 돈다.
아래는 사용자가 강도를 **고를 수 있게** 하는 설정이다.

### pi

`compat.thinkingFormat: "chat-template"` 가 정확히 이 용도다. `supportsReasoningEffort: false`
로 top-level 필드를 막고, `thinkingLevelMap` 으로 pi 의 단계를 모델 어휘(low/medium/xhigh)에 맞춘다.

```jsonc
{
  "providers": {
    "vllm": {
      "baseUrl": "http://<사내IP>:8006/v1",
      "api": "openai-completions",
      "apiKey": "$VLLM_API_KEY",      // site.env 의 VLM_SERVE_UPSTREAM_API_KEY 와 같은 값
      "compat": {
        "supportsReasoningEffort": false,   // top-level reasoning_effort 금지 (400/500 함정)
        "thinkingFormat": "chat-template",
        "chatTemplateKwargs": {
          "enable_thinking": { "$var": "thinking.enabled" },
          "reasoning_effort": { "$var": "thinking.effort", "omitWhenOff": true }
        },
        "thinkingTokenBudgetField": "thinking_token_budget"  // --reasoning-config 로 살아 있다
      },
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 16384,
          // 모델 어휘는 low/medium/xhigh 뿐이다. pi 의 high/max 를 xhigh 로 접는다.
          "thinkingLevelMap": {
            "minimal": "low", "low": "low", "medium": "medium",
            "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"
          },
          "samplingParams": { "temperature": 1.0, "top_p": 0.95, "top_k": 20 }
        }
      ]
    }
  }
}
```

### opencode

모델의 `options` 가 `@ai-sdk/openai-compatible` 의 providerOptions 로 들어가 **요청 본문에 그대로
병합**된다(`chat_template_kwargs` 같은 임의 키도 통과). `variants` 는 id 를 키로 한 객체이고,
값이 그 변형의 `options` 로 덮어써진다 - 사용자가 TUI 에서 모델 변형을 고르듯 강도를 고른다.
`interleaved.field` 를 `reasoning_content` 로 두면 `--reasoning-parser qwen3` 이 분리한 사고가
TUI 의 thinking 블록으로 보인다.

**`reasoningEffort` 옵션은 쓰지 말 것** - 그게 top-level 필드로 나가는 경로다.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM (사내 H200)",
      "options": {
        "baseURL": "http://<사내IP>:8006/v1",
        "apiKey": "{env:VLLM_API_KEY}"
      },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B",
          "reasoning": true,
          "tool_call": true,
          "interleaved": { "field": "reasoning_content" },
          "limit": { "context": 262144, "output": 16384 },
          "options": {
            "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": "medium" }
          },
          "variants": {
            "low":     { "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": "low" } },
            "xhigh":   { "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": "xhigh" } },
            "nothink": { "chat_template_kwargs": { "enable_thinking": false } }
          }
        }
      }
    }
  }
}
```

버전 주의: 위 키 이름은 opencode **1.18.x** 의 `https://opencode.ai/config.json` 스키마로 확인한
것이다(최상위 `provider`, 모델 `options`/`variants`, `variants` 는 배열이 아니라 객체). 키 이름이
틀리면 오류 없이 조용히 무시되고 그냥 기본값(medium)으로 돈다 - 붙여넣기 전에
`opencode --version` 을 보고, 의심되면 `curl -s https://opencode.ai/config.json` 로 대조한다.

### 확인

`--api-key` 를 켠 뒤에는 이 디렉토리의 스크립트도 인증이 필요하다. `qwen_client.py` 의 `API_KEY`
상수가 `VLM_SERVE_UPSTREAM_API_KEY` 를 읽으므로 셸에 export 하거나 상수를 직접 채운다.

```bash
export VLM_SERVE_UPSTREAM_API_KEY=...   # site.env 와 같은 값

# 서버에서: 이 vLLM 빌드에 인자가 있는지 (없으면 기동이 바로 죽는다 - 로그에 unrecognized arguments)
vllm serve --help | grep default-chat-template-kwargs

# 사내 어디서든: 인증과 도달 확인
curl -H "Authorization: Bearer $VLLM_API_KEY" http://<사내IP>:8006/v1/models

# 기본 강도가 medium 으로 깔렸는지 - 사고 토큰 길이로 갈린다
python scripts/effort_ladder.py
```
