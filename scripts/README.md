# scripts/ - qwen3.8-27b 를 끝까지 써 보기 (thinking · effort · budget)

서버(vLLM 0.19.1, `--reasoning-parser qwen3`)에서 **검증된 조작법**만 적는다. 근거는
vLLM v0.19.1 소스와 `Qwen/Qwen3.8-27B` 의 `chat_template.jinja` 다. 스크립트는 전부
stdlib 만 쓰고 `qwen_client.py` 하나를 공유한다. CLI 인자는 없다 - 상수 블록을 고친다.

| 스크립트 | 무엇을 재나 | 소요 |
|---|---|---|
| `effort_ladder.py` | 과제 7개 × (off / low / medium / xhigh) - 정확도·사고 토큰·시간 | 5-15분 |
| `stream_thinking.py` | 프롬프트 하나를 스트리밍 - 사고와 답이 갈리는 순간, TTFT | 1분 |
| `thinking_budget.py` | `thinking_token_budget` 이 먹는지, 어디서 품질이 꺾이는지 | 5분 |

```bash
python scripts/effort_ladder.py     # 먼저 이걸로 "어느 단계가 기본값이면 되는지" 정한다
python scripts/stream_thinking.py
python scripts/thinking_budget.py
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

기본 **xhigh**. 이건 상한이 아니라 **프롬프트 문구**다 - template 이 시스템 프롬프트에
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
- **프록시 대신 vLLM 에 직접 붙는다**(`127.0.0.1:8006`). `/api/vlm_serve/...` 는 응답을
  끝까지 버퍼링하고 read timeout 이 300s 라, xhigh 로 몇 분씩 생각하는 요청이 HTTP 경로
  때문에 끊긴다. 프록시로 가려면 `BASE_URL` 을 `http://<flask>/api/vlm_serve/qwen3.8-27b`
  로, 토큰이 있으면 `TOKEN` 을 채운다.
- **tool calling** 을 켜려면 `EXTRA_VLLM_ARGS` 에 `--enable-auto-tool-choice
  --tool-call-parser qwen3_coder` 를 붙인다(vLLM 공식 레시피, 0.19.1 에 파서 있음).
  template 의 tool 포맷이 `<tool_call><function=…><parameter=…>` 라 `qwen3_coder` 가 맞다.
- **1M 컨텍스트**는 `--max-model-len 1010000 --hf-overrides '{"text_config":
  {"max_position_embeddings": 1010000}}'` 로 열린다. fp8 KV 기준 시퀀스당 32GiB 라
  `MAX_NUM_SEQS` 를 2 이하로 내려야 하고, `check_kv_longctx.py` 를 그 길이로 다시 돌려야
  한다. 필요가 생기기 전엔 262144 로 둔다.
