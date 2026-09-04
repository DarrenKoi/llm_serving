"""thinking_token_budget 이 이 서버에서 실제로 먹는지, 먹으면 품질이 어디서 꺾이는지 잰다.

reasoning_effort 는 모델에게 "짧게 생각하라" 고 **부탁**하는 프롬프트 문구다. 상한이 아니다.
진짜 상한은 thinking_token_budget 인데, vLLM 이 budget 에 닿으면 </think> 를 강제로 넣는다.
단, 서버가 사고 경계 토큰을 알아야 하므로 qwen3.8-27b.env 의 EXTRA_VLLM_ARGS 끝에
다음이 있어야 한다:

  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}'

없으면 요청은 200 으로 정상 응답하지만 budget 은 조용히 무시된다. 이 스크립트는
그 "무시" 를 실측 사고 토큰 수 > budget 으로 잡아낸다.

읽는 법:
  - budget 열보다 think 열이 뚜렷이 크면(1.2배 초과) -> 서버에 --reasoning-config 가 없다.
  - think 가 budget 근처에서 잘리고 답이 틀리면 -> 그 과제엔 그 budget 이 모자란 것.
  - none 행은 대조군(무제한)이다.

사용법:
  python scripts/thinking_budget.py
"""

from qwen_client import chat, count_tokens

# ── 인자는 여기 있다 ───────────────────────────────────────────────────────
BUDGETS = [128, 512, 2048, None]
EFFORT = "xhigh"
MAX_TOKENS = 16384
QUESTION = (
    "1부터 100까지의 자연수 중 3 또는 5 의 배수이면서 7 의 배수가 아닌 수의 합을 구하라."
    "\n\n마지막 줄에 '정답: <값>' 형식으로만 답하라."
)
EXPECTED = str(sum(n for n in range(1, 101) if (n % 3 == 0 or n % 5 == 0) and n % 7 != 0))
SLACK = 1.2  # 강제 종료 문구/토크나이저 오차 허용


def main():
    print(f"[INFO] effort={EFFORT} expected={EXPECTED}")
    print(f"{'budget':>8}{'think':>8}{'answer':>8}{'sec':>8}  ok    finish  last line")
    ignored = False
    for budget in BUDGETS:
        reply = chat([{"role": "user", "content": QUESTION}], thinking=True, effort=EFFORT,
                     budget=budget, max_tokens=MAX_TOKENS)
        think = count_tokens(reply.reasoning) or 0
        answer_tokens = reply.completion_tokens - think
        ok = EXPECTED in reply.content.replace(",", "")
        last = reply.content.strip().splitlines()[-1][:40] if reply.content.strip() else ""
        print(f"{str(budget):>8}{think:>8}{answer_tokens:>8}{reply.elapsed_sec:>8.1f}  "
              f"{'PASS' if ok else 'FAIL':<5} {reply.finish_reason:<6}  {last}")
        if budget is not None and think > budget * SLACK:
            ignored = True
    if ignored:
        print("[WARNING] budget 을 넘겨 생각했다 - 서버에 --reasoning-config 가 없다. "
              "qwen3.8-27b.env 의 EXTRA_VLLM_ARGS 에 위 docstring 의 인자를 붙이고 재기동할 것.")


if __name__ == "__main__":
    main()
