"""프롬프트 하나를 스트리밍으로 보내 사고 과정과 답이 나오는 것을 실시간으로 본다.

effort 를 바꿔가며 같은 질문을 던져 보면 사고의 길이와 결의 차이가 눈에 들어온다.
TTFT(첫 토큰까지)와 "생각이 끝나고 답이 시작되는 시각" 을 따로 찍는다 - 사용자가 체감하는
지연은 후자다. 사고는 회색, 답은 기본색으로 찍는다.

사용법:
  python scripts/stream_thinking.py
"""

import sys
import time

from qwen_client import stream_chat

# ── 인자는 여기 있다 ───────────────────────────────────────────────────────
PROMPT = (
    "반도체 팹에서 야간 교대 중 챔버 압력이 서서히 오르는 현상을 관찰했다. "
    "가능한 원인 세 가지를 확률 순으로 들고, 각각을 10분 안에 배제할 수 있는 점검 방법을 제시하라."
)
THINKING = True
EFFORT = "medium"   # low | medium | xhigh
BUDGET = None       # 예: 2048. 서버에 --reasoning-config 가 있어야 먹는다 (thinking_budget.py 참고)
MAX_TOKENS = 8192

_DIM, _RESET = "\033[2m", "\033[0m"


def main():
    print(f"[INFO] thinking={THINKING} effort={EFFORT} budget={BUDGET}\n")
    started = time.monotonic()
    first_token_at = answer_started_at = None
    mode = None
    for kind, text in stream_chat([{"role": "user", "content": PROMPT}],
                                  thinking=THINKING, effort=EFFORT, budget=BUDGET, max_tokens=MAX_TOKENS):
        now = time.monotonic()
        first_token_at = first_token_at or now
        if kind != mode:
            if kind == "reasoning":
                sys.stdout.write(f"{_DIM}--- thinking ---\n")
            else:
                answer_started_at = answer_started_at or now
                sys.stdout.write(f"{_RESET}\n--- answer ---\n")
            mode = kind
        sys.stdout.write(text)
        sys.stdout.flush()
    sys.stdout.write(f"{_RESET}\n")
    total = time.monotonic() - started
    print(f"\n[INFO] first token {((first_token_at or started) - started):.1f}s"
          f" | answer started {((answer_started_at or started) - started):.1f}s | total {total:.1f}s")


if __name__ == "__main__":
    main()
