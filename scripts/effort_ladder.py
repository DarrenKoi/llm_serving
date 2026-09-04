"""같은 과제 묶음을 thinking off / low / medium / xhigh 로 돌려 정확도·토큰·시간을 비교한다.

"이 모델을 어디까지 쓸 수 있나" 를 재는 첫 번째 스크립트다. 결과표를 보고 정한다:
  - off 로도 다 맞는 과제  -> instruct 모드로 쓴다 (가장 싸고 빠르다).
  - low/medium 에서 맞기 시작하는 과제 -> 그 단계가 실사용 기본값이다.
  - xhigh 에서만 맞는 과제 -> 그 종류의 요청만 xhigh 로 올린다. 기본값(xhigh)을 전역으로
    두면 추론 토큰이 답 토큰의 수십 배가 되어 처리량을 갉아먹는다.
  - finish=length 가 보이면 MAX_TOKENS 가 모자란 것이다. 답이 아니라 사고가 잘린 것.

한 단계의 과제들은 동시에 보낸다(MAX_NUM_SEQS=8 이므로 8개까지). 단계는 순서대로 돈다 -
xhigh 가 low 와 섞이면 시간 비교가 흐려진다.

사용법:
  python scripts/effort_ladder.py
"""

import datetime
import re
from concurrent.futures import ThreadPoolExecutor

from qwen_client import chat, count_tokens

# ── 인자는 여기 있다 ───────────────────────────────────────────────────────
LEVELS = [("off", False, "xhigh"), ("low", True, "low"), ("medium", True, "medium"), ("xhigh", True, "xhigh")]
MAX_TOKENS = 16384  # 사고 + 답. xhigh 가 이걸 넘기면 finish=length 로 표에 찍힌다.
MAX_WORKERS = 7     # <= MAX_NUM_SEQS(8). 남는 1 슬롯은 다른 사용자 몫.

_TAIL = "\n\n마지막 줄에 '정답: <값>' 형식으로만 답하라."

# (이름, 질문, 정답에 반드시 들어가야 할 문자열들). 정답은 파이썬이 계산한다 - 손으로 적은
# 기대값이 틀려서 모델을 억울하게 떨어뜨리는 일을 막는다.
_WEEKDAY_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
TASKS = [
    ("compare", "9.11 과 9.8 중 어느 수가 더 큰가?", ["9.8"]),
    ("count-r", "strawberry 라는 단어에 알파벳 r 은 몇 번 나오는가?", ["3"]),
    ("multiply", "347 × 829 를 계산하라.", [str(347 * 829)]),
    ("weekday", "2026-09-04 는 금요일이다. 2027-03-15 는 무슨 요일인가?",
     [_WEEKDAY_KO[datetime.date(2027, 3, 15).weekday()]]),
    ("logic", "민수, 지영, 현우 세 사람은 커피, 녹차, 주스 중 서로 다른 음료를 하나씩 마신다. "
              "민수는 커피를 마시지 않는다. 지영은 녹차도 주스도 마시지 않는다. "
              "현우는 주스를 마시지 않는다. 누가 주스를 마시는가?", ["민수"]),
    ("trace", "다음 파이썬 코드의 출력은?\n\ns = 0\nfor i in range(1, 11):\n    if i % 3 == 0:\n        s += i * i\nprint(s)",
     [str(sum(i * i for i in range(1, 11) if i % 3 == 0))]),
    ("json", "다음 문장에서 이름과 나이를 뽑아 {\"name\": ..., \"age\": ...} JSON 하나만 출력하라: "
             "'김철수 씨는 올해 마흔둘이다.'", ["김철수", "42"]),
]


def normalize(text):
    return re.sub(r"[\s,]", "", text).lower()


def passed(content, expected):
    got = normalize(content)
    return all(normalize(item) in got for item in expected)


def run_task(task, thinking, effort):
    name, question, expected = task
    reply = chat([{"role": "user", "content": question + _TAIL}],
                 thinking=thinking, effort=effort, max_tokens=MAX_TOKENS)
    think_tokens = count_tokens(reply.reasoning) if reply.reasoning else 0
    return {
        "task": name,
        "ok": passed(reply.content, expected),
        "think": think_tokens,
        "answer": reply.completion_tokens - (think_tokens or 0),
        "sec": reply.elapsed_sec,
        "finish": reply.finish_reason,
        "last": reply.content.strip().splitlines()[-1][:40] if reply.content.strip() else "",
    }


def main():
    print(f"[INFO] {len(TASKS)} tasks x {len(LEVELS)} levels, max_tokens={MAX_TOKENS}")
    summary = []
    for label, thinking, effort in LEVELS:
        print(f"\n== {label} (enable_thinking={thinking}, reasoning_effort={effort if thinking else '-'})")
        print(f"{'task':<10}{'ok':<6}{'think':>7}{'answer':>8}{'sec':>8}  finish  last line")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            rows = list(pool.map(lambda t: run_task(t, thinking, effort), TASKS))
        for r in rows:
            print(f"{r['task']:<10}{'PASS' if r['ok'] else 'FAIL':<6}{r['think'] or 0:>7}{r['answer']:>8}"
                  f"{r['sec']:>8.1f}  {r['finish']:<6}  {r['last']}")
        summary.append((label, sum(r["ok"] for r in rows), sum(r["think"] or 0 for r in rows),
                        max(r["sec"] for r in rows)))

    print("\n== summary")
    print(f"{'level':<8}{'pass':>6}{'think tok':>11}{'wall sec':>10}")
    for label, ok, think, sec in summary:
        print(f"{label:<8}{ok:>3}/{len(TASKS):<2}{think:>11}{sec:>10.1f}")


if __name__ == "__main__":
    main()
