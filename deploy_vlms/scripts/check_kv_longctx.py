"""fp8 KV cache 가 긴 컨텍스트에서 회수 능력을 잃는지 확인한다 (needle-in-a-haystack).

왜 필요한가:
  vLLM 2026-04-22 분석에 따르면 Hopper(H200 포함) 의 FP8 Tensor Core 는 contraction
  차원이 커지면(= 컨텍스트가 길어지면) FP32 누산 정밀도를 잃는다. 128k needle 과제에서
  BF16 91% -> fp8 KV 13% 로 무너진 사례가 보고됐고, two-level accumulation
  (flash-attention#104, +#96/#91) 으로 89% 까지 복구되어 mainline 에 들어갔다.

  문제는 그 수정이 **이 서버의 vLLM 0.19.1 에 들어있는지 알 수 없다**는 것이다.
  버전 족보를 파는 것보다 직접 재는 편이 빠르고 확실하다 - 이 스크립트가 그 측정이다.

읽는 법:
  - 전 구간 PASS -> 수정이 들어있다. `--kv-cache-dtype fp8` 을 그대로 두면 된다.
  - 짧은 구간은 PASS 인데 100k 부터 FAIL -> 알려진 그 증상이다. env 를 바꾼다
    (`qwen3.8-27b.env` 의 `--kv-cache-dtype fp8` 제거 + `MAX_NUM_SEQS` 8 -> 4).
    bf16 KV 는 토큰당 64KiB 라 262k 는 4-way 가 상한이다.
  - 전 구간 FAIL -> KV 문제가 아니라 프롬프트/모델 문제다. 짧은 길이부터 다시 볼 것.

주의:
  - vLLM 에 **직접** 붙는다 (127.0.0.1:8006). Flask proxy 와 nginx 를 우회하는데,
    그 둘의 timeout(300s) 과 body size 상한이 측정에 섞이지 않게 하려는 것이다.
    측정 대상은 모델이지 HTTP 경로가 아니다.
  - 200k 프롬프트 하나의 prefill 이 ~40초다. 기본 설정 12회면 5-10분 걸린다.
  - 다른 사람이 그 인스턴스를 쓰는 중이면 큐가 밀린다. 한산할 때 돌릴 것.

사용법:
  uv run python deploy_vlms/scripts/check_kv_longctx.py
"""

import json
import random
import sys
import urllib.error
import urllib.request


# ── 인자는 여기 있다 (이 저장소는 CLI 플래그를 쓰지 않는다) ────────────────
BASE_URL = "http://127.0.0.1:8006"
MODEL = "qwen3.8-27b"

# 잴 컨텍스트 길이(토큰). MAX_MODEL_LEN=262144 보다 넉넉히 아래로 둔다.
# 8000 은 대조군이다 - 여기까지 틀리면 KV 정밀도 문제가 아니라 프롬프트 문제다.
LENGTHS = [8000, 64000, 128000, 200000]

# needle 을 문서의 어디에 심을지 (0.0=맨 앞, 1.0=맨 뒤).
# 중간 깊이가 가장 잘 틀린다 ("lost in the middle").
DEPTHS = [0.1, 0.5, 0.9]

REQUEST_TIMEOUT_SEC = 900.0
SEED = 20260904
PASS_THRESHOLD = 0.9  # 이 비율 미만이면 그 길이는 FAIL 로 본다


_NEEDLE_TEMPLATE = "야간 정비 승인 코드는 {code} 이다."
_QUESTION = (
    "위 문서에서 야간 정비 승인 코드를 찾아라. "
    "설명 없이 숫자 6자리만 출력하라."
)


def build_filler_line(index):
    """압축되지 않는 채움 문장 한 줄. 줄마다 값이 달라야 한다.

    같은 문장을 반복하면 모델이 내용을 안 읽고도 답할 수 있고, prefix cache 가
    통째로 히트해서 정작 재려던 긴 KV 를 안 만든다.
    """
    return (
        f"{index:06d}번 설비 점검 기록. 챔버 압력 {index % 997}Pa, "
        f"스테이지 온도 {20 + index % 15}도, 담당자 교대 {index % 3}조."
    )


def build_haystack(line_count, needle, depth):
    """채움 문장 사이에 needle 을 depth 위치에 끼운 문서를 만든다."""
    lines = [build_filler_line(i) for i in range(line_count)]
    position = min(line_count, max(0, int(line_count * depth)))
    lines.insert(position, needle)
    return "\n".join(lines)


def count_tokens(text):
    """vLLM /tokenize 로 실제 토큰 수를 센다. 실패하면 None."""
    payload = {"model": MODEL, "prompt": text}
    try:
        body = post_json("/tokenize", payload)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    return body.get("count")


def post_json(path, payload):
    request = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(document):
    """문서를 주고 코드를 묻는다. 온도 0, 출력은 짧게."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": f"{document}\n\n{_QUESTION}"},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
        # thinking 토큰이 붙으면 답만 뽑기 번거로워진다. 이 과제는 추론이 필요 없다.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = post_json("/v1/chat/completions", payload)
    return body["choices"][0]["message"]["content"]


def lines_for_target_tokens(target_tokens):
    """목표 토큰 수에 맞는 줄 수를 추정한다.

    한 번 재고 한 번 보정한다. 정확히 맞출 필요는 없다 - 실제 토큰 수를
    측정해서 보고하므로, 목표에 근접하기만 하면 된다.
    """
    probe_lines = 200
    probe_text = "\n".join(build_filler_line(i) for i in range(probe_lines))
    measured = count_tokens(probe_text)
    if not measured:
        # /tokenize 가 없으면 한글 섞인 문장 기준 대략치로 간다.
        tokens_per_line = 34.0
    else:
        tokens_per_line = measured / probe_lines
    return max(1, int(target_tokens / tokens_per_line))


def run_one(target_tokens, depth, code):
    needle = _NEEDLE_TEMPLATE.format(code=code)
    document = build_haystack(lines_for_target_tokens(target_tokens), needle, depth)
    actual = count_tokens(document)
    answer = ask(document)
    return code in answer, actual, answer.strip().replace("\n", " ")[:60]


def self_check():
    """서버 없이 needle 삽입/판정 로직만 검증한다 (Mac 에서도 돈다)."""
    needle = _NEEDLE_TEMPLATE.format(code="123456")
    for depth in (0.0, 0.5, 1.0):
        document = build_haystack(100, needle, depth)
        assert needle in document, depth
        assert document.count(needle) == 1, depth
        assert len(document.splitlines()) == 101, depth
    # 채움 줄은 서로 달라야 한다 (같으면 needle 없이도 맞힐 수 있다).
    filler = [build_filler_line(i) for i in range(100)]
    assert len(set(filler)) == 100
    # needle 이 채움 문장과 섞이지 않아야 한다.
    assert not any(_NEEDLE_TEMPLATE.split("{")[0] in line for line in filler)
    print("[INFO] self-check OK - needle 삽입/유일성/채움 다양성 정상")


LONG_CONTEXT_FLOOR = 100000  # 이 이상을 "긴 컨텍스트" 로 본다 (Hopper 증상 발현 구간)


def classify_verdicts(verdicts, lengths):
    """길이별 통과율에서 결론 하나를 뽑는다.

    짧은 구간과 긴 구간을 갈라 보는 것이 요점이다. 둘 다 틀리면 KV 정밀도가
    원인이 아니므로(대조군이 이미 틀렸다) 다른 진단으로 보내야 한다.
    """
    short = [t for t in lengths if t < LONG_CONTEXT_FLOOR]
    long_ = [t for t in lengths if t >= LONG_CONTEXT_FLOOR]
    short_ok = all(verdicts.get(t, 0.0) >= PASS_THRESHOLD for t in short)
    long_ok = all(verdicts.get(t, 0.0) >= PASS_THRESHOLD for t in long_)
    if short_ok and long_ok:
        return "ok"
    if short_ok:
        return "long_ctx_fail"
    return "broken"


VERDICT_MESSAGES = {
    "ok": [
        "[INFO] 전 구간 PASS - fp8 KV 를 그대로 두면 된다.",
        "       (two-level accumulation 수정이 이 vLLM 에 들어있다는 뜻이다)",
    ],
    "long_ctx_fail": [
        "[ERROR] 짧은 구간은 되는데 100k 이상에서 무너진다.",
        "        알려진 Hopper fp8 KV 누산 정밀도 증상이다. qwen3.8-27b.env 를 고칠 것:",
        "          EXTRA_VLLM_ARGS 에서 '--kv-cache-dtype fp8' 제거",
        "          MAX_NUM_SEQS=8 -> 4   (bf16 KV 는 64KiB/token 이라 262k 는 4-way 가 상한)",
    ],
    "broken": [
        "[WARNING] 짧은 구간부터 틀린다. KV 정밀도 문제가 아니라 프롬프트/모델 쪽이다.",
        "          LENGTHS 를 [2000, 8000] 으로 줄여 먼저 재현할 것.",
    ],
}


def main():
    random.seed(SEED)

    if count_tokens("ping") is None:
        print(f"[WARNING] {BASE_URL} 에 붙지 못했다. GPU 서버에서 실행할 것.")
        print("          (인스턴스가 떠 있는지: uv run python deploy_vlms/scripts/check_vlm.py)")
        self_check()
        return

    print(f"[INFO] target={BASE_URL} model={MODEL}")
    print(f"[INFO] lengths={LENGTHS} depths={DEPTHS}")
    print()

    header = f"  {'목표토큰':>9} {'실측토큰':>9}  " + "  ".join(f"d={d:<4}" for d in DEPTHS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    verdicts = {}
    for target in LENGTHS:
        marks = []
        hits = 0
        measured = None
        for depth in DEPTHS:
            code = f"{random.randint(100000, 999999)}"
            try:
                ok, measured, answer = run_one(target, depth, code)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                print(f"\n[ERROR] target={target} depth={depth} 요청 실패: {exc}")
                marks.append("ERR ")
                continue
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                print(f"\n[ERROR] target={target} depth={depth} 응답 파싱 실패: {exc}")
                marks.append("ERR ")
                continue
            hits += 1 if ok else 0
            marks.append("PASS" if ok else "FAIL")
            if not ok:
                print(f"        [miss] target={target} depth={depth} 기대={code} 응답='{answer}'")
        verdicts[target] = hits / len(DEPTHS)
        shown = f"{measured:,}" if measured else "?"
        print(f"  {target:>9,} {shown:>9}  " + "  ".join(f"{m:<6}" for m in marks))

    print()
    verdict = classify_verdicts(verdicts, LENGTHS)
    for line in VERDICT_MESSAGES[verdict]:
        print(line)
    sys.exit(0 if verdict == "ok" else 1)


if __name__ == "__main__":
    main()
