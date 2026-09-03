"""check_kv_longctx.py 단위 테스트 (서버/GPU 불필요, Mac 에서 돈다).

네트워크를 타는 부분은 테스트하지 않는다. 재는 값이 서버 상태라 고정할 수 없고,
고정하면 그 순간 측정이 아니라 mock 을 재는 것이 된다. 여기서 지키는 것은 둘이다 -
**needle 이 문서에 정확히 한 번 들어가는가**(안 그러면 측정 자체가 무효),
그리고 **통과율이 올바른 결론으로 번역되는가**(여기가 틀리면 멀쩡한 설정을 고치거나
깨진 설정을 놔둔다).

  uv run pytest deploy_vlms/scripts/test_check_kv_longctx.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_kv_longctx as mod


# ── needle 삽입 ───────────────────────────────────────────────────────────

def test_needle_appears_exactly_once_at_every_depth():
    needle = "야간 정비 승인 코드는 424242 이다."
    for depth in (0.0, 0.1, 0.5, 0.9, 1.0):
        document = mod.build_haystack(50, needle, depth)
        assert document.count(needle) == 1, depth


def test_needle_position_follows_depth():
    needle = "NEEDLE"
    front = mod.build_haystack(100, needle, 0.0).splitlines().index(needle)
    middle = mod.build_haystack(100, needle, 0.5).splitlines().index(needle)
    back = mod.build_haystack(100, needle, 1.0).splitlines().index(needle)
    assert front < middle < back
    assert front == 0
    assert back == 100  # 채움 100줄 뒤


def test_depth_out_of_range_is_clamped_not_crashed():
    """depth 를 잘못 줘도 needle 은 문서 안에 남아야 한다."""
    for depth in (-1.0, 2.0):
        document = mod.build_haystack(10, "NEEDLE", depth)
        assert document.count("NEEDLE") == 1, depth


def test_filler_lines_are_all_distinct():
    """같은 문장을 반복하면 모델이 안 읽고도 맞힐 수 있고 prefix cache 가
    통째로 히트해서 정작 재려던 긴 KV 가 안 만들어진다."""
    lines = [mod.build_filler_line(i) for i in range(500)]
    assert len(set(lines)) == 500


def test_filler_never_contains_the_needle_phrase():
    """채움 문장이 needle 문구를 포함하면 오답이 정답으로 셈해진다."""
    prefix = mod._NEEDLE_TEMPLATE.split("{")[0]
    assert all(prefix not in mod.build_filler_line(i) for i in range(500))


def test_line_count_matches_request():
    document = mod.build_haystack(37, "NEEDLE", 0.5)
    assert len(document.splitlines()) == 38  # 채움 37 + needle 1


# ── 판정 분기 ─────────────────────────────────────────────────────────────

LENGTHS = [8000, 64000, 128000, 200000]


def test_all_pass_means_leave_config_alone():
    verdicts = {t: 1.0 for t in LENGTHS}
    assert mod.classify_verdicts(verdicts, LENGTHS) == "ok"


def test_long_context_only_failure_is_the_hopper_symptom():
    verdicts = {8000: 1.0, 64000: 1.0, 128000: 0.0, 200000: 0.0}
    assert mod.classify_verdicts(verdicts, LENGTHS) == "long_ctx_fail"


def test_short_context_failure_is_not_a_kv_verdict():
    """대조군(8k)이 이미 틀렸으면 KV 정밀도로 결론내면 안 된다.
    긴 구간까지 같이 틀려도 마찬가지다 - 원인이 다른 곳에 있다."""
    verdicts = {8000: 0.0, 64000: 1.0, 128000: 0.0, 200000: 0.0}
    assert mod.classify_verdicts(verdicts, LENGTHS) == "broken"
    assert mod.classify_verdicts({t: 0.0 for t in LENGTHS}, LENGTHS) == "broken"


def test_partial_pass_below_threshold_counts_as_failure():
    """3회 중 2회만 맞는 것은 통과가 아니다 (PASS_THRESHOLD=0.9)."""
    verdicts = {8000: 1.0, 64000: 1.0, 128000: 2 / 3, 200000: 1.0}
    assert mod.classify_verdicts(verdicts, LENGTHS) == "long_ctx_fail"


def test_missing_length_is_treated_as_failure_not_success():
    """요청이 전부 ERR 나서 기록이 없는 길이를 통과로 세면 안 된다."""
    assert mod.classify_verdicts({8000: 1.0, 64000: 1.0}, LENGTHS) == "long_ctx_fail"
    assert mod.classify_verdicts({}, LENGTHS) == "broken"


def test_every_verdict_has_a_message():
    """분기를 늘리고 메시지를 안 적으면 KeyError 로 죽는다."""
    for verdict in ("ok", "long_ctx_fail", "broken"):
        assert mod.VERDICT_MESSAGES[verdict]


def test_long_context_floor_sits_where_the_symptom_starts():
    """보고된 발현 구간(~100k)과 LENGTHS 가 어긋나면 분기가 무의미해진다 -
    긴 구간이 비면 all() 이 공집합에 True 를 주어 항상 ok 가 된다."""
    assert any(t >= mod.LONG_CONTEXT_FLOOR for t in mod.LENGTHS)
    assert any(t < mod.LONG_CONTEXT_FLOOR for t in mod.LENGTHS)


# ── 설정 정합 ─────────────────────────────────────────────────────────────

def test_lengths_stay_inside_max_model_len():
    """MAX_MODEL_LEN=262144 를 넘기면 측정이 아니라 400 에러를 재게 된다.
    출력 토큰과 질문 여유를 남긴다."""
    assert max(mod.LENGTHS) <= 250000


def test_self_check_runs_clean():
    mod.self_check()
