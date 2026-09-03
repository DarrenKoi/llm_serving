"""호스트 RAM 실사용량 보고 - 16GB 천장에서 몇 개까지 띄울 수 있는지 판단용.

`ps` 의 RSS 는 **못 쓴다**. 가중치를 mmap 으로 읽으므로(safetensors lazy) 파일 캐시가
RSS 에 잡혀 실제보다 훨씬 크게 보이고, 그 페이지는 커널이 언제든 회수할 수 있다.
그래서 여기서는 `/proc/<pid>/smaps_rollup` 의 세 값을 갈라서 본다:

  Pss       공유 페이지를 프로세스 수로 나눠 더한 값. **합계를 낼 때 이걸 쓴다**
            (RSS 를 그냥 더하면 공유 라이브러리를 인스턴스 수만큼 중복 계산한다).
  Pss_Anon  익명 메모리의 PSS 몫 = 회수 불가. **위험한 쪽**.
  Pss_File  파일 backed 의 PSS 몫 = 가중치 mmap/라이브러리. 압박이 오면 커널이 회수한다.

주의: `smaps_rollup` 에는 `RssAnon`/`RssFile` 이 **없다**. 그 이름은 `/proc/<pid>/status`
쪽이고, rollup 은 `Pss_Anon`/`Pss_File`/`Pss_Shmem`(+`Anonymous`)을 낸다. 없는 키를
읽으면 조용히 0 이 나와서 "회수 불가 = 0" 이라는 안전해 보이는 거짓말이 찍힌다.
합계에 Pss 를 쓰므로 분해도 Pss_* 로 맞추는 편이 일관된다(서로 더해서 Pss 가 된다).

판단 기준은 `MemAvailable` 이다 (free 의 "free" 가 아니다 - 회수 가능한 캐시를 뺀
값이라 커널이 직접 계산해준 '실제로 더 쓸 수 있는 양'이다).

사용법 (모델을 다 띄운 뒤 warm 상태에서):
  uv run python deploy_vlms/scripts/check_host_ram.py
"""

import os
import sys
from pathlib import Path


# MemAvailable 이 이 아래로 떨어지면 OOM killer 사정권으로 본다.
LOW_AVAILABLE_MIB = 2048
KIB = 1024.0


def read_meminfo() -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            values[key.strip()] = float(parts[0]) / KIB  # MiB
    return values


def parse_smaps_rollup(text: str) -> dict:
    """smaps_rollup 본문을 {키: MiB} 로 파싱한다 (/proc 없이도 시험 가능하게 분리)."""
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key.strip()] = float(parts[0]) / KIB  # MiB
    return values


def read_smaps_rollup(pid: int) -> dict:
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return {}
    return parse_smaps_rollup(text)


def read_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").replace("\0", " ").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def label_for(cmdline: str) -> str:
    """cmdline 에서 어떤 모델인지 뽑는다. 자식(EngineCore)은 인자가 없을 수 있다."""
    parts = cmdline.split()
    if "--served-model-name" in parts:
        idx = parts.index("--served-model-name")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "VLLM::EngineCore" in cmdline or "EngineCore" in cmdline:
        return "(EngineCore)"
    return "(vllm child)"


def find_vllm_pids() -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmdline = read_cmdline(pid)
        if not cmdline:
            continue
        if "vllm" in cmdline.lower() or "EngineCore" in cmdline:
            pids.append(pid)
    return sorted(pids)


# 실제 커널 출력에서 잘라온 표본. 여기 없는 키를 읽으면 조용히 0 이 되므로,
# 필드명이 바뀌거나 오타가 나면 이 점검이 먼저 깨져야 한다.
_SAMPLE_ROLLUP = """\
Rss:             1234567 kB
Pss:              987654 kB
Pss_Anon:         654321 kB
Pss_File:         333333 kB
Pss_Shmem:             0 kB
Anonymous:        700000 kB
Swap:                  0 kB
"""


def self_check() -> None:
    """/proc 이 없는 개발 PC 에서 파서만 검증한다."""
    roll = parse_smaps_rollup(_SAMPLE_ROLLUP)
    assert abs(roll["Pss"] - 987654 / KIB) < 1e-6, roll
    # 회귀 방지의 핵심: smaps_rollup 에 RssAnon/RssFile 은 존재하지 않는다.
    # 예전 코드가 그 이름을 읽어서 anon 이 항상 0 으로 찍혔다.
    assert "RssAnon" not in roll and "Rss_Anon" not in roll, roll
    assert roll["Pss_Anon"] > 0 and roll["Pss_File"] > 0, roll
    anon = roll.get("Pss_Anon", 0.0)
    rfile = roll.get("Pss_File", 0.0) + roll.get("Pss_Shmem", 0.0)
    assert abs((anon + rfile) - roll["Pss"]) < 1.0, (anon, rfile, roll["Pss"])
    print("[INFO] self-check OK - smaps_rollup 파서 정상 (Pss_Anon/Pss_File)")


def main() -> None:
    if not Path("/proc/meminfo").is_file():
        print("[WARNING] /proc 이 없다. 이 스크립트의 보고 기능은 Linux 전용이다"
              " (GPU 서버에서 실행할 것). 파서 자체 점검만 돌린다.")
        self_check()
        return

    mem = read_meminfo()
    total = mem.get("MemTotal", 0.0)
    available = mem.get("MemAvailable", 0.0)
    swap_total = mem.get("SwapTotal", 0.0)
    swap_free = mem.get("SwapFree", 0.0)

    pids = find_vllm_pids()
    if not pids:
        print("[WARNING] vLLM 프로세스를 찾지 못했다. 모델이 떠 있는 상태에서 실행할 것.")

    print(f"[INFO] MemTotal={total:,.0f} MiB  MemAvailable={available:,.0f} MiB")
    if swap_total > 0:
        print(f"[INFO] Swap={swap_free:,.0f} / {swap_total:,.0f} MiB free")
    else:
        print("[WARNING] swap 이 없다. 일시적 스파이크가 곧바로 OOM kill 이 된다.")
        print("          16GB 고정 호스트라면 swap 파일 8~16GB 를 두는 것이 가장 싼 보험이다.")
    print()

    header = f"  {'PID':>7}  {'모델':<20} {'Pss':>10} {'Pss_Anon':>10} {'Pss_File':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_pss = 0.0
    total_anon = 0.0
    for pid in pids:
        roll = read_smaps_rollup(pid)
        if not roll:
            continue
        pss = roll.get("Pss", 0.0)
        anon = roll.get("Pss_Anon", 0.0)
        rfile = roll.get("Pss_File", 0.0) + roll.get("Pss_Shmem", 0.0)
        total_pss += pss
        total_anon += anon
        print(f"  {pid:>7}  {label_for(read_cmdline(pid)):<20} {pss:>9,.0f}M {anon:>9,.0f}M {rfile:>9,.0f}M")

    print("  " + "-" * (len(header) - 2))
    print(f"  {'':>7}  {'합계':<20} {total_pss:>9,.0f}M {total_anon:>9,.0f}M")
    print()
    print(f"[INFO] vLLM 합계 PSS = {total_pss:,.0f} MiB ({total_pss/1024:.1f} GiB), "
          f"그중 회수 불가(anon) = {total_anon:,.0f} MiB ({total_anon/1024:.1f} GiB)")
    print(f"[INFO] 프로세스 수 = {len(pids)} (TP=1 이면 모델당 2개: API + EngineCore)")
    print()

    if available < LOW_AVAILABLE_MIB:
        print(f"[ERROR] MemAvailable 이 {available:,.0f} MiB 로 {LOW_AVAILABLE_MIB} MiB 아래다.")
        print("        이 구성은 유지하면 안 된다. 인스턴스를 하나 내릴 것 -")
        print("        production 루프는 mai-ui + paddleocr 이므로 qwen3.8-27b 를 먼저 내린다.")
        sys.exit(1)
    if available < LOW_AVAILABLE_MIB * 2:
        print(f"[WARNING] MemAvailable {available:,.0f} MiB - 여유가 적다. 요청이 몰리면 위험하다.")
        print("          swap 이 없다면 지금 만들 것. -O1 로 컴파일 단계 부담도 줄일 수 있다.")
        return
    print(f"[INFO] MemAvailable {available:,.0f} MiB - 현재 구성은 여유가 있다.")


if __name__ == "__main__":
    main()
