# Serving Knobs: 개념 정리

> 작성 2026-09-04. `deploy_vlms/config/` 의 값들이 **무엇을 하는 knob 인지** 를 설명하는
> 개념 문서다. 절차는 `01`~`04`, 실제 값은 env 파일이 정본이고, 이 문서는 그 값을
> 읽을 수 있게 하는 배경이다.

---

## 0. 하나의 그림: GPU 메모리는 세 덩어리다

거의 모든 서빙 knob 은 결국 이 셋 중 하나를 건드린다. 어떤 설정을 만나든
"이건 셋 중 어디를 움직이나" 를 먼저 물으면 대부분 정리된다.

```
GPU 한 장 (H200 140GiB)
├─ Weights        모델 파라미터. 로드하면 끝. 길이/동시성과 무관하게 고정.
├─ KV cache       토큰마다 쌓이는 attention 상태. 길이 x 동시성에 비례해서 커진다.
└─ Activations    지금 계산 중인 중간값 + CUDA graph. 배치 크기에 비례.
```

`GPU_MEMORY_UTILIZATION` 은 이 셋 전부가 들어갈 **총 예산**을 정한다. vLLM 은
weights 를 먼저 올린 다음 activation 을 재보고 **남는 전부를 KV pool 로 잡는다.**
그래서 `u` 를 올리면 실질적으로 늘어나는 것은 KV pool 이다.

`qwen3.8-27b` 기준 실제 배분:

```
140GiB x 0.90 = 126GiB
  - BF16 weights          ~48GiB
  - activations/graph      ~6GiB
  - GDN recurrent state    ~1GiB   (아래 3절)
  ────────────────────────────────
  = KV pool               ~71GiB
```

---

## 1. 숫자 형식 (BF16 / FP16 / FP8)

모델 안의 모든 수는 어떤 형식으로 저장된다. 형식은 정해진 비트 예산을
**범위(range)** 와 **정밀도(precision)** 로 나눠 갖는다.

| 형식 | 비트 | 부호/지수/가수 | 최대값 | 상대 정밀도 |
|---|---|---|---|---|
| FP32 | 32 | 1 / 8 / 23 | ~3.4e38 | ~0.00001% |
| **BF16** | 16 | 1 / **8** / 7 | ~3.4e38 | ~0.4% |
| FP16 | 16 | 1 / 5 / 10 | 65,504 | ~0.05% |
| **FP8 (E4M3)** | 8 | 1 / 4 / 3 | 448 | ~6% |
| FP8 (E5M2) | 8 | 1 / 5 / 2 | 57,344 | ~12% |

BF16 의 요점은 **FP32 와 같은 8비트 지수**를 유지하고 가수만 깎았다는 것이다.
범위가 같으니 FP32 -> BF16 변환은 아래 16비트를 잘라내는 것과 같고 오버플로가
원리적으로 없다. FP16 은 반대 선택(정밀도 우선, 범위 희생)이라 학습에서
loss scaling 같은 보정이 필요했다. 요즘 가중치 기본값은 BF16 이다.

FP8 E4M3 는 최대값이 448 밖에 안 되므로 **그냥 저장할 수 없다** - 텐서/블록마다
scaling factor 를 함께 들고 다닌다. H200 은 Hopper 라 FP8 이 네이티브다
(NVFP4 는 Blackwell 전용이라 여기선 선택지가 아니다).

### 중요: 형식은 "덩어리별로" 따로 정한다

`qwen3.8-27b.env` 에 BF16 과 FP8 이 **동시에** 나오는 것은 모순이 아니다.
서로 다른 메모리 덩어리에 붙은 별개의 dial 이다.

```
common.env:               DTYPE=bfloat16          <- Weights
qwen3.8-27b.env:  ... --kv-cache-dtype fp8        <- KV cache
```

같은 "비트를 절반으로" 라는 동작인데 사는 것이 다르다:

| 어디에 적용 | 무엇을 사나 | 왜 |
|---|---|---|
| **Weights** 를 FP8 로 | **속도** (단일 스트림 ~2배) | decode 는 대역폭 바운드다. 토큰 하나 만들 때마다 가중치 전체를 HBM 에서 읽으므로 바이트가 절반이면 대략 2배 빠르다. |
| **KV** 를 FP8 로 | **길이/동시성** (2배) | KV 는 길이에 비례해 커지는 유일한 덩어리다. 토큰당 바이트가 절반이면 2배 담긴다. |

위험도도 다르다. Qwen 공식 FP8 빌드는 **선택적 양자화**라서 MLP 만 FP8 이고
attention / SSM / vision tower / lm_head 는 BF16 그대로다 - 품질 영향이 작다.
반면 KV 양자화는 attention 이 실제로 뒤지는 key/value 를 값당 ~6% 오차로
뭉갠다. 평균적으로는 상쇄되지만 **"80만 토큰 앞의 그 문장 찾기"** 같은 정밀
회수에서는 상쇄되지 않는다. 긴 컨텍스트를 여는 목적과 가장 잘 충돌하는 knob 이
KV 양자화인 이유다.

### 실측 근거: FP8 은 실제로 얼마나 손해인가

"평균은 멀쩡한데 특정 도메인이 빠진다" 가 정확한 요약이다. Qwen3-32B 도메인별 측정:

| 도메인 | BF16 | FP8 | 차이 |
|---|---|---|---|
| Math | 81.87% | 81.87% | 0 |
| Law | 43.05% | 40.60% | -2.5pt |
| **Engineering** | 49.64% | **43.45%** | **-6.2pt** |

수학은 그대로인데 engineering 이 6.2점 빠진다. **코딩 용도라면 이 칸이 가장 관련 있다** -
PPL 이나 종합 점수만 보고 판단하면 이 손실이 안 보인다.

함께 확인된 것들:

- **작은 모델일수록 민감하다.** 7B 미만이 특히 그렇고 FP8 이 INT8 대비 우위를 갖는 것도
  6.7B 이상부터다. 27B 는 중간 지대다.
- **어려운 추론에서 갈린다.** prefill 과 decode 를 **둘 다** FP8 로 하면 AIME 급 난문에서
  정확도가 떨어지고 decode 를 BF16 으로 남기면 보존된다.
- **구현이 전부다.** naive FP8 rollout 은 Qwen 계열에서 눈에 띄게 나빠질 수 있다는 보고가
  있는데, Qwen 공식 FP8 은 선택적 양자화(MLP 만)라 위 연구가 말하는 "민감한 부분은 full
  precision 유지" 패턴에 부합한다. 공식 빌드를 쓰는 한 리스크는 일반적인 W8A8 보고보다 낮다.

### fp8 KV 의 알려진 실패 모드 (Hopper, 미확인)

vLLM 2026-04-22 분석:

> **128k needle-in-a-haystack: BF16 91% -> fp8 KV 13%**

원인이 모델이 아니라 **하드웨어**다. Hopper 의 FP8 Tensor Core 는 contraction 차원이
커지면(= 컨텍스트가 길어지면) FP32 누산 정밀도를 잃는다. H200 이 Hopper 이고 대략
**100k 토큰 이상**에서 발현된다. two-level accumulation
(`flash-attention#104`, +#96/#91)으로 89% 까지 복구되어 mainline 에 들어갔지만
**이 서버의 vLLM 0.19.1 에 그 수정이 들어있는지는 확인되지 않았다.**

vLLM 이 제시한 안전/위험 구분:

| | 조건 |
|---|---|
| 안전 | decode 중심 · 메모리 바운드 · 컨텍스트 >7k · head_dim 64/128 · FA3/FlashInfer |
| 위험 | 컨텍스트 <7k · head_dim=256 + prefill 중심 · **작은 sliding-window 층을 가진 hybrid** · uncalibrated 정확도 <95% |

마지막 항목이 걸린다 - `qwen3.8-27b` 이 hybrid 다. GatedDeltaNet 은 sliding window 와
다르지만 확인 없이 안전하다고 볼 근거도 없다.

**버전 족보를 파는 것보다 직접 재는 편이 빠르다:**

```bash
uv run python deploy_vlms/scripts/check_kv_longctx.py
```

8k(대조군) / 64k / 128k / 200k 를 깊이 3곳씩 재서 결론 하나를 낸다. 짧은 구간은 통과하는데
100k 부터 무너지면 그 증상이고 그때의 조치는 `--kv-cache-dtype fp8` 제거 +
`MAX_NUM_SEQS` 8 -> 4 다(bf16 KV 는 64KiB/token 이라 262k 는 4-way 가 상한).
전 구간 통과면 아무것도 바꾸지 않는다.

---

## 2. KV cache: 길이에 비례하는 유일한 덩어리

attention 은 매 토큰마다 앞의 모든 토큰의 key/value 를 본다. 매번 다시 계산하면
O(n^2) 이므로 계산해 둔 것을 캐싱한다. 그게 KV cache 다.

```
토큰당 KV 바이트 = 2 (K,V) x attention 층 수 x KV head 수 x head_dim x 바이트/값
```

`--kv-cache-dtype` 이 마지막 항을 2 -> 1 로 바꾼다. 나머지는 모델 구조가 정한
상수라 **설정으로 못 바꾼다**.

현재 서빙 중인 셋:

| 모델 | KV/token | max_model_len | max_num_seqs | 최대 KV 소요 |
|---|---|---|---|---|
| `mai-ui-8b` | ~56 KiB (bf16) | 16,384 | 16 | ~14 GiB |
| `paddleocr-vl-1.5` | 작음 (0.9B) | 8,192 | 8 | 무시 가능 |
| `qwen3.8-27b` | **32 KiB (fp8)** | 262,144 | 8 | ~64 GiB |

### hybrid attention 이 이 계산을 깨뜨린다

`qwen3.8-27b` 는 64개 층 중 **16개만** full attention 이다(`full_attention_interval: 4`).
나머지 48개는 GatedDeltaNet - 선형 attention 이라 **컨텍스트 길이와 무관한
고정 크기 상태**만 든다 (시퀀스당 ~150MB, 8토큰이든 100만 토큰이든 같다).

같은 크기의 평범한 dense 27B 대비 KV 가 **1/4** 이다.

| | 토큰당 KV (fp8) | 262,144 토큰 | 1,010,000 토큰 |
|---|---|---|---|
| qwen3.8-27b (16/64 층) | 32 KiB | 8 GiB | 30.8 GiB |
| *가상의 dense 27B (64/64 층)* | *128 KiB* | *32 GiB* | *123 GiB (안 들어감)* |

**주의:** `serve_vlm.py` 의 `estimate_kv_cache_bytes_per_token()` 은 `num_hidden_layers`
전부가 KV 를 든다고 계산한다. 이 모델에서는 ~4배 과대평가라서 AUTO_TUNE 을 켜면
"안 들어간다" 며 기동을 막는다. `qwen3.8-27b.env` 가 AUTO_TUNE 을 끄고 `u` 를
직접 박은 이유다.

---

## 3. Context window: 세 개의 다른 숫자

"컨텍스트 26만" 이라고 할 때 실제로는 서로 다른 세 가지가 섞여 있다.

| 숫자 | 어디서 오나 | 바꿀 수 있나 |
|---|---|---|
| **native** | 체크포인트 `config.json` 의 `max_position_embeddings`. 학습된 길이. | 아니오 |
| **max_model_len** | 우리가 여는 길이. `--max-model-len`. | 예, native 이하로는 자유롭게 |
| **effective** | 실제로 정확히 회수되는 길이. 벤치로만 알 수 있다. | 간접적으로만 |

`max_model_len` 을 native 이상으로 올리려면 위치 인코딩을 늘려야 한다 (다음 절).
그냥 올리면 vLLM 이 거부한다. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 로 그 검사를
끄면 **경고 없이 쓰레기 출력**이 나온다 (검사만 껐지 아무것도 안 늘렸으므로).

### 기동 게이트와 런타임 상한은 다른 것이다

혼동이 잦은 지점이다.

- **기동 시** vLLM 은 시퀀스 **1개**가 `max_model_len` 까지 들어가는지만 본다.
  `qwen3.8-27b` 는 262144 x 32KiB = 8GiB <= 71GiB pool 이므로 항상 통과한다.
- **`MAX_NUM_SEQS` 는 기동 검사와 무관하다.** 런타임 동시성 상한일 뿐이고
  8개가 동시에 꽉 채우면 vLLM 이 알아서 preempt 한다.

그래서 기동이 KV 부족으로 실패하면 레버는 `GPU_MEMORY_UTILIZATION` 을 올리거나
`MAX_MODEL_LEN` 을 내리는 **둘뿐**이다. `MAX_NUM_SEQS` 를 줄여도 안 고쳐진다.

---

## 4. RoPE 와 YaRN: native 를 넘길 때

**RoPE** 는 위치를 벡터 **회전**으로 인코딩한다. head 차원을 쌍으로 나누고 쌍 i 는
고유 주파수 `theta_i = base^(-2i/d)` 로 회전한다. 낮은 인덱스는 빠르게(국소 위치),
높은 인덱스는 느리게(전역 위치) 돈다. attention 점수가 `(m - n)` 에만 의존하게 되는
것이 이 방식의 우아함이다.

**native 를 넘기면 왜 깨지나:** 느리게 도는 차원들은 학습 중 한 바퀴도 못 돌았다.
262,144 를 넘긴 위치에서는 모델이 **본 적 없는 각도**가 나온다.

**순진한 해법 (Position Interpolation):** 모든 위치를 4로 나눠 범위 안에 넣는다.
동작은 하는데 **빠른 차원까지 같이 뭉개진다** - "바로 앞 토큰" 과 "두 칸 앞 토큰" 의
회전이 거의 같아져 국소 해상도를 잃는다. 길이를 사고 근거리를 판 셈이다.

**YaRN (Yet another RoPE extensioN):** **차원마다 문제가 다르니 처방도 다르게**
간다. 그게 핵심이다.

- 빠른 차원 (파장 << 262k): 이미 여러 바퀴 돌았다 -> **그대로 둔다**
- 느린 차원 (파장 >= 262k): 한 바퀴도 못 돌았다 -> **보간한다**
- 중간 대역: 부드러운 ramp

여기에 **attention temperature** 보정이 붙는다. 토큰이 4배 늘면 attention 엔트로피가
올라 분포가 평평해지므로 logit 에 `~0.1*ln(s)+1` 을 곱해 다시 날카롭게 만든다.
q/k 스케일링에 접히므로 런타임 비용은 0이다.

vLLM 은 YaRN 을 **네이티브 지원**한다 (플러그인 불필요, 0.19.1 로 충분):

```
--hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}'
--max-model-len 1010000
```

키 이름이 transformers 버전에 따라 `rope_parameters`(최신) / `rope_scaling`(구버전)로
갈린다. **틀린 키는 조용히 무시**되므로 체크포인트의 `config.json` 을 직접 확인하고
기동 로그에 스케일링이 찍히는지 볼 것.

### "static" 이 대가다

factor 는 로드 시점에 고정되어 **모든 요청에** 적용된다. 800 토큰짜리 프롬프트도
80만 토큰짜리와 같은 4배 처방을 받고 쓰지도 않는 범위 때문에 정확도를 낸다.
Qwen 문서가 경고하는 지점이고 **긴 컨텍스트가 상시 요구사항일 때만** 켜는 이유다.

---

## 5. Prefix cache: 멀티턴이 싼 이유

vLLM V1 은 KV 블록을 프롬프트 prefix 로 해싱해 캐싱한다(기본 on). 대화 2턴째는
바뀐 **꼬리만** prefill 하고 앞부분은 캐시 히트다. 코딩 에이전트나 반복 grounding
호출이 실용적인 것은 사실상 이 기능 덕분이다.

**핵심 트레이드오프:** 캐시된 prefix 는 **KV pool 을 그대로 쓴다.** 별도 공간이 아니다.

```
71GiB pool 기준
  대화당 262k (8GiB)   -> ~8개 대화 캐시 유지
  대화당 1M  (30.8GiB) -> ~2개
```

즉 **길이를 늘리면 캐시 보존 개수가 준다.** 인터랙티브 용도에서는 이게 대개
손해다 - 거의 안 닿는 천장보다 턴 지연이 훨씬 자주 체감된다. 여유를 쓰고 싶으면
`max_model_len` 을 늘리기보다 `GPU_MEMORY_UTILIZATION` 을 올려 pool 자체를
키우는 편이 낫다 (품질 비용 0, 새 실패 모드 0).

---

## 6. Prefill vs Decode: 병목이 다른 두 국면

| | Prefill | Decode |
|---|---|---|
| 하는 일 | 프롬프트 전체를 한 번에 읽어 KV 를 채운다 | 토큰을 하나씩 만든다 |
| 병목 | **compute 바운드** | **메모리 대역폭 바운드** |
| 비용 스케일 | 선형항 + attention 의 n^2 항 | 토큰당 가중치 전체를 1회 읽음 |
| 체감 지표 | **TTFT** (첫 토큰까지) | tokens/sec |
| 듣는 knob | `MAX_NUM_BATCHED_TOKENS`, prefix cache | 가중치 정밀도, 추측 디코딩 |

`qwen3.8-27b` 단일 H200 대략치:

| 프롬프트 길이 | 선형항 | attention 항 (16층) | TTFT 대략 |
|---|---|---|---|
| 262,144 | ~30s | ~20s | **~50s** |
| 1,010,000 | ~2min | ~5min | **5-10min** |

attention 항이 n^2 이라 4배 길이에 16배로 뛰는 것이 보인다. hybrid 구조가 이걸
이미 4배 깎아준 결과가 저 값이다.

**`MAX_NUM_BATCHED_TOKENS` (chunked prefill):** 긴 프롬프트를 한 번에 넣지 않고
이 크기로 쪼개 넣는다. 16384 면 100만 토큰은 62 라운드다. 키우면 prefill 이
빨라지지만 activation 메모리를 더 쓴다. 이미지 1장(1920x1080)이 ~1300-2600 토큰
이므로 16384 는 한 청크에 넉넉히 들어간다.

---

## 7. GPU 밖의 제약: 호스트 RAM 16GB

이 서버에서 **GPU 보다 먼저 한계에 닿는 것은 호스트 RAM 이다.** 프로세스 수가
실질 상한을 만든다 (TP=1 이면 모델당 API + EngineCore 2개).

GPU 가 아니라 시스템 메모리를 쓰는 knob 들:

| 인자 | 기본값 | 왜 만지나 |
|---|---|---|
| `--mm-processor-cache-gb` | 4GiB (**host**) | 디코딩 이미지 캐시. 스크린샷은 매번 달라 히트가 없다 -> **0**. API/EngineCore 양쪽에 중복 배치되므로 3인스턴스 기준 가장 큰 확정 절감. |
| `--load-format safetensors --safetensors-load-strategy lazy` | 자동 판단 | lazy=mmap 이라 가중치를 익명 메모리로 복사하지 않는다. 기본값은 NFS 에서 "체크포인트 <= 가용 RAM 90%" 면 자동 prefetch 로 새므로 **명시**한다. |
| `--swap-space` | - | **넣지 말 것.** 이 서버의 vLLM 에서 제거된 인자다 (V1 이 preemption 을 recompute 로 처리). 2026-08-11 기동 실패 실측. |

점검은 `deploy_vlms/scripts/check_host_ram.py` (warm 상태에서 실행). `ps` 의 RSS 는
mmap 파일 캐시가 섞여 못 쓰고 `smaps_rollup` 의 `Pss` / `Pss_Anon` 을 본다.
`MemAvailable` 이 2048MiB 아래면 그 구성은 유지하면 안 된다.

**주의:** OOM killer 는 원인 프로세스를 고르지 않는다. qwen 이 RAM 을 밀어붙여도
죽는 것은 mai-ui / paddleocr - 즉 production 루프일 수 있다.

---

## 8. HTTP 경로: 가장 약한 고리가 지배한다

모델이 아무리 길게 받아도 요청이 거기 못 닿으면 의미가 없다. 사슬 전체:

```
client (poc/workflow_3/vlm/vlm_client.py, timeout_sec=120.0)
  -> nginx      (proxy_read_timeout ~300s / client_max_body_size ?)
  -> Flask      (VLM_SERVE_READ_TIMEOUT_SEC=300.0, stream=False)
  -> vLLM
```

| 길이 | 필요 시간 | 현재 사슬 |
|---|---|---|
| 262,144 | ~50s + decode | 통과 (client 120s 가 최단이지만 여유) |
| 1,010,000 | 5-10min | **client 에서 먼저 끊긴다** |

두 가지 개념이 헷갈리기 쉽다:

- **`proxy_read_timeout` 은 "연속된 read 사이" 간격**이지 응답 전체 시간이 아니다.
  Flask proxy 를 `stream=True` 로 바꾸면 SSE 토큰마다 타이머가 리셋되어
  **decode 구간은 문제에서 사라진다.** 다만 첫 토큰은 prefill 이 끝나야 나오므로
  **TTFT 는 여전히 timeout 을 넘겨야 한다.**
- **`client_max_body_size` 는 timeout 과 별개다.** 100만 토큰 프롬프트는 UTF-8
  2-4MB 다. nginx 기본값 `1m` 이면 Flask 에 닿기도 전에 413 이다. 큰 이미지
  payload 에도 걸리는 문제라 길이와 무관하게 확인할 값이다
  (`/api/model_upload/` 만 128m 로 튜닝되어 있다 - `deploy_vlms/nginx/`).

---

## 9. 현재 설정 읽어보기

| knob | mai-ui-8b | paddleocr-vl-1.5 | qwen3.8-27b |
|---|---|---|---|
| GPU | 0 (공유) | 0 (공유) | 1 (단독) |
| `GPU_MEMORY_UTILIZATION` | 0.45 | 0.20 | 0.90 |
| `MAX_MODEL_LEN` | 16,384 | 8,192 | 262,144 |
| `MAX_NUM_SEQS` | 16 | 8 | 8 |
| `MAX_NUM_BATCHED_TOKENS` | 16,384 | - | 16,384 |
| KV dtype | bf16 | bf16 | **fp8** |
| 역할 | grounding (지연 민감) | OCR 보조 | 범용 추론 + 코딩 |

`mai-ui` 가 `MAX_NUM_SEQS=16` 인데 지연 민감이라는 것이 모순처럼 보이지만
2단 locator 는 **순차 2콜**이라 배치 크기가 아니라 prefill 여유가 레버다.

---

## 10. 결정 기록: 1M 컨텍스트 (2026-09-03 검토, 미채택)

**결론: `MAX_MODEL_LEN=262144` 유지.** GPU 메모리는 문제가 아니었다.

- fp8 KV 기준 1M 은 시퀀스당 30.8GiB. 71GiB pool 에 **2개 동시 수용** 가능하고
  기동 게이트(시퀀스 1개)는 40GiB 여유로 통과한다.
- 막는 것은 메모리가 아니라 넷이다:
  1. **static YaRN 세금** - 코딩/grounding 은 짧은 턴이 대부분인데 전부 4배 처방을 받는다
  2. **prefix cache 축소** - 대화당 8GiB -> 30.8GiB 라 캐시 보존이 8개 -> 2개
  3. **prefill 5-10분** - HTTP 사슬 전체를 900s 로 올려야 하고 인터랙티브에 부적합
  4. **호스트 RAM** - 요청당 수백 MB 가 16GB 호스트에 얹힌다 (swap 없음)
- 코딩 용도로 재확인: repo 전체가 `.py`+`.md` 2.9M 토큰이라 **1M 으로도 안 들어간다.**
  "다 넣기" 는 어느 설정에서도 성립하지 않으므로 최적화할 축이 아니다. 실제 코딩
  세션은 시스템 프롬프트 + 타겟 파일 읽기 + diff 로 100-250k 에 수렴한다.

여유가 필요하면 길이 대신 pool 을 키운다: `GPU_MEMORY_UTILIZATION` 0.90 -> 0.93
(단독 점유라 가능, 품질 비용 없음). 진짜로 >262k 가 필요한 **비대화형** 작업이
생기면 별도 env 파일(`qwen3.8-27b-1m.env`, `MAX_NUM_SEQS=2`)로 띄우고 262k 인스턴스를
내린다 - 호스트 RAM 규칙상 어차피 공존 못 한다.

---

## 참고

- 모델/추천 설정: [vLLM Recipes - Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B),
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- 저장소 정본: `deploy_vlms/config/common.env`, `deploy_vlms/config/models/*.env`
- 관련 문서: [`01-runtime-layout-and-capacity.md`](./01-runtime-layout-and-capacity.md),
  [`07-h100-downgrade-capacity.md`](./07-h100-downgrade-capacity.md)
