# VLM 스택 vs. Kimi-K2 리소스 비교 (H200 140GB 기준)

본 문서는 현재 프로젝트에 배포된 다중 VLM 스택과, 동일 하드웨어에서 단일 프런티어 MoE 모델(Kimi-K2 / K2.5 계열)을 운영하는 경우를 비교한다. 목적은 "왜 거대 단일 모델 대신 5개의 특화 모델을 운영하는가"에 대한 의사결정 자료를 제공하는 것이다.

근거 자료:

- 로컬 설정: `deploy_vlms/config/common.env`, `deploy_vlms/config/models/*.env`
- 서비스 레지스트리: `flask_api/vlm_serve/config.py`
- 용량 산정 공식: `docs/setup_vlms/01-runtime-layout-and-capacity.md` §6

## 1. 하드웨어 기준

- GPU 서버: **H200 140 GiB × 2** (`common.env: GPU_TOTAL_MEMORY_GIB=140`)
- 서빙 런타임: vLLM (BF16) + transformers (GOT-OCR)
- GPU당 예약 오버헤드: `GPU_SHARED_RESERVE_GIB=8`, `GPU_PROCESS_RESERVE_GIB=4`

## 2. 현재 배포된 서비스 현황

| 서비스 | 파라미터 수 | 가중치(BF16) | GPU | vLLM `u` | 예약 VRAM | 역할 |
|---|---|---|---|---|---|---|
| UI-Venus-1.5-8B | 8.3 B | ~16.6 GiB | GPU 0 | 0.44 (auto) | ~62 GiB | 메인 GUI grounding |
| UI-TARS-1.5-7B | 7.6 B | ~15.2 GiB | GPU 0 | 0.44 (auto) | ~62 GiB | GUI agent (대안 경로) |
| MAI-UI-8B | 8 B | ~16 GiB | GPU 1 | 0.45 | ~63 GiB | UI grounding (대안) |
| PaddleOCR-VL-1.5 | 0.9 B | ~1.8 GiB | GPU 1 | 0.10 | ~14 GiB | OCR 보조 |
| GOT-OCR-2.0-hf | 0.58 B | ~1.2 GiB | GPU 1 | (transformers) | ~4 GiB | OCR fallback |
| **합계** | **~25 B** | **~51 GiB 가중치** | **2 × H200** | — | **~205 GiB / 280 GiB (~73%)** | **5개 특화 서비스** |

카드별 실제 설정 사용률:

- **GPU 0 (140 GiB):** UI-Venus + UI-TARS, auto-tune으로 각 `u≈0.44` → **~123 GiB 사용 (88%)**, vLLM 엔진 2개 코로케이션.
- **GPU 1 (140 GiB):** MAI-UI (`u=0.45`) + PaddleOCR-VL (`u=0.10`) + GOT-OCR (~4 GiB transformers) → **~81 GiB 사용 (58%)**, 엔진 3개 코로케이션, 버스팅용 헤드룸 보유.

런타임 문서의 auto-tune 공식:

```
u_recommended = ((M_gpu − M_shared) / N_models − M_proc) / M_gpu
              = ((140 − 8) / 2 − 4) / 140 ≈ 0.44   # GPU 1장에 8B급 모델 2개 올릴 때
```

BF16 가중치 풋프린트는 `2 bytes × params`. vLLM PagedAttention KV 캐시와 멀티모달 비전 인코더(Qwen2.5-VL ViT ≈ 675 M, UI-Venus / UI-TARS 계열에서 공유)가 시퀀스 할당 전부터 통상 20–30%를 추가로 점유한다.

## 3. Kimi-K2 리소스 풋프린트

**명칭 주의.** Moonshot이 공개한 라인은 **Kimi-K2** (2025년 7월)와 **Kimi-K2-Thinking** (2025년 11월)이며, 공개적으로 확인되는 **Kimi-K2.5** 릴리스는 없다. 아래 수치는 K2 계열 아키텍처 기준이며, K2.5가 아키텍처를 변경하지 않았다면(Moonshot은 마이너 버전 사이에 아키텍처를 바꾼 전례가 거의 없음) 동일한 자릿수로 봐도 무방하다.

**Kimi-K2 아키텍처 (Moonshot AI):**

- 총 **1.04 T 파라미터** (Mixture-of-Experts)
- 토큰당 **활성 32 B 파라미터**
- Expert 384개, 토큰당 8개 선택 + 1개 공유
- 컨텍스트 128 K
- 네이티브 서빙 정밀도: **INT8 / FP8** weight-only (Moonshot 레퍼런스)

**가중치만 고려한 메모리 (KV 캐시·활성값 제외):**

| 정밀도 | 바이트/파라미터 | 풋프린트 | H200 140 GiB 1장에 적재 가능? |
|---|---|---|---|
| BF16 | 2 B | **~2.08 TiB** | 불가 — H200 약 15장 필요 |
| FP8 / INT8 | 1 B | **~1.04 TiB** | 불가 — H200 약 8장 필요 |
| INT4 (AWQ / GPTQ) | 0.5 B | **~520 GiB** | 불가 — H200 약 4장 필요 |
| INT4 + 공격적 offload | — | ~520 GiB 디스크 + RAM swap | 이론상 가능하나 단자릿수 tok/s, **서비스 운영 불가** |

**Kimi-K2 1개 인스턴스의 현실적 운영 구성:**

- **FP8:** H200 8장, TP=8 (Moonshot 공식 권장)
- **INT4 (커뮤니티 양자화):** 최소 H200 4장, 다단계 추론 정확도 손실 측정됨
- **추가로** 128 K 컨텍스트 + 적당한 동시성에서 KV 캐시 헤드룸 50–100 GiB 필요

## 4. H200 140 GiB 1장 기준 동일 조건 비교

| 지표 | 본 프로젝트 스택 (H200 1장) | Kimi-K2 (H200 1장) |
|---|---|---|
| 동시 적재 모델 수 | **2–3개** (예: UI-Venus + UI-TARS, 또는 MAI-UI + OCR 2종) | **0개** — 어떤 운영 가능 정밀도로도 적재 불가 |
| 서빙 파라미터 합계 | 12–17 B | 0 |
| 동시 특화 태스크 | GUI grounding + OCR + agent fallback | 없음 |
| Kimi-K2 1회 서빙 최소 하드웨어 | — | **H200 8장 (FP8)** 또는 H200 4장 (INT4, 손실 있음) |

전체 2× H200 서버 기준으로 본 프로젝트는 **280 GiB 중 ~205 GiB를 사용해 5개 특화 모델을 서빙**한다. 동일한 2장으로는 Kimi-K2 1개 인스턴스의 **0%**도 서빙할 수 없다.

## 5. 보고용 핵심 메시지

1. **하드웨어 배수.** Kimi-K2 FP8 1개 인스턴스에는 **H200 약 8장**이 필요. 본 5-모델 특화 스택은 **H200 2장**으로 운영. 레시피 생성 파이프라인에 실제 필요한 GUI + OCR 태스크 전부를 커버하면서 **하드웨어 4배 절감**.
2. **GPU당 모델 밀도.** 본 스택은 **H200당 2.5개 모델** (5 서비스 / 2 카드). Kimi-K2는 **H200당 0.125개 모델** (1 모델 / 8 카드). **약 20배 밀도 우위**.
3. **가중치 풋프린트 효율.** CD-SEM 레시피 태스크 표면(GUI grounding + OCR)은 **가중치 ~51 GiB**로 충분. Kimi-K2는 동일 도메인을 커버하려 해도 별도 fine-tuning이 필요한 **범용 모델 가중치 ~1,040 GiB**를 소비. **약 20배 풋프린트 절감**.
4. **레이턴시 측면 (정성적).** 32 B-active MoE를 TP=8로 돌리면 저배치 기준 통상 30–60 tok/s/req 수준. 본 7–8 B dense VLM은 H200 1장 + vLLM PagedAttention에서 통상 80–150 tok/s/req (좌표/JSON 짧은 출력 기준). 즉, 특화로 인한 레이턴시 페널티가 없으며 실제 출력 형태에서는 오히려 유리.
5. **포기하는 것.** 범용 추론 능력. Kimi-K2-Thinking은 프런티어 추론 모델이며 본 스택이 이를 대체하지 않음. 솔직한 프레이밍: 우리는 **불필요한 범용 지능을 의도적으로 포기**하고, 우리가 실제로 읽어야 하는 화면에 대해 **하드웨어 효율을 약 20배 확보**했다.

## 6. 단서 및 한계

- K2 FP8 기준 H200 8장 수치는 Moonshot 공식 레퍼런스. 실제 운영(Together, Fireworks, DeepInfra)에서는 H100 SXM 8장 또는 H200 8장이 일반적이며 처리량 확보를 위해 16-way를 쓰기도 함.
- INT4 K2 배포(커뮤니티 양자화)는 존재하나 다단계 추론 정확도가 눈에 띄게 떨어져 FP8 K2 수치와의 직접 비교는 공정하지 않음.
- GPU 1의 `u=0.45`가 보수적인 이유는 GOT-OCR이 vLLM이 아닌 `transformers`로 동작하여 VRAM이 두 할당자에 분산되기 때문. 따라서 MAI-UI를 더 올리지 않음. **GPU 1에는 ~50 GiB 헤드룸**이 남아 있어 6번째 서비스를 추가할 여지가 있음.
- 본 비교는 정적 용량 비교임. 피크 부하에서의 throughput-per-dollar는 별도 벤치마크가 필요하나 정성적으로도 특화 스택이 우위.

## 7. 수치 출처

- 모델별 `u` 값, GPU 배치, 포트: `deploy_vlms/config/models/*.env` (저장소 커밋됨)
- `GPU_TOTAL_MEMORY_GIB`, `GPU_SHARED_RESERVE_GIB`, `GPU_PROCESS_RESERVE_GIB`: `deploy_vlms/config/common.env`
- Auto-tune 공식: `docs/setup_vlms/01-runtime-layout-and-capacity.md` §6
- Kimi-K2 아키텍처 (1 T MoE, 32 B active, expert 384개): Moonshot AI 공식 model card 및 technical report
- 하드웨어 배수 (FP8 기준 H200 8장): Moonshot 레퍼런스 배포 가이드
