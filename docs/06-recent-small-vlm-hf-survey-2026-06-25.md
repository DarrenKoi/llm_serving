# 06. 최근 Hugging Face 소형 VLM 조사

조사일: 2026-06-25 KST

목적: 현재 `deploy_vlms`에서 쓰는 소형/특화 VLM 스택보다 나은 Hugging Face 후보가 새로 나왔는지 확인하고, GUI grounding, OCR/문서 파싱, 일반 VLM 추론 관점에서 교체 가능성을 판단한다.

## 1. 결론

현재 스택을 전면 교체할 모델은 아직 없다.

- GUI grounding은 현재 `UI-Venus-1.5-8B`와 `MAI-UI-8B` 계열이 이미 최신 상위권이다. 최근 `VISTA-9B`가 `ScreenSpot-Pro 69.2`로 `UI-Venus-1.5-8B`의 공개 수치인 `68.4`보다 약간 높지만, 차이가 작고 RCS/SEM 화면 검증이 필요하다.
- `UI-TARS-1.5-7B`는 여전히 agent-style 비교군으로 가치가 있다. 다만 좌표 grounding만 보면 `UI-Venus-1.5`/`VISTA` 계열을 먼저 비교하는 편이 맞다.
- OCR/문서 파싱은 변화가 있다. `dots.ocr`는 1.7B급 문서 파싱 모델로 `GOT-OCR`보다 OmniDocBench 표/문서 지표가 강하게 나온다. `PaddleOCR-VL-1.5`를 바로 빼기보다는 `dots.ocr`를 문서/표 파싱 sidecar로 추가 벤치하는 쪽이 현실적이다.
- 일반 VLM reasoning, 장문/동영상, OCR 보조에는 `MiniCPM-V 4.5`와 `Step3-VL-10B`가 강하다. 하지만 GUI click 좌표 모델로는 현재 GUI-specialized 모델보다 우선순위가 낮다.

권장 액션:

1. `VISTA-9B`를 `ui-venus`/`mai-ui`와 같은 screenshot set으로 head-to-head 벤치한다.
2. `dots.ocr`를 `got-ocr` 대체가 아니라 `paddleocr-vl-1.5` 이후의 문서/표 파싱 후보로 추가 벤치한다.
3. `Step3-VL-10B` 또는 `MiniCPM-V 4.5`는 align-fail 설명, post-hoc review, 장문/다중 이미지 분석 후보로만 먼저 평가한다.
4. 어떤 모델도 CV 좌표 결정을 대체하지 않는다. 이 repo의 원칙은 계속 `VLM proposes / CV verifies and decides`다.

## 2. 현재 repo 기준선

현재 운영 문서와 설정 기준:

| Service | Model | Repo 역할 | 배포 특성 |
|---|---:|---|---|
| `ui-venus` | `UI-Venus-1.5-8B` | 전체 화면 GUI grounding, coarse bbox | vLLM, GPU 0 |
| `mai-ui` | `MAI-UI-8B` | crop/zoom 후 정밀 클릭점 보정 | vLLM, GPU 1 |
| `ui-tars` | `UI-TARS-1.5-7B` | GUI agent 대안 경로 | vLLM, Qwen2.5-VL 계열 template 주의 |
| `paddleocr-vl-1.5` | `PaddleOCR-VL-1.5` | OCR, spotting, table/layout extraction | vLLM, 0.9B급 |
| `got-ocr` | `GOT-OCR-2.0-hf` | hard crop OCR fallback | transformers 직접 추론 |

근거: [`docs/project_progress/01_vlm_deployment.md`](../project_progress/01_vlm_deployment.md), [`deploy_vlms/config/models/`](../../deploy_vlms/config/models/), [`docs/setup_vlms/04-operations-integration-and-benchmarking.md`](./04-operations-integration-and-benchmarking.md).

## 3. 후보 요약

| 후보 | 크기 | 강한 영역 | 공개 benchmark 근거 | repo 적용 판단 |
|---|---:|---|---|---|
| [`inclusionAI/VISTA-9B`](https://huggingface.co/inclusionAI/VISTA-9B) | 9B | GUI click-point grounding | SSPro `69.2`, SSV2 `95.8`, OSWorld-G `68.1`, OSWorld-G-R `75.5` | 가장 먼저 벤치할 신규 GUI 후보 |
| [`inclusionAI/VISTA-4B`](https://huggingface.co/inclusionAI/VISTA-4B) | 4B (HF card는 `5B params`로 표기) | 저비용 GUI grounding | SSPro `64.2`, SSV2 `93.8`, OSWorld-G `61.2` | latency/VRAM 절감 후보 |
| [`inclusionAI/UI-Venus-1.5-30B-A3B`](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B) | 31B total MoE | GUI grounding/navigation | family 공개 수치: SSPro `69.6`, VenusBench-GD `75.0`, AndroidWorld `77.6` | 현재 8B가 부족할 때 scale-up 후보 |
| [`stepfun-ai/Step3-VL-10B`](https://huggingface.co/stepfun-ai/Step3-VL-10B) | 10B | multimodal reasoning, OCR, GUI benchmark 전반 | MMMU `78.11`, OCRBench `86.75`, ScreenSpot-V2 `92.61`, ScreenSpot-Pro `51.55`, OSWorld-G `59.02` | GUI primary 교체보다 reasoning sidecar 후보 |
| [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) / [`Thinking`](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | 9B | 범용 VLM, visual agent, OCR, video/long context | 비교표 기준 Qwen3-VL-Thinking 8B: OCRBench `82.85`, SSV2 `93.60`, SSPro `46.60`, OSWorld-G `56.70` | UI-TARS/Qwen2.5 계열 후속 후보지만 GUI 점수는 UI-Venus/VISTA보다 낮음 |
| [`openbmb/MiniCPM-V-4_5`](https://huggingface.co/openbmb/MiniCPM-V-4_5) | 8B | OCR, document parsing, high-FPS video, mobile/edge | OpenCompass 평균 `77.0`, OCRBench leading claim, Video-MME/long-video 효율 강조 | 문서/동영상/다중 이미지 분석 후보 |
| [`moonshotai/Kimi-VL-A3B-Instruct`](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) | 16B total, 2.8B active | long context, OCR, agent benchmark | OCRBench `867`, InfoVQA `83.2`, ScreenSpot-V2 `92.8`, ScreenSpot-Pro `34.5` | vLLM MR branch 의존. 즉시 운영 후보는 아님 |
| [`OpenGVLab/InternVL3_5-8B`](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | 8B | 범용 VLM, CascadeRL, reasoning | 비교표 기준 OCRBench `83.70`, SSPro `15.39`, MMMU `71.69` | GUI/RCS 용도 우선순위 낮음 |
| [`rednote-hilab/dots.ocr`](https://huggingface.co/rednote-hilab/dots.ocr) | 1.7B LLM foundation | OCR, layout, table, formula, reading order | OmniDocBench EN/ZH overall edit `0.125/0.160`, table TEDS `88.6/89.0`; GOT-OCR보다 문서 파싱 지표 우수 | `got-ocr` 보완/대체 후보. crop OCR보다 document parser로 평가 |

## 4. GUI grounding 후보 판단

### 4.1 VISTA-9B

`VISTA-9B`는 2026년 6월 공개된 GUI-grounding 모델이다. screenshot과 자연어 instruction을 받아 normalized `0-1000` 좌표계의 click coordinate를 직접 반환한다.

공개 수치:

| Model | SSPro | SSV2 | OSWorld-G | OSWorld-G-R |
|---|---:|---:|---:|---:|
| Qwen3.5-9B | 65.2 | 91.9 | 63.1 | 74.6 |
| GRPO-9B | 68.3 | 95.2 | 67.5 | 75.2 |
| VISTA-9B | 69.2 | 95.8 | 68.1 | 75.5 |

좋은 점:

- 현재 `UI-Venus-1.5-8B` 공개 수치인 ScreenSpot-Pro `68.4`보다 약간 높다.
- 출력이 click point 중심이라 `MAI-UI`의 refined point 역할과 직접 비교하기 좋다.
- vLLM/SGLang 예제가 model card에 있다.

주의:

- SSPro `+0.8p` 수준의 차이라 production 교체 근거로는 약하다.
- RCS/SEM 화면은 일반 GUI benchmark와 다르다. 특히 작은 공정 파라미터 텍스트와 반복 패턴이 많다.
- 좌표를 직접 반환하더라도 repo 원칙상 최종 클릭/이동 좌표는 CV snap 또는 post-verification gate를 통과해야 한다.

판단: `ui-venus` 대체 후보라기보다, `ui-venus -> mai-ui` 2-stage와 같은 screenshot set에서 비교할 신규 9B 후보다.

### 4.2 UI-Venus-1.5 scale-up

현재 repo는 `UI-Venus-1.5-8B`를 쓴다. Hugging Face model card의 scale trend는 ScreenSpot-Pro가 `2B 57.7 -> 8B 68.4 -> 30B-A3B 69.6`으로 오른다고 설명한다.

판단:

- 8B에서 이미 대부분의 이득이 나온다.
- 30B-A3B는 `+1.2p` 정도라 H200 slot을 더 쓰는 비용 대비 이득이 작을 수 있다.
- 현재 문제 표면이 RCS/SEM 특화라면 scale-up보다 local screenshot benchmark가 먼저다.

### 4.3 UI-TARS-1.5 유지 여부

`UI-TARS-1.5-7B` model card는 다음 수치를 공개한다.

| Benchmark | UI-TARS-1.5 |
|---|---:|
| OSWorld, 100 steps | 42.5 |
| Windows Agent Arena, 50 steps | 42.1 |
| WebVoyager | 84.8 |
| ScreenSpot-V2 | 94.2 |
| ScreenSpot-Pro | 61.6 |

판단:

- agent-style long-horizon 비교군으로는 여전히 의미가 있다.
- 순수 click grounding만 보면 `UI-Venus-1.5-8B`, `VISTA-9B`, `MAI-UI` 계열보다 우선순위가 낮다.
- 이미 repo에서 UI-TARS는 "대안 경로"로 분리되어 있으므로 그 위치가 맞다.

## 5. OCR/문서 파싱 후보 판단

### 5.1 dots.ocr

`dots.ocr`는 1.7B foundation의 multilingual document parser다. layout detection, content recognition, reading order, table, formula를 단일 VLM으로 처리한다.

OmniDocBench 공개 지표 일부:

| Model | Overall Edit EN/ZH 낮을수록 좋음 | Table TEDS EN/ZH 높을수록 좋음 | Text Edit EN/ZH 낮을수록 좋음 |
|---|---:|---:|---:|
| GOT-OCR | 0.287 / 0.411 | 53.2 / 47.2 | 0.189 / 0.315 |
| Qwen2.5-VL-72B | 0.214 / 0.261 | 82.9 / 83.9 | 0.092 / 0.180 |
| Gemini2.5-Pro | 0.148 / 0.212 | 85.8 / 86.4 | 0.055 / 0.168 |
| dots.ocr | 0.125 / 0.160 | 88.6 / 89.0 | 0.032 / 0.066 |

좋은 점:

- `GOT-OCR`보다 문서/표/reading-order 지표가 훨씬 좋다.
- vLLM/SGLang 예제가 있고 output을 JSON/Markdown 형태로 설계하기 쉽다.
- `PaddleOCR-VL`처럼 "텍스트만 읽기"보다 document layout extraction에 강점이 있다.

주의:

- RCS 작은 crop 재판독은 문서 파싱 benchmark와 다르다.
- 현재 `PaddleOCR-VL-1.5`는 crop-first OCR evidence engine으로 이미 repo 패턴이 잡혀 있다.
- `dots.ocr`를 모든 OCR에 쓰면 latency와 prompt/output schema가 불필요하게 커질 수 있다.

판단: `GOT-OCR` fallback을 대체할 가능성은 있다. 단, 첫 적용은 "문서/표/parameter panel parsing"으로 제한한다.

### 5.2 MiniCPM-V 4.5

`MiniCPM-V 4.5`는 Qwen3-8B와 SigLIP2-400M 기반의 8B급 범용 MLLM이다. model card는 OpenCompass 평균 `77.0`, OCRBench leading, 1.8M pixel 고해상도 이미지 처리, up to 10FPS video understanding을 강조한다.

판단:

- document/video/multi-image review에는 강한 후보다.
- GUI click 좌표 모델로는 specialized GUI benchmark 근거가 부족하다.
- workflow_2 align-fail 분석에서는 "같은 구조인가", "어떤 ROI를 봐야 하는가" 같은 feasibility/review 질문에 먼저 쓰는 편이 맞다.

## 6. 범용 reasoning VLM 후보 판단

### 6.1 Step3-VL-10B

`Step3-VL-10B`는 10B급으로, model card가 vLLM 공식 지원과 FP8 weight를 명시한다. 공개 수치가 텍스트로 잘 정리되어 있어 재현 추적이 쉽다.

주요 수치:

| Benchmark | Step3-VL-10B |
|---|---:|
| MMMU | 78.11 |
| MMBench EN | 92.05 |
| OCRBench | 86.75 |
| ScreenSpot-V2 | 92.61 |
| ScreenSpot-Pro | 51.55 |
| OSWorld-G | 59.02 |

판단:

- reasoning/OCR 종합은 강하지만 GUI grounding은 `UI-Venus-1.5`, `VISTA-9B`, `UI-TARS-1.5`보다 낮다.
- align-fail 화면을 설명하거나, CV 결과를 human-readable audit으로 요약하는 용도에 먼저 적합하다.
- 24GB VRAM급 최소 요구라 H200에서는 운영 가능하지만, GUI primary slot을 빼앗을 정도의 근거는 아직 없다.

### 6.2 Qwen3-VL-8B

`Qwen3-VL-8B`는 dense 8B/9B급 범용 VLM이다. model card는 PC/mobile GUI 인식, stronger 2D grounding, 256K context, 32개 언어 OCR을 강조한다.

판단:

- UI-TARS가 Qwen2.5-VL 계열이라, Qwen3-VL은 자연스러운 후속 base 후보다.
- 그러나 공개 비교표 기준 Qwen3-VL-Thinking 8B의 GUI 수치는 `ScreenSpot-Pro 46.60`, `OSWorld-G 56.70`이라 현재 GUI-specialized 모델보다 낮다. (주의: 이 수치는 HF model card 본문이 아니라 외부 비교표 출처다 — model card는 차트 이미지만 싣고 숫자를 본문에 두지 않는다. 채택 전 1차 출처 확인 필요.)
- 범용 VLM으로 stage해도 되지만, `ui-venus`/`mai-ui` 교체 후보는 아니다.

### 6.3 Kimi-VL-A3B-Instruct

Kimi-VL은 total 16B, activated 2.8B MoE다. long context 128K, OCR, ScreenSpot, OSWorld 지표를 공개한다.

좋은 점:

- activated params가 작아 비용 대비 성능이 좋다.
- InfoVQA `83.2`, OCRBench `867`, ScreenSpot-V2 `92.8` 등 문서/OCR/agent 지표가 나쁘지 않다.

주의:

- model card 기준 vLLM은 upstream MR branch 사용이 필요하다고 되어 있어 운영 리스크가 있다.
- ScreenSpot-Pro `34.5`로 GUI grounding primary 후보는 아니다.

판단: 지금은 관망. vLLM mainline 지원이 안정화되면 재검토한다.

### 6.4 InternVL3.5-8B

InternVL3.5는 1B부터 241B-A28B까지 넓은 family를 제공하고 CascadeRL을 적용한다. 하지만 8B의 GUI grounding 공개 비교 수치는 `ScreenSpot-Pro 15.39`로 이 repo의 GUI 자동화에는 약하다. (주의: `OCRBench 83.70`/`SSPro 15.39`/`MMMU 71.69`는 HF model card 본문이 아니라 외부 비교표 출처다 — card는 차트 이미지만 싣는다. 특히 SSPro 15.39는 비정상적으로 낮으니 채택 검토 시 1차 출처 재확인 필요.)

판단: 범용 VLM 연구 후보. RCS/SEM GUI 자동화 후보는 아니다.

## 7. 벤치마크 설계

공개 benchmark는 RCS/SEM Monitor와 다르다. 따라서 교체 판단은 아래 repo-local metric으로 해야 한다.

| Metric | 의미 | 측정 대상 |
|---|---|---|
| `element hit rate` | 정답 UI/영역 안에 point/bbox가 들어가는 비율 | UI-Venus, MAI-UI, VISTA, UI-TARS |
| `click-point drift(px)` | 기준 클릭점 또는 CV snap 결과와의 거리 | point-return 모델 |
| `retry count` | crop retry, sidecar 호출 횟수 | 2-stage pipeline |
| `step completion rate` | login/tool selection/search step 성공률 | workflow_1/2 smoke |
| `small-text OCR recall` | 작은 crop에서 필수 문자열 recall | PaddleOCR-VL, GOT-OCR, dots.ocr |
| `latency` | end-to-end 응답 시간 | 모든 후보 |
| `sidecar escalation rate` | primary 실패 후 OCR/sidecar 호출 비율 | 운영 비용 판단 |

벤치 순서:

1. `ui-venus` vs `VISTA-9B` vs `ui-tars`를 같은 full screenshot prompt로 비교한다.
2. best primary에 `MAI-UI` crop retry를 붙인다.
3. `VISTA-9B`를 crop retry sidecar로도 비교한다.
4. OCR은 `PaddleOCR-VL` baseline 후 `GOT-OCR`와 `dots.ocr`를 crop/document task로 나눠 비교한다.
5. `Step3-VL`/`MiniCPM-V 4.5`는 coordinate task가 아니라 explanation/review task로 별도 비교한다.

## 8. 최종 추천

| 결정 | 모델 | 이유 |
|---|---|---|
| Keep | `UI-Venus-1.5-8B` | 현재 primary로 충분히 강하고, 신규 9B 후보와 공개 점수 차이가 작음 |
| Keep | `MAI-UI-8B` | crop/zoom refined point sidecar라는 repo 역할에 맞음 |
| Keep as alternate | `UI-TARS-1.5-7B` | agent-style 대안 경로로 의미 있음 |
| Benchmark first | `VISTA-9B` | 최신 9B GUI grounding 후보. SSPro `69.2` |
| Benchmark for low-cost | `VISTA-4B` | SSPro `64.2`; latency/VRAM 절감 후보 (이름은 4B지만 HF card는 `5B params`) |
| Add as OCR/doc candidate | `dots.ocr` | 문서/표/reading order에서 GOT-OCR 대비 강함 |
| Evaluate as reasoning sidecar | `Step3-VL-10B`, `MiniCPM-V 4.5` | 설명, audit, OCR/reasoning에는 강하지만 GUI primary는 아님 |
| Watch | `Kimi-VL-A3B`, `InternVL3.5-8B` | 모델은 의미 있지만 현 운영/GUI 목적에는 우선순위 낮음 |

## 9. Sources

- Current repo baseline: [`docs/project_progress/01_vlm_deployment.md`](../project_progress/01_vlm_deployment.md), [`deploy_vlms/config/models/`](../../deploy_vlms/config/models/), [`docs/setup_vlms/04-operations-integration-and-benchmarking.md`](./04-operations-integration-and-benchmarking.md)
- UI-Venus 1.5: [`inclusionAI/UI-Venus-1.5-8B`](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B), [`UI-Venus collection`](https://huggingface.co/collections/inclusionAI/ui-venus)
- VISTA: [`inclusionAI/VISTA-9B`](https://huggingface.co/inclusionAI/VISTA-9B), [`inclusionAI/VISTA-4B`](https://huggingface.co/inclusionAI/VISTA-4B)
- UI-TARS: [`ByteDance-Seed/UI-TARS-1.5-7B`](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- Qwen3-VL: [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [`Qwen/Qwen3-VL-8B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
- Step3-VL: [`stepfun-ai/Step3-VL-10B`](https://huggingface.co/stepfun-ai/Step3-VL-10B)
- MiniCPM-V 4.5: [`openbmb/MiniCPM-V-4_5`](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- Kimi-VL: [`moonshotai/Kimi-VL-A3B-Instruct`](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)
- InternVL3.5: [`OpenGVLab/InternVL3_5-8B`](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
- PaddleOCR-VL: [`PaddlePaddle/PaddleOCR-VL`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- GOT-OCR: [`stepfun-ai/GOT-OCR-2.0-hf`](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- dots.ocr: [`rednote-hilab/dots.ocr`](https://huggingface.co/rednote-hilab/dots.ocr)
