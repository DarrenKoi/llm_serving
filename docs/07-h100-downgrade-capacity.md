# H100 Downgrade Capacity Check

이 문서는 현재 H200 2장 기준으로 운영 중인 VLM 스택을 H100으로 낮출 때의 정적 용량 판단을 정리한다.

## 1. 결론

동일한 동시성이 필요 없고, 입력 이미지가 보통 1MB 이하이며, 빠른 단건 응답이 목표라면 **H100 80GB 2장으로 충분할 가능성이 높다**.

권장 판단:

| 목표 | 필요한 H100 | 판단 |
|------|-------------|------|
| 5개 VLM/OCR 서비스를 모두 띄우고 저동시성으로 운영 (현재 설정) | H100 2장 | 권장 최소 구성 — 카드당 ~30-40GB 사용, 여유 큼 (2.1장 산정) |
| H200 운영과 비슷한 동시성 유지 | H100 3장 | 안정 여유 확보용 |
| H100 1장에 5개 서비스를 모두 상시 적재 | H100 1장 | 비권장 |

현재 요구 조건에서는 **H100 2장**을 기준으로 잡는 것이 현실적인 최소선이다.

## 2. 현재 VLM 스택

현재 배포 대상 서비스는 다음 5개다.

| 서비스 | 파라미터 | 역할 |
|--------|----------|------|
| `ui-venus` | 8.3B | primary full-screen grounding |
| `mai-ui` | 8B | crop retry / refined point |
| `ui-tars` | 7.6B | alternate GUI agent |
| `paddleocr-vl-1.5` | 0.9B | OCR/layout sidecar |
| `got-ocr` | 0.58B | OCR fallback |

현재 H200 기준 배치는 다음과 같다.

```text
GPU 0: UI-Venus + UI-TARS
GPU 1: MAI-UI + PaddleOCR-VL + GOT-OCR
```

H100 2장에서도 같은 배치가 가장 단순하다.

```text
H100 GPU 0: UI-Venus + UI-TARS
H100 GPU 1: MAI-UI + PaddleOCR-VL + GOT-OCR
```

### 2.1 현재 구성 정적 VRAM 산정 (왜 정확히 2장인가)

가정: BF16 = 2 byte/param. 가중치 합산은 다음과 같다.

| 서비스 | 파라미터 | BF16 가중치 |
|--------|----------|-------------|
| `ui-venus` | 8.3B | ~16.6 GB |
| `ui-tars` | 7.6B | ~15.2 GB |
| `mai-ui` | 8B | ~16.0 GB |
| `paddleocr-vl-1.5` | 0.9B | ~1.8 GB |
| `got-ocr` | 0.58B | ~1.2 GB |
| 합계 | - | **~50.8 GB** |

가중치 외 프로세스당 추가분: CUDA context(~1-2GB), vision encoder activation, KV cache(저동시성이면 작음), fragmentation. `GOT-OCR`은 vLLM이 아니라 transformers 런타임이라 `mai-ui.env` 주석 기준 ~4GiB로 잡는다.

현재 2-GPU split 기준, 저동시성(`MAX_MODEL_LEN=4096`, `MAX_NUM_SEQS=2`) 카드별 실사용 추정:

| GPU | 적재 | 가중치 | +오버헤드 | 카드 실사용(추정) | 80GB 대비 |
|-----|------|--------|-----------|-------------------|-----------|
| GPU 0 | UI-Venus + UI-TARS | ~31.8 GB | ~6-8 GB | **~38-40 GB** | ~50% |
| GPU 1 | MAI-UI + PaddleOCR-VL + GOT-OCR | ~19 GB | ~8-10 GB | **~28-30 GB** | ~37% |

**결론: 현재 설정(5개 서비스, 저동시성)은 H100 80GB 2장이 정확한 기준선이다.** 카드당 30-40GB만 쓰므로 큰 여유가 있고, H200-like(`8192`/`8`)로 올려도 80GB 안에 들어온다. 1장은 가중치 합(~50.8GB)만 보면 들어갈 듯 보이나, 5개 프로세스(vLLM 4 + transformers 1)의 context/KV/vision/fragmentation을 더하면 실사용이 ~65-78GB까지 올라 여유가 없다(7장 참조).

## 3. 왜 2장이 가능한가

입력 이미지가 1MB 이하라는 점은 네트워크 전송과 HTTP 처리에는 유리하다. 다만 GPU 메모리 관점에서 더 중요한 것은 파일 크기보다 다음 항목이다.

- 모델 가중치 크기
- vision encoder가 만드는 image token 수
- `MAX_MODEL_LEN`
- `MAX_NUM_SEQS`
- vLLM KV cache 예약량
- 여러 vLLM 프로세스가 같은 GPU를 공유할 때의 fragmentation

동일 동시성이 필요 없다면 `MAX_NUM_SEQS`를 크게 잡을 이유가 없다. 단건 응답 중심이면 큰 sequence 동시성 예약은 오히려 H100에서 VRAM만 잡아먹는다.

## 4. H100용 권장 설정

H200 설정을 그대로 쓰지 않는다. 특히 `GPU_TOTAL_MEMORY_GIB=140`은 H100에서 잘못된 sizing을 만든다.

H100 80GB 기준 시작 설정:

```env
GPU_TOTAL_MEMORY_GIB=80
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=2
```

운영 중 여유가 있으면 다음 순서로 올린다.

1. `MAX_MODEL_LEN=8192`
2. `MAX_NUM_SEQS=4`

반대로 OOM 또는 vLLM startup failure가 나면 다음 순서로 낮춘다.

1. `MAX_NUM_SEQS`
2. `MAX_MODEL_LEN`
3. `GPU_MEMORY_UTILIZATION`

## 5. 2장 구성의 운영 프로파일

### GPU 0

```text
UI-Venus + UI-TARS
```

두 모델 모두 7-8B급이므로 GPU 0이 가장 빡빡하다. 저동시성 조건에서는 다음 값부터 시작한다.

```env
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=2
AUTO_TUNE_GPU_MEMORY_UTILIZATION=1
COLOCATED_MODELS_PER_GPU=2
```

안정적으로 뜨면 `MAX_MODEL_LEN=8192` 또는 `MAX_NUM_SEQS=4`를 단계적으로 올린다.

### GPU 1

```text
MAI-UI + PaddleOCR-VL + GOT-OCR
```

`PaddleOCR-VL`은 0.9B 모델이고 이미 낮은 sequence 설정이 적합하다. `GOT-OCR`은 vLLM이 아니라 transformers 기반 별도 프로세스이므로, MAI-UI 쪽에서 과도하게 VRAM을 예약하지 않는 것이 중요하다.

H100 2장 저동시성 기준으로는 현재 역할 분담을 유지하되, MAI-UI도 `MAX_NUM_SEQS=2~4` 범위에 맞춘다.

## 6. 3장이 필요한 경우

아래 조건이면 H100 3장을 권장한다.

- H200 때와 비슷한 동시 요청 여유가 필요하다.
- `MAX_MODEL_LEN=8192`, `MAX_NUM_SEQS=8`에 가깝게 유지해야 한다.
- UI-Venus와 UI-TARS를 동시에 자주 호출한다.
- vLLM engine startup 실패 없이 넉넉한 운영 여유가 필요하다.

이때 권장 배치는 다음과 같다.

```text
H100 GPU 0: UI-Venus
H100 GPU 1: UI-TARS
H100 GPU 2: MAI-UI + PaddleOCR-VL + GOT-OCR
```

이 구성에서는 UI-Venus와 UI-TARS의 `COLOCATED_MODELS_PER_GPU`를 `1`로 바꾸는 것이 맞다.

## 7. 1장이 비권장인 이유

5개 모델의 BF16 가중치 합(~50.8GB, 2.1장 표 참조)만 보면 H100 80GB 한 장에 들어갈 것처럼 보일 수 있다. 하지만 실제 상시 서빙에는 다음 메모리가 추가로 필요하며, 다 더하면 실사용이 ~65-78GB까지 올라 80GB 한 장에는 여유가 없다.

- vLLM KV cache
- vision encoder 메모리
- tokenizer/processor runtime 메모리
- CUDA context
- 여러 프로세스 간 fragmentation
- GOT-OCR transformers process

빠른 응답이 목표라면 요청마다 모델을 올렸다 내리는 cold-start 방식도 맞지 않는다. 따라서 1장은 실험용으로만 보고, 운영 기준에서는 제외한다.

## 8. 검증 순서

H100 서버에서 모델 파일이 이미 stage되어 있다는 전제에서 다음 순서로 검증한다.

```bash
uv run python deploy_vlms/scripts/start_all.py
uv run python deploy_vlms/scripts/check_vlm.py
```

개별 vLLM 모델만 단계적으로 확인할 때는 `start_model.py <instance>`를 사용한다. `GOT-OCR`은 `start_all.py`가 백그라운드 프로세스로 함께 띄우는 경로를 우선 사용한다.

최종 ready 판정은 단순 `/v1/models` health가 아니라 실제 workflow에서 쓰는 1MB 이하 스크린샷 요청까지 통과해야 한다.

## 9. 최종 권장안

현재 조건에서는 다음으로 정리한다.

- **구매/할당 기준:** H100 2장
- **운영 목표:** 저동시성, 빠른 warm response
- **초기 설정:** `GPU_TOTAL_MEMORY_GIB=80`, `MAX_MODEL_LEN=4096`, `MAX_NUM_SEQS=2`
- **확장 설정:** 안정화 후 `MAX_MODEL_LEN=8192`, `MAX_NUM_SEQS=4`
- **3장 전환 기준:** H200급 동시성 또는 UI-Venus/UI-TARS 동시 다발 호출이 필요해질 때
