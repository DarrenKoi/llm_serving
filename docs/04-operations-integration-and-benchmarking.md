# Operations, Integration, And Benchmarking

> [2026-09-03] ui-venus / ui-tars / got-ocr 은 가중치와 기동 스크립트를 모두 삭제했다.
> 아래 그 세 모델 관련 절차는 역사 기록이며, 실행하면 파일이 없다.
> 현재 서빙은 mai-ui / paddleocr-vl-1.5 / qwen3.8-27b 셋뿐이다.

This document merges the old operations map, coworker setup map, repo integration notes, and the benchmark-oriented model-role guidance.

## 1. Daily Operations

Typical commands:

```bash
cd /path/to/deploy_vlms

python scripts/start_model.py mai-ui
python scripts/check_vlm.py http://127.0.0.1:8002 mai-ui-8b
tail -f runtime/logs/mai-ui.log
python scripts/stop_model.py ui-venus
```

Useful checks:

- `curl http://127.0.0.1:<port>/v1/models`
- `ss -ltnp | grep 800`
- `tail -n 200 runtime/logs/<instance>.log`

## 2. Endpoint Patterns

### 2.1 Direct Port

- `http://vlm-host.internal:8001`
- `http://vlm-host.internal:8002`
- `http://vlm-host.internal:8003`

### 2.2 Flask Proxy

- `http://vlm-host.internal/api/vlm_serve/ui-venus`
- `http://vlm-host.internal/api/vlm_serve/mai-ui`
- `http://vlm-host.internal/api/vlm_serve/ui-tars`

Coworker-facing automation should prefer the proxy-style contract.

## 3. Repo Integration Contract

### 3.1 `poc/work2`

Current code uses:

- `poc/work2/flask_vlm.py` for service registry
- `poc/work2/vlm_client.py` for service-slug based calls
- `poc/work2/connection_check.py` for live service discovery

Legacy `poc/work` has been removed after the migration to `work2`.

Important rule:

- `poc/work2` must stay client-side independent
- it should not import server-side `flask_api` code or depend on server env files

## 4. Model Roles For Benchmarking

| Role | Preferred service |
|------|-------------------|
| primary GUI grounding | `ui-venus` |
| alternate primary comparison | `ui-tars` |
| crop retry sidecar | `mai-ui` |
| default OCR | `paddleocr-vl-1.5` |
| hard OCR fallback | `got-ocr` |

Optional external baselines such as `Kimi-K2.5` or `Qwen3-VL-30B-Instruct` can still be useful, but they should be compared under the same screenshot and task set.

## 5. Benchmark Order

Recommended order:

1. compare primary full-screen models first
2. keep sidecars off during the first head-to-head
3. add `MAI-UI` crop retry to the better primary
4. add `PaddleOCR-VL-1.5` only for text-heavy cases
5. add `GOT-OCR` only for unresolved OCR failures

Why:

- it separates primary-model quality from sidecar effects
- it keeps latency analysis interpretable
- it avoids building an always-call-everything pipeline

## 6. What To Measure

Use a consistent comparison table:

- `element hit rate`
- `click-point drift(px)`
- `retry count`
- `step completion rate`
- `small-text OCR recall`
- `latency`
- `sidecar escalation rate`

## 7. Troubleshooting Checklist

### 7.1 Health Works But Quality Is Bad

- compare on the same prompt and same screenshot set
- do not change model and prompt at the same time
- confirm the served alias is what the client expects

### 7.2 Connection Works Only On Direct Port

- verify the Flask proxy service slug mapping
- verify `/api/vlm_serve/<service>/v1/models`
- verify `poc/work2/flask_vlm.py` matches the proxy contract

### 7.3 Service Starts But Coworkers Still Fail

- run `python poc/work2/connection_check.py`
- test the exact slug used by the task script
- verify the proxy URL rather than only the direct port
