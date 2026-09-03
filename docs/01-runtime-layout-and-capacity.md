# Runtime Layout And Capacity

This document merges the old layout, multi-size, offline policy, host RAM, and vLLM runtime background notes into one runtime reference.

## 1. Baseline Assumptions

- GPU server: `H200 140GB x 2`
- OS: Linux
- serving runtime: dedicated Python environment with `vLLM`
- model weights: already staged under the cloud server's local `data/models/`
- office environment: offline or tightly restricted outbound access

`vLLM` is the serving engine, not the training framework. Fine-tuning belongs to the Hugging Face / TRL / PEFT / Unsloth side; serving belongs here.

## 2. Recommended Layout

```text
/project/.../data/models/
  UI-Venus-1.5-8B/
  MAI-UI-8B/
  UI-TARS-1.5-7B/
  PaddleOCR-VL-1.5/
  GOT-OCR-2.0-hf/

deploy_vlms/
  scripts/
  config/
    common.env
    models/
      ui-venus.env
      mai-ui.env
      ui-tars.env
      paddleocr-vl-1.5.env
      got-ocr-2.0-hf.env
```

Guiding rule:

- model files stay in `data/models/`
- shared runtime knobs stay in `common.env`
- per-service port/GPU/alias settings stay in `models/*.env`

## 3. Naming And Port Rules

- keep stable instance names such as `ui-venus`, `mai-ui`, `ui-tars`
- keep stable served aliases such as `ui-venus-1.5-8b`
- reserve `8001+` for live services
- use a separate canary port instead of overwriting a working port

For size variants, use `family-size` naming:

- `ui-venus-2b`
- `ui-venus-7b`
- `ui-venus-30b`

## 4. Core Settings

Useful shared defaults:

```bash
HOST=127.0.0.1
DTYPE=bfloat16
GPU_MEMORY_UTILIZATION=0.80
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=8
TENSOR_PARALLEL_SIZE=1
API_KEY=
```

Important per-model keys:

- `MODEL_ID`
- `SERVED_MODEL_NAME`
- `PORT`
- `GPU_ID`
- `TRUST_REMOTE_CODE`
- `CHAT_TEMPLATE`
- `LIMIT_MM_PER_PROMPT`
- `EXTRA_VLLM_ARGS`

## 5. vLLM Runtime Concepts That Matter

### 5.1 Tokenizer And Processor Are Part Of Runtime

The server does not only load weights. It also needs the correct:

- tokenizer
- processor
- chat template
- preprocessor configuration

This matters especially for multimodal models such as `UI-TARS`.

### 5.2 Why vLLM Feels Fast

Important runtime features:

- `PagedAttention`: better KV-cache memory management
- continuous batching: keeps the GPU busy across incoming requests
- prefix caching: reuses shared prompt prefixes
- chunked prefill: helps long-context prefill workloads

In this repo's workload, prefix caching is useful because many requests share long fixed prompt instructions while only the image and a small amount of context change.

### 5.3 vLLM Is For Serving, Not Fine-Tuning

If a model needs adaptation:

- fine-tune with `Transformers + TRL + PEFT` or `Unsloth`
- then either merge the adapter or serve the LoRA path separately
- use `vLLM` for the final inference endpoint

## 6. Capacity Planning

When colocating small models, do not guess `GPU_MEMORY_UTILIZATION`.

Practical rule:

`u_recommended = ((M_gpu - M_shared) / N_models - M_proc) / M_gpu`

Typical H200 starting points:

- two 8B-class models on one GPU: about `0.44`
- three 8B-class models on one GPU: about `0.29`

If automatic sizing fails:

1. reduce `MAX_NUM_SEQS`
2. reduce `MAX_MODEL_LEN`
3. only then consider more aggressive runtime flags

Useful knobs:

```bash
AUTO_TUNE_GPU_MEMORY_UTILIZATION=1
COLOCATED_MODELS_PER_GPU=2
GPU_SHARED_RESERVE_GIB=8
GPU_PROCESS_RESERVE_GIB=4
```

## 7. Host RAM Matters

Large GPU VRAM does not remove the need for host RAM.

Host RAM is still used for:

- Python and vLLM processes
- tokenizer/processor loading
- safetensor metadata and shard handling
- request buffers and CPU-side staging memory
- Flask/proxy/logging side services

Typical warning signs:

- `EngineCore ... died unexpectedly`
- `AsyncLLM output_handler failed`
- API server exits before first successful request
- kernel OOM entries in `dmesg`

First commands to check:

```bash
free -h
dmesg -T | tail -n 100
ps aux --sort=-%mem | head
tail -n 200 deploy_vlms/runtime/logs/<instance>.log
```

## 8. Multi-Size Variants

Use the generic scripts when comparing families by size:

```bash
python deploy_vlms/scripts/prepare_variant_envs.py ui-venus
python deploy_vlms/scripts/start_model.py ui-venus 2b
python deploy_vlms/scripts/start_model.py ui-venus 30b
```

Use model-specific overrides for:

- `PORT`
- `GPU_ID`
- `TENSOR_PARALLEL_SIZE`
- `GPU_MEMORY_UTILIZATION`
- `MAX_MODEL_LEN`
- `MAX_NUM_SEQS`
- `EXTRA_VLLM_ARGS`

## 9. Offline Policy

- do not depend on live Hugging Face pulls on the office server
- stage models and required assets ahead of time
- disable telemetry and usage reporting
- strip proxy settings unless a controlled internal route is required
- prefer absolute local paths over repo-relative guesses
