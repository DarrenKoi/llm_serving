# study

이 저장소를 운용하는 데 필요한 **배경 개념** 문서. 절차와 설정값은 `docs/01`~`09`
와 각 `.env` 파일이 정본이고, 여기는 그것을 읽을 수 있게 만드는 쪽이다.

## 웹 서빙 스택 (nginx / uWSGI / Flask)

Flask 코드는 짜 봤지만 그 아래 레이어는 모르는 상태를 전제로 쓰였다.
순서대로 읽는 것을 권한다.

| | 문서 | 내용 |
|---|---|---|
| 00 | [`00-why-nginx-and-wsgi-exist.md`](00-why-nginx-and-wsgi-exist.md) | CGI → mod_python → WSGI, C10K → nginx, Apache 의 자리. 왜 레이어가 셋인가 |
| 01 | [`01-nginx-uwsgi-flask.md`](01-nginx-uwsgi-flask.md) | 각 레이어의 역할, 인바운드/아웃바운드, 타임아웃 사슬, 증상별 진단표 |
| 02 | [`02-load-balancing.md`](02-load-balancing.md) | 밸런싱 알고리즘, 헬스체크, LLM 서빙에서 통념이 깨지는 지점 |

## vLLM / 모델 서빙 개념

| | 문서 | 내용 |
|---|---|---|
| 03 | [`03-safetensors-and-model-formats.md`](03-safetensors-and-model-formats.md) | `.bin`(pickle) 대신 safetensors 를 쓰는 이유 - 코드 실행 경로 제거 + mmap lazy load |
| - | [`../08-serving-knobs-concepts.md`](../08-serving-knobs-concepts.md) | BF16/FP8 등 숫자 형식이 실제로 뭘 하는지, `max_model_len`/`max_num_seqs`/`gpu_memory_utilization`, prefill/decode 병목 차이 - 이미 정본급으로 두꺼워 여기서 따로 안 만든다 |
