# nginx location 블록

`server { }` 안에 넣거나 `include` 로 끌어 쓴다. `proxy_pass` 포트는 uWSGI 가 실제로
듣는 곳(`../uwsgi/uwsgi.ini` 의 `http-socket`)과 맞춰야 한다.

| 파일 | 덮는 경로 | 이게 없으면 |
|---|---|---|
| `model_upload.conf` | `/api/model_upload/` | 413(기본 1m) → 504(재해싱 중) → 청크마다 임시파일 |
| `vlm_serve.conf` | `/api/vlm_serve/` | thinking 이 긴 요청이 60s 에서 504 |

**nginx 를 직접 못 고치는 배포라면** 이 두 파일은 관리자에게 요청할 내용 그 자체다.
숫자마다 "없으면 무엇이 터지는가"를 주석에 달아 둔 이유가 그것이다.

타임아웃은 바깥일수록 길어야 안쪽이 먼저 끊고 제대로 된 오류를 만든다:

```
nginx proxy_read_timeout  >  uWSGI harakiri(870)  >  앱 자신의 타임아웃
      600s / 900s                                    VLM_SERVE_READ_TIMEOUT_SEC=300
```

거꾸로면 앱은 성공했는데 호출자는 504 를 본다.
