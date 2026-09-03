# 모델 가중치 업로드 (로컬 PC -> 사내 Flask 서버)

code-server 웹 드래그앤드롭이 1GB 근처에서 깨지는 것을 대체한다.
요청 하나에 파일 하나를 통째로 싣지 않고 **청크로 쪼개** 올린다.

| 요구 | 구현 |
|---|---|
| 스트리밍 | 서버는 `request.stream` 을 1MB 블록으로 흘려 `.part` 에 바로 쓴다. 파일이 메모리에 올라오지 않는다 |
| 이어받기 | 서버가 커밋된 offset 을 디스크에 기록. 끊기면 **같은 명령을 다시 실행**하면 그 지점부터 |
| 무결성 | 청크마다 sha256 + 완료 시 파일 전체 sha256 재검증. 통과해야만 목적지로 원자적 이동 |

## 서버 (사내 private cloud)

`flask_api` 에 이미 배선돼 있다. 앱만 띄우면 `/api/model_upload/*` 가 열린다.

```bash
MODEL_UPLOAD_ROOT=/path/to/models \
MODEL_UPLOAD_TOKEN=<공유비밀>                              \
uv run python index.py     # 또는 기존 WSGI 기동 방식 그대로
```

| env | 기본값 | 뜻 |
|---|---|---|
| `MODEL_UPLOAD_ROOT` | `ALLOWED_MODEL_ROOT` env, 없으면 하드코딩된 같은 경로 | 업로드 목적지 루트. **이 밖으로는 절대 쓰지 않는다** |
| `MODEL_UPLOAD_TOKEN` | (빈값 = 인증 없음) | 설정하면 모든 업로드 요청에 `X-Upload-Token` 필요 |
| `MODEL_UPLOAD_MAX_CHUNK_MB` | 64 | 청크 하나의 상한 |
| `MODEL_UPLOAD_STAGING_DIR` | `<root>/.upload_staging` | 받는 중인 `.part` 위치. **반드시 root 와 같은 파일시스템** |
| `MODEL_UPLOAD_ENABLED` | 1 | 0 이면 엔드포인트를 아예 등록하지 않는다 |

도달성 확인:

```bash
curl -s http://<서버>:<포트>/api/model_upload/health
```

## nginx 가 앞에 있을 때 (필수 확인)

**기본 설정 그대로면 세 가지가 순서대로 터진다.** `deploy_vlms/nginx/model_upload.conf`
를 server 블록에 넣고 `nginx -t` 로 검증한 뒤 reload 할 것.

| nginx 기본값 | 증상 | 해결 |
|---|---|---|
| `client_max_body_size 1m` | 청크가 Flask 에 닿기 전에 **413** | `client_max_body_size 128m` |
| `proxy_read_timeout 60s` | `/complete` 의 전체 재해싱 중 **504** (서버는 성공했는데 실패로 보임) | `proxy_read_timeout 900s` |
| `proxy_request_buffering on` | 청크마다 임시파일에 통째로 받아썼다가 전달 - 동작은 하나 디스크 I/O 2배 | `proxy_request_buffering off` (+ `proxy_http_version 1.1`) |

클라이언트는 이 셋에 대해 **스스로 버티도록** 되어 있다 - 근본 해결은 위 설정이지만,
당장 nginx 를 못 고치는 상황에서도 업로드는 끝난다:

- **413 을 만나면 청크를 절반씩 줄인다** (하한 256KB). 왕복이 늘 뿐 실패하지 않는다.
  알아낸 상한은 **세션 전체로 전파**되므로 파일마다 다시 탐색하지 않는다 - 첫 실행이
  찍어 주는 값을 `CHUNK_MB` 상수에 박으면 첫 파일의 탐색도 사라진다.
- **완료 응답이 잘리면(504) 재요청 대신 상태를 먼저 물어본다.** 서버가 이미 끝냈으면
  성공으로 처리한다 - 무작정 `/complete` 를 다시 부르면 같은 재해싱을 반복하게 된다.

`nginx.conf` 를 못 건드리는 경우 `MODEL_UPLOAD_CHUNK_MB` 를 nginx 상한 아래로
직접 맞춰도 된다(예: 상한이 1m 이면 `MODEL_UPLOAD_CHUNK_MB=1`).

## 클라이언트 (로컬 PC)

**인자는 `upload_model.py` 상단 상수 블록을 고쳐 쓴다** (셸 env 는 1회성 override).

```python
BASE_URL = "http://<서버>"   # 이미 채워져 있다
SRC = r"C:/models/MAI-UI-8B" # 올릴 폴더/파일
TOKEN = ""                   # 서버가 토큰을 안 쓰면 빈 문자열
CHUNK_MB = None              # None = 기본값 32
```

```bash
uv run python deploy_vlms/scripts/upload_model.py
```

우선순위는 **실제 셸 env > 파일 상수 > 코드 기본값**이고, env 에 밀려 무시된 상수는
콘솔에 찍힌다. 아래 표의 env 이름은 그 override 용으로 그대로 유효하다.

폴더면 재귀로 전부, 단일 파일이면 그것만 올린다. 숨김 파일/폴더(`.git`, `.cache`)는 건너뛴다.

| env | 기본값 | 뜻 |
|---|---|---|
| `MODEL_UPLOAD_URL` | (필수) | Flask 서버 base URL |
| `MODEL_UPLOAD_SRC` | (필수) | 올릴 로컬 폴더 또는 파일 |
| `MODEL_UPLOAD_DEST` | 소스 폴더명 | 서버 루트 아래 목적지 경로 |
| `MODEL_UPLOAD_TOKEN` | (빈값) | 서버가 요구하면 필수 |
| `MODEL_UPLOAD_CHUNK_MB` | 32 | 서버 상한보다 크면 자동으로 줄인다 |
| `MODEL_UPLOAD_MAX_RETRIES` | 12 | 청크 하나당 재시도 한도 (지수 백오프, 최대 30s) |

### 끊겼을 때

**같은 명령을 다시 실행하면 된다.** 이미 올라간 파일은 건너뛰고, 진행 중이던 파일은
서버가 확실히 받은 지점부터 이어간다. 실측: 1.2GB 업로드를 중간에 kill -9 한 뒤
재실행 -> 남은 352MB 만 전송, 최종 sha256 원본 일치.

### 자주 나오는 실패

| 증상 | 원인 / 조치 |
|---|---|
| `HTTP 413` 경고 후 계속 진행 | 프록시 `client_max_body_size` 가 청크보다 작아 클라이언트가 청크를 반씩 줄인 것. 동작은 하지만 nginx 를 고치는 게 낫다 |
| `HTTP 401` | `MODEL_UPLOAD_TOKEN` 불일치 |
| `HTTP 400 PathNotAllowed` | `MODEL_UPLOAD_DEST` 가 루트를 벗어난다 |
| `HTTP 422` 반복 | 청크가 계속 손상돼 도착한다. 네트워크 경로를 의심 |
| `완료 응답을 못 받았습니다` 반복 | `proxy_read_timeout` 이 재해싱 시간보다 짧다. nginx 를 못 고치면 `MAX_RETRIES` 를 더 올린다(기본 12회=~3분, 백오프 상한 30s) |
| 완료 시 `ChecksumMismatch` | 청크는 다 통과했는데 조립 결과가 다르다 = 디스크 의심. 서버가 `.part` 를 버리고 0 부터 다시 받는다 |

### `.upload_staging` 를 손으로 지우지 말 것

`.part` 는 받는 중인 실파일이라 크지만, 같이 있는 `.json` 은 수백 바이트짜리 상태
파일이고 **"이 파일은 이미 다 올렸다"는 유일한 기록**이다. 이걸 지우면 다음 실행이
멀쩡히 올라간 파일까지 처음부터 다시 올린다. 정리는 아래 DELETE 로 할 것.

### 잘못 올린 세션 정리

`.upload_staging` 에 `.part` 가 남아 디스크를 먹으면:

```bash
curl -X DELETE -H "X-Upload-Token: <비밀>" \
  http://<서버>:<포트>/api/model_upload/sessions/<upload_id>
```

## 테스트

전부 Mac 에서 실서버 없이 돈다 (마지막 것만 로컬에 임시 서버를 띄운다).

```bash
uv run pytest flask_api/model_upload              # 36 (store / routes / 배선)
uv run pytest deploy_vlms/scripts                 # 21 (클라이언트 루프 + 실제 HTTP 왕복)
```
