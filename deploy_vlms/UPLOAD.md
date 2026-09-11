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
목적지 루트와 인증 키는 `deploy_vlms/config/site.env` 의 `MODEL_ROOT` · `VLLM_API_KEY` 를
그대로 쓴다(flask_api 가 import 시점에 읽는다). 업로드 전용 설정은 따로 없다.

```bash
python index.py     # 또는 기존 WSGI 기동 방식 그대로
```

| env | 기본값 | 뜻 |
|---|---|---|
| `MODEL_ROOT` | (site.env) | 업로드 목적지 루트 = vLLM 이 읽는 루트. **이 밖으로는 절대 쓰지 않는다** |
| `VLLM_API_KEY` | (빈값 = 인증 없음) | 설정하면 모든 업로드 요청에 `X-Upload-Token` (또는 `Authorization: Bearer`) 필요. vLLM·프록시와 같은 키 |
| `MODEL_UPLOAD_MAX_CHUNK_MB` | 64 | 청크 하나의 상한 |
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

## 올리기 전에: 목적지가 진짜 PVC 인가

**업로드는 200 인데 나중에 `start_all.py` 가 "디렉토리 없음" 으로 죽는 사고**가 있었다
(2026-09-05. pod 재생성으로 풀렸고, 1시간을 기다려도 안 풀렸으므로 NFS 속성 캐시는
아니었다 - `acdirmax` 기본이 60초다). 원인은 전부 **마운트 계층**에 있다:
업로드를 받은 쪽과 모델을 읽는 쪽이 서로 다른 것을 본다.

그래서 **아무것도 올리기 전에** 목적지를 확인한다. 48GB 를 올린 뒤 기동에서
발견하는 것보다 지금 아는 편이 압도적으로 싸다.

```bash
curl -s http://<서버>:<포트>/api/model_upload/health | python3 -m json.tool
```

`dest_mount` 를 본다:

| `fs_type` / 값 | 판정 | 조치 |
|---|---|---|
| `nfs` / `nfs4`, `mount_point` 가 기대한 PVC 경로 | **정상** | 올려도 된다 |
| `overlay` (= `container_layer: true`) | **위험** | PVC 가 안 붙었다. 컨테이너 쓰기 레이어에 쓰는 중이라 **pod 이 재생성되면 가중치가 통째로 사라진다.** 올리지 말 것 |
| `mount_point` 가 `/` | 볼륨 미마운트 | pod 스펙 확인 |
| `propagation` 이 `private` | 잠재 위험 | pod 시작 **후** 호스트에 새로 붙은 마운트는 이 pod 에 영영 안 보인다. 이번 사고의 유력 원인 |
| `available: false` | 리눅스가 아님 | 개발 노트북. 진단만 비고 업로드는 정상 동작 |

`/proc/self/mountinfo` 를 쓰는 이유는 pod 안에서 **특권 없이** 읽을 수 있는 유일한
실마운트 증거이기 때문이다. `mount` 명령은 컨테이너에 없을 수 있고 `/etc/mtab` 은
호스트 것이 섞인다.

### PVC 를 쓸 때 확인할 세 가지

`MODEL_ROOT`(`site.env`) 를 PVC 경로로 바꾸는 것만으로는 절반이다. 업로드 목적지와 vLLM 이
읽는 루트는 둘 다 `MODEL_ROOT` 라 설정상으로는 갈리지 않지만, 아래는 여전히 확인해야 한다.

1. **AccessMode.** 업로드 pod 과 vLLM pod 이 다르면 `ReadWriteMany` 여야 한다.
   `ReadWriteOnce` 면 둘이 **같은 pod** 이어야 한다.
   `kubectl get pvc <이름> -o jsonpath='{.spec.accessModes}'`
2. **`subPath` 를 쓰지 말 것.** 마운트 시점에 경로가 해석되어 "pod 시작 후 생긴 것이
   안 보임" 함정의 단골이다. PVC 를 통째로 마운트하고 앱이 하위 디렉토리를 쓰게 한다.
3. **staging 은 항상 `<root>/.upload_staging` 이다** (오버라이드 없음). 같은 파일시스템이어야
   `os.replace` 가 원자적이고, 다르면 `EXDEV` 로 실패하기 때문이다.

용량도 미리 본다 - 27B BF16 이 ~48GB 다.
`kubectl get pvc <이름> -o jsonpath='{.status.capacity.storage}'`

## 클라이언트 (로컬 PC)

**인자는 `upload_model.py` 상단 상수 블록을 고쳐 쓴다** (셸 env 는 1회성 override).

```python
BASE_URL = "http://<서버>"   # 이미 채워져 있다
SRC = r"C:/models/MAI-UI-8B" # 올릴 폴더/파일
TOKEN = ""                   # 서버 site.env 의 VLLM_API_KEY 와 같은 값. 서버가 인증을 안 쓰면 빈 문자열
CHUNK_MB = None              # None = 기본값 32
```

```bash
python deploy_vlms/scripts/upload_model.py
```

우선순위는 **실제 셸 env > 파일 상수 > 코드 기본값**이고, env 에 밀려 무시된 상수는
콘솔에 찍힌다. 아래 표의 env 이름은 그 override 용으로 그대로 유효하다.

폴더면 재귀로 전부, 단일 파일이면 그것만 올린다. 숨김 파일/폴더(`.git`, `.cache`)는 건너뛴다.

| env | 기본값 | 뜻 |
|---|---|---|
| `MODEL_UPLOAD_URL` | (필수) | Flask 서버 base URL |
| `MODEL_UPLOAD_SRC` | (필수) | 올릴 로컬 폴더 또는 파일 |
| `MODEL_UPLOAD_DEST` | 소스 폴더명 | 서버 루트 아래 목적지 경로 |
| `VLLM_API_KEY` | (빈값) | 서버가 요구하면 필수. 서버 site.env 와 같은 값 |
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
| `HTTP 401` | `VLLM_API_KEY` 불일치 (서버 site.env 와 대조) |
| `HTTP 400 PathNotAllowed` | `MODEL_UPLOAD_DEST` 가 루트를 벗어난다 |
| `HTTP 422` 반복 | 청크가 계속 손상돼 도착한다. 네트워크 경로를 의심 |
| `완료 응답을 못 받았습니다` 반복 | `proxy_read_timeout` 이 재해싱 시간보다 짧다. nginx 를 못 고치면 `MAX_RETRIES` 를 더 올린다(기본 12회=~3분, 백오프 상한 30s) |
| 완료 시 `ChecksumMismatch` | 청크는 다 통과했는데 조립 결과가 다르다 = 디스크 의심. 서버가 `.part` 를 버리고 0 부터 다시 받는다 |

### 업로드는 성공했는데 런처가 파일을 못 찾을 때

`/complete` 응답의 `verification` 을 먼저 본다. `os.replace` **이후의 목적지**를
되읽은 결과다 - 청크 해시와 전체 해시는 `.part` 를 읽은 것이라 다른 것을 잡는다.

```
해시 검증  -> 전송/조립 중 손상
verification -> 마운트 어긋남(ESTALE), 옮긴 결과가 이 프로세스에 안 보이는 상태
```

`stat` 만으로는 캐시된 속성으로 답할 수 있어서 **1바이트를 실제로 read** 한다.
여기서 실패하면 업로드가 `DestinationUnverified`(500)로 떨어지므로, 200 을 받았다면
적어도 **업로드 프로세스의 시선에서는** 파일이 실재한다.

그런데도 런처가 못 찾으면 두 프로세스가 다른 마운트를 보는 것이다.
`verification.mount` 와 런처 쪽을 대조한다:

```bash
python deploy_vlms/scripts/diagnose_paths.py   # 런처 자신의 시선
```

`ls` 결과가 원인을 가른다:

| `ls ${MODEL_ROOT}/<model>/` | 원인 | pod 안에서 고칠 수 있나 |
|---|---|---|
| 디렉토리는 있는데 **비어 있음** | mount propagation (`private`) | 불가. pod 재생성 |
| `Stale file handle` / `I/O error` | ESTALE. `os.replace` 로 inode 가 바뀐 뒤 옛 핸들 | 불가. pod 재생성 |
| `fs_type` 이 `overlay` | PVC 미마운트 | 불가. pod 스펙 수정 |
| 정상 출력 | 마운트는 멀쩡. 권한이나 포트 점유 | `chmod a+rX`, `stop_model.py all` |

셋 다 pod 안에서는 못 고친다 - 마운트를 만든 주체가 kubelet 이라 컨테이너에는
`CAP_SYS_ADMIN` 도 대상도 없다. **재시작이 유일한 대응인 게 맞다.** 다만 재시작은
원인을 안 남기므로, 위 표로 어느 것이었는지는 기록해 둘 것. 반복되면 인프라 쪽 일이다.

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
pytest flask_api/model_upload              # 42 (store / routes / 배선 / 목적지 검증)
pytest deploy_vlms/scripts                 # 44 (클라이언트 루프 + 실제 HTTP 왕복 + 기동 가드)
```
