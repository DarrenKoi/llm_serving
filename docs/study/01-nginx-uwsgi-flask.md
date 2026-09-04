# nginx / uWSGI / Flask: 요청 하나가 지나가는 길

> 작성 2026-09-05. **Flask 코드만 짜 본 사람**을 대상으로, 그 코드가 실제로 어떤
> 구조 위에서 돌고 있는지를 설명하는 개념 문서다. 이 저장소의 `flask_api/` 를
> 예시로 쓰지만 내용은 사내 private cloud 의 nginx + uWSGI + Flask 스택 일반에
> 적용된다. 로드밸런싱 원리는 `02-load-balancing.md` 로 이어진다.

---

## 0. 한 장의 그림

브라우저가 `POST /api/vlm_serve/qwen3.8-27b/v1/chat/completions` 를 던졌을 때,
그 요청이 실제로 지나가는 길이다.

```
  [브라우저 / OpenAI 클라이언트]
        │  ① HTTP  (TCP:443)
        ▼
  ┌───────────────────────────────────────┐
  │ nginx            :80 / :443           │  ← 인터넷을 마주보는 유일한 프로세스
  │  · TLS 종료   · 정적파일 직접 응답     │
  │  · 라우팅     · 타임아웃/바디크기 제한 │
  └───────────────────────────────────────┘
        │  ② uwsgi 프로토콜 (unix socket 또는 TCP:5000)
        ▼
  ┌───────────────────────────────────────┐
  │ uWSGI            (master 프로세스)     │  ← 파이썬을 몇 벌 띄울지 관리
  │   ├── worker 1  ┐                     │
  │   ├── worker 2  ├ 전부 같은 app 객체   │
  │   └── worker 3  ┘  (각자 별개 메모리)  │
  └───────────────────────────────────────┘
        │  ③ WSGI 함수 호출 (파이썬 함수 호출. 네트워크 아님)
        ▼
  ┌───────────────────────────────────────┐
  │ Flask            app = Flask(__name__) │  ← 여러분이 짜는 곳
  │   @bp.route(...) def chat(): ...       │
  └───────────────────────────────────────┘
        │  ④ requests.post(...)   ← 여기서부터가 "아웃바운드"
        ▼
  ┌───────────────────────────────────────┐
  │ vLLM             127.0.0.1:8006        │
  └───────────────────────────────────────┘
```

핵심은 **①②③ 이 서로 다른 종류의 경계**라는 것이다.

- ① 은 **HTTP over TCP**. 남의 컴퓨터에서 온다. 신뢰할 수 없다.
- ② 도 네트워크지만 **같은 장비 안**이고 프로토콜이 HTTP 가 아니다 (uwsgi 바이너리 프로토콜).
- ③ 은 **네트워크가 아니다**. 그냥 파이썬 함수 호출이다.

이 셋을 구분하면 "왜 레이어가 세 개나 되나" 가 대부분 풀린다.

---

## 1. Flask 는 서버가 아니다

가장 먼저 깨야 할 오해. `app.run()` 을 쓰니까 Flask 가 웹서버처럼 보이지만
**Flask 는 서버가 아니라 함수 하나**다.

WSGI(Web Server Gateway Interface, PEP 3333)는 파이썬 웹앱의 규격인데 내용이
놀랄 만큼 단순하다. "요청 정보가 담긴 dict 와 콜백 하나를 받아서, 응답 바디를
내놓는 호출 가능한 객체" 가 전부다.

```python
def application(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"hello"]
```

이게 완전한 WSGI 앱이다. Flask 의 `app` 객체도 결국 이 시그니처를 만족하는
객체일 뿐이다. 이 저장소의 `index.py` 가 그걸 그대로 보여준다:

```python
from web_main import app as application     # ← uWSGI 가 찾아 쓸 이름
```

`application` 이라는 이름은 관례다. uWSGI 설정의 `module = index:application` 이
이 이름을 가리킨다. 이름을 바꾸면 uWSGI 가 앱을 못 찾는다.

### 그럼 `app.run()` 은 뭔가

Flask 가 개발 편의를 위해 끼워 넣은 **장난감 서버**(Werkzeug dev server)다.
`index.py` 의 `if __name__ == "__main__":` 블록이 그것이고 로컬에서
`python index.py` 로 :5000 을 띄울 때만 쓰인다.

운영에 쓰면 안 되는 이유:

| | dev server | uWSGI |
|---|---|---|
| 동시성 | 기본 1요청씩 (스레드 옵션 있어도 빈약) | 워커 프로세스 N개 |
| 크래시 복구 | 없음. 죽으면 끝 | master 가 워커를 재기동 |
| 느린 클라이언트 | 하나가 붙잡으면 전체가 멈춤 | nginx 가 앞에서 흡수 |
| 성능 | 순수 파이썬 | C 로 구현, 훨씬 빠름 |

> **★ 요점** — "Flask 앱을 배포한다" 는 말은 정확히는 **"내 WSGI 앱 객체를
> uWSGI 라는 서버에게 넘겨준다"** 는 뜻이다. Flask 는 코드고, 서버는 uWSGI 다.

---

## 2. uWSGI 가 하는 일: 파이썬을 여러 벌 띄운다

파이썬에는 GIL(Global Interpreter Lock)이 있어서 **한 프로세스 안에서는 파이썬
바이트코드가 동시에 한 줄씩만** 실행된다. CPU 코어가 16개여도 파이썬 프로세스
하나는 그중 1개만 쓴다.

해법이 단순하다. **프로세스를 여러 개 띄우면 된다.** 그게 uWSGI 의 본업이다.

전형적인 `uwsgi.ini`:

```ini
[uwsgi]
module = index:application      ; ← 어떤 WSGI 객체를 띄울지

master = true                   ; ← 워커를 감시/재기동하는 관리 프로세스
processes = 4                   ; ← 파이썬 인터프리터를 4벌 fork
threads = 2                     ; ← 각 워커 안에 스레드 2개

socket = /run/uwsgi/app.sock    ; ← nginx 가 붙을 자리
chmod-socket = 660

harakiri = 300                  ; ← 300초 넘게 잡고 있는 요청은 워커째 죽인다
max-requests = 1000             ; ← 1000요청 처리하면 워커를 새로 fork (메모리 누수 방어)
vacuum = true                   ; ← 종료 시 소켓 파일 정리
```

### processes 와 threads 의 차이

```
processes = 4, threads = 2  →  동시에 처리 가능한 요청 = 8

  worker 1 [메모리 독립] ── thread A ── thread B
  worker 2 [메모리 독립] ── thread A ── thread B
  worker 3 [메모리 독립] ── thread A ── thread B
  worker 4 [메모리 독립] ── thread A ── thread B
```

- **processes**: 진짜 병렬. 각자 별개 메모리 공간. CPU 코어를 실제로 나눠 쓴다.
- **threads**: 한 워커 안에서 메모리 공유. GIL 때문에 CPU 작업은 병렬이 안 되지만
  **I/O 대기 중에는 GIL 을 놓는다.** DB 응답을 기다리거나 `requests.post()` 가
  네트워크를 기다리는 동안 다른 스레드가 돈다.

이 저장소처럼 **거의 모든 시간을 upstream 대기로 보내는 프록시**는 스레드가
효율적이다. 반대로 이미지 리사이즈처럼 CPU 를 태우는 앱은 프로세스를 늘려야 한다.

> **함정 — 프로세스는 상태를 공유하지 않는다.**
> 전역 변수, 모듈 레벨 카운터, 인메모리 캐시, `itertools.count()` — 전부 워커마다
> **별개의 사본**이다. 워커 1에서 올린 카운터를 워커 2가 볼 수 없다.
> "가끔만 이상하게 동작하는 버그" 의 단골 원인이고, 뒤의 로드밸런싱 이야기에서
> 다시 나온다. 공유가 필요하면 Redis 나 DB 같은 **프로세스 밖**으로 나가야 한다.

### harakiri 를 꼭 걸어라

`harakiri = 300` 은 "300초 넘게 붙잡고 있는 요청은 워커를 통째로 죽여서 회수한다"
는 뜻이다. 이게 없으면 upstream 이 응답을 안 줄 때 워커가 영원히 묶이고
4개가 다 묶이면 **앱 전체가 멈춘다.** LLM 프록시처럼 응답이 긴 앱은 이 값을
upstream 타임아웃보다 넉넉히 크게 잡아야 한다 — 이 저장소는
`VLM_SERVE_READ_TIMEOUT_SEC` 기본값이 300초다.

---

## 3. nginx 가 하는 일

uWSGI 가 이미 서버인데 왜 앞에 하나를 더 두나. 이유가 여섯 개쯤 된다.

### 3.1 느린 클라이언트 흡수 (가장 중요)

이게 nginx 의 존재 이유 1번이다.

지하철에서 3G 로 접속한 사용자가 10MB 를 업로드한다고 하자. 30초가 걸린다.
nginx 가 없으면 그 30초 내내 **uWSGI 워커 하나가 통째로 묶인다.** 워커 4개짜리
서버는 느린 사용자 4명이면 마비된다.

nginx 는 이벤트 기반이라 커넥션 수천 개를 프로세스 하나로 든다. 그래서:

```
① nginx 가 30초 동안 요청을 다 받아 버퍼에 모은다   (워커는 논다)
② 다 모이면 uWSGI 에 한 번에 넘긴다              (워커 0.05초 사용)
③ 응답도 nginx 가 받아서 천천히 클라이언트에 보낸다 (워커는 이미 자유)
```

**비싼 자원(파이썬 워커)을 싼 자원(nginx 커넥션) 뒤에 숨기는 것**이 이 구조의 핵심이다.

### 3.2 TLS 종료

HTTPS 암복호화를 nginx 가 처리하고 뒤로는 평문으로 넘긴다. 파이썬이 암호화를
하지 않아도 되고 인증서 갱신을 한 곳에서만 하면 된다.

### 3.3 정적 파일

CSS/JS/이미지는 nginx 가 디스크에서 바로 읽어 보낸다. 파이썬을 깨우지 않는다.

### 3.4 라우팅 / 리버스 프록시

경로별로 다른 백엔드에 보낼 수 있다.

```nginx
location /static/          { root /srv/www; }               # 디스크에서 직접
location /api/model_upload/{ proxy_pass http://127.0.0.1:5000; }
location /                 { uwsgi_pass unix:/run/uwsgi/app.sock; }
```

> `proxy_pass` 는 **HTTP** 로 넘기고, `uwsgi_pass` 는 **uwsgi 프로토콜**로 넘긴다.
> 후자가 조금 더 빠르다 (HTTP 헤더를 다시 파싱하지 않는다). 이 저장소의
> `deploy_vlms/nginx/model_upload.conf` 는 `proxy_pass` 를 쓰는데, Flask 가
> 별도 포트(:5000)로 HTTP 를 듣는 배치를 가정하기 때문이다.

### 3.5 방어선

`client_max_body_size`, `proxy_read_timeout`, rate limit, IP 제한 — 악의적이거나
그냥 큰 요청을 **파이썬에 닿기 전에** 자른다. `model_upload.conf` 주석이 이걸
실전 사례로 설명한다: 기본값 `client_max_body_size 1m` 때문에 청크가 Flask 에
도착하기도 전에 413 이 나던 문제.

### 3.6 로드밸런싱

`upstream` 블록으로 백엔드를 여러 개 두고 나눠 보낸다. → `02-load-balancing.md`

---

## 4. 인바운드 / 아웃바운드

용어 자체는 단순하다. **"내 프로세스 기준으로 요청이 들어오는가, 나가는가."**

```
        ┌──────── 내 Flask 앱 ────────┐
        │                              │
 요청 ──▶  인바운드 (inbound)          │
        │   · 남이 나를 부른다          │
        │   · @app.route 가 받는다      │
        │   · nginx / uWSGI 가 관리     │
        │                              │
        │          아웃바운드 (outbound) ──▶ 요청
        │           · 내가 남을 부른다   │
        │           · requests.post()   │
        │           · 관리 주체가 없다 ← │
        └──────────────────────────────┘
```

여기서 **가장 자주 틀리는 지점**이 이것이다.

> **nginx 와 uWSGI 는 인바운드만 관리한다. 아웃바운드는 아무도 안 봐준다.**

nginx 는 요청을 어느 워커에 줄지 정하고 uWSGI 는 워커를 몇 개 띄울지 정한다.
하지만 그 워커가 코드 안에서 `requests.post("http://127.0.0.1:8006")` 을 부르면,
그건 그냥 **평범한 소켓 통신**이다. 밸런싱도, 재시도도, 헬스체크도, 타임아웃 기본값도
없다. 전부 코드가 직접 하거나 **또 하나의 nginx 를 아웃바운드 쪽에 세워야** 한다.

이 저장소가 순수한 예시다. `flask_api/vlm_serve/service_template.py` 에서:

```python
@property
def upstream_base_url(self) -> str:
    service_url = os.environ.get(f"VLM_SERVE_{self.env_prefix}_BASE_URL", ...)
    if service_url:
        return service_url
    return f"http://{upstream_host}:{self.upstream_port}"   # ← 문자열 하나
```

목적지가 문자열 **하나**다. uWSGI 워커가 4개든 40개든 전부 같은 `:8006` 으로
나간다. 워커를 늘려도 vLLM 인스턴스가 늘어나지 않는다 — 당연한 얘기 같지만
"프로세스를 늘리면 처리량이 는다" 는 인바운드 쪽 직관을 아웃바운드에 잘못
옮겨오면 정확히 여기서 헷갈린다.

### 타임아웃은 사슬이다

인바운드/아웃바운드가 **각 레이어마다 따로** 있고 안쪽이 바깥쪽보다 짧아야 한다.

```
nginx   proxy_read_timeout 900s   ← 가장 김
  └ uWSGI  harakiri 600s
      └ Flask  requests(timeout=(5, 300))   ← 가장 짧아야 한다
          └ vLLM
```

거꾸로 되어 있으면 — 예컨대 harakiri 30초, requests timeout 300초 — 워커가
매번 하라키리로 죽고, 클라이언트는 502 를 보며, 로그에는 원인이 안 남는다.
**"이상하게 30초쯤에서 끊긴다" 는 증상을 만나면 이 사슬을 위에서부터 훑어라.**

---

## 5. 이 저장소에 대입해 보기

```
[팀원의 OpenAI 파이썬 클라이언트]
   │  Authorization: Bearer <VLM_SERVE_TOKEN>
   ▼
nginx  ──────────────────────────────── 인바운드 시작
   │  TLS 종료, /api/ 를 uWSGI 로
   ▼
uWSGI worker N
   ▼
Flask: register_flask_api(app)
   │  /api → /api/vlm_serve → /api/vlm_serve/qwen3.8-27b
   │  before_request: 토큰 검사 (compare_digest)
   │  _build_upstream_headers(): 클라이언트 Authorization 을 **삭제**하고
   │                             upstream 용 API_KEY 를 새로 주입
   ▼  ──────────────────────────────── 아웃바운드 시작
requests.post("http://127.0.0.1:8006/v1/chat/completions", timeout=(5, 300))
   ▼
vLLM (별개 프로세스. Flask 와 코드를 한 줄도 공유하지 않는다)
```

Authorization 헤더를 지웠다 다시 넣는 부분이 인바운드/아웃바운드 구분의
좋은 예다. 클라이언트가 보낸 `Authorization` 은 **이 프록시에게 온 인바운드
자격증명**이지, vLLM 에게 갈 것이 아니다. 그대로 흘려보내면 vLLM 이
`API_KEY` 를 켰을 때 엉뚱한 키가 도착해 401 이 난다. 그래서 지우고
아웃바운드용 키를 따로 채운다.

---

## 6. 자주 만나는 증상 → 어느 레이어인가

| 증상 | 대개 여기 | 볼 것 |
|---|---|---|
| 413 Request Entity Too Large | nginx | `client_max_body_size` |
| 504 Gateway Timeout | nginx | `proxy_read_timeout` |
| 502 Bad Gateway | uWSGI 가 죽었거나 소켓 경로 불일치 | uWSGI 로그, `socket=` 경로/권한 |
| 큰 요청에서만 502 | uWSGI | `harakiri` 가 먼저 끊는다 |
| 부하 시 응답이 줄줄이 밀림 | uWSGI | `processes`/`threads` 부족, 또는 워커가 아웃바운드 대기로 묶임 |
| 새로고침마다 결과가 다름 | Flask | 워커별 전역 상태. 프로세스가 다르다 |
| 정적 파일 404 | nginx | `location /static` 의 `root`/`alias` |
| 코드를 고쳤는데 반영이 안 됨 | uWSGI | 리로드 안 함. `touch-reload` 또는 재기동 |

**진단 순서는 항상 바깥에서 안쪽으로.** nginx access/error 로그 → uWSGI 로그 →
앱 로그. 요청이 어디까지 도달했는지를 먼저 확정하면 후보가 절반씩 줄어든다.

---

## 7. 한 줄 요약

- **Flask** = WSGI 규격을 만족하는 파이썬 객체 하나. 서버가 아니다.
- **uWSGI** = 그 객체를 프로세스 N벌로 띄우고 감시하는 **애플리케이션 서버**. 인바운드 담당.
- **nginx** = 인터넷을 마주보는 **웹 서버 / 리버스 프록시**. 느린 클라이언트를 흡수해
  비싼 파이썬 워커를 지킨다.
- **아웃바운드는 아무도 안 봐준다.** 내가 남을 부르는 통신은 코드가 직접 책임진다.

---

## 더 볼 것

- `02-load-balancing.md` — 로드밸런싱 원리, 알고리즘, LLM 서빙에서 달라지는 점
- `deploy_vlms/nginx/model_upload.conf` — nginx 기본값이 어떻게 앱을 깨뜨리는지의 실전 주석
- `flask_api/vlm_serve/service_template.py` — 아웃바운드 프록시 구현
