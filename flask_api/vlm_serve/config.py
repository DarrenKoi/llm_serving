"""VLM 서비스 중앙 설정.

모든 VLM 서비스의 포트, 모델명, 활성 여부를 한 곳에서 관리한다.
현재 활성 포트: 8002 (mai-ui), 8004 (paddleocr-vl-1.5), 8006 (qwen3.8-27b)

2026-08-11 부터 grounding 은 mai-ui 단일 모델, OCR 보조는 paddleocr 만 사용한다
(호스트 RAM 16GB 제약 - 프로세스 수가 GPU 메모리보다 먼저 한계에 닿는다).

ui-venus / ui-tars / got-ocr 은 2026-09-03 에 **가중치를 서버에서 삭제**해서
목록에서 뺐다. enabled=False 로 남겨 두면 "플래그만 되돌리면 살아난다"는
잘못된 신호를 준다 - 되살리려면 체크포인트를 다시 반입해야 한다.
route 템플릿은 13줄짜리 service_template 사본이라 필요하면 다시 만들면 된다
(mai_ui.py 를 복사해 slug/port 만 바꾸는 수준).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VLMServiceEntry:
    """VLM 서비스 정의."""

    route_slug: str  # Flask proxy route slug
    display_name: str  # 표시용 이름
    model_name: str  # vLLM --served-model-name
    upstream_port: int  # vLLM 서빙 포트
    enabled: bool = True  # 활성 여부


# ── 전체 VLM 서비스 목록 ──────────────────────────────────────────────
# 포트 및 활성 여부를 변경하려면 이 목록만 수정하면 된다.
ALL_VLM_SERVICES: list[VLMServiceEntry] = [
    VLMServiceEntry("mai-ui", "MAI-UI-8B", "mai-ui-8b", 8002, enabled=True),
    # A/B 벤치 전용. 상시 기동이 아니라서 기본 off - 켜려면 여기 enabled=True 로 바꾸고
    # deploy 쪽에서도 mai-ui-2b 를 띄운다 (호스트 RAM 16GB, 4번째 인스턴스 금지).
    VLMServiceEntry("mai-ui-2b", "MAI-UI-2B", "mai-ui-2b", 8007, enabled=False),
    VLMServiceEntry("paddleocr-vl-1.5", "PaddleOCR-VL-1.5", "paddleocr-vl-1.5", 8004, enabled=True),
    # route_slug 과 model_name 을 일부러 같게 둔다 - 호출부가 slug 대신 모델명을
    # 넘기는 실수가 조용히 통과하지 않도록 (paddleocr 와 같은 규약).
    VLMServiceEntry("qwen3.8-27b", "Qwen3.8-27B", "qwen3.8-27b", 8006, enabled=True),
]

# 활성 서비스만 필터링
ENABLED_VLM_SERVICES: list[VLMServiceEntry] = [s for s in ALL_VLM_SERVICES if s.enabled]

# route_slug → VLMServiceEntry 매핑
_SERVICE_MAP: dict[str, VLMServiceEntry] = {s.route_slug: s for s in ALL_VLM_SERVICES}


def get_service_by_slug(slug: str) -> VLMServiceEntry | None:
    """route_slug 으로 서비스를 찾는다."""
    return _SERVICE_MAP.get(slug)


def get_enabled_slugs() -> set[str]:
    """활성 서비스 route_slug 집합을 반환한다."""
    return {s.route_slug for s in ENABLED_VLM_SERVICES}


__all__ = [
    "ALL_VLM_SERVICES",
    "ENABLED_VLM_SERVICES",
    "VLMServiceEntry",
    "get_enabled_slugs",
    "get_service_by_slug",
]
