"""VLM 서비스 중앙 설정 - 이 패키지의 유일한 레지스트리.

한 모델 = VLMServiceConfig 한 줄. __init__ 이 이 목록을 돌며 proxy blueprint 를
만들고 등록한다. 포트는 deploy_vlms/config/models/<slug>.env 의 PORT 와 같아야 하고,
route_slug 은 그 파일의 이름(stem)과 같아야 health 가 둘을 한 항목으로 합친다.

현재 포트: 8002 (mai-ui), 8004 (paddleocr-vl-1.5), 8006 (qwen3.8-27b)

2026-08-11 부터 grounding 은 mai-ui 단일 모델, OCR 보조는 paddleocr 만 사용한다
(호스트 RAM 16GB 제약 - 프로세스 수가 GPU 메모리보다 먼저 한계에 닿는다).

ui-venus / ui-tars / got-ocr 은 2026-09-03 에 **가중치를 서버에서 삭제**해서
목록에서 뺐다. 되살리려면 체크포인트를 다시 반입한 뒤 여기 한 줄을 더한다.
"""

from .service_template import VLMServiceConfig

VLM_SERVICES: list[VLMServiceConfig] = [
    VLMServiceConfig(route_slug="mai-ui", display_name="MAI-UI-8B", upstream_port=8002),
    VLMServiceConfig(route_slug="paddleocr-vl-1.5", display_name="PaddleOCR-VL-1.5", upstream_port=8004),
    # route_slug 과 served-model-name 을 일부러 같게 둔다 - 호출부가 slug 대신 모델명을
    # 넘기는 실수가 조용히 통과하지 않도록 (paddleocr 와 같은 규약).
    VLMServiceConfig(route_slug="qwen3.8-27b", display_name="Qwen3.8-27B", upstream_port=8006),
]

__all__ = ["VLM_SERVICES", "VLMServiceConfig"]
