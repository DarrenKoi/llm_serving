"""PaddleOCR-VL route template."""

from .service_template import VLMServiceConfig, create_vlm_service_blueprint

SERVICE_CONFIG = VLMServiceConfig(
    blueprint_name="paddleocr_vl",
    route_slug="paddleocr-vl-1.5",
    display_name="PaddleOCR-VL-1.5",
    upstream_port=8004,
)
service_blueprint = create_vlm_service_blueprint(SERVICE_CONFIG)

__all__ = ["SERVICE_CONFIG", "service_blueprint"]
