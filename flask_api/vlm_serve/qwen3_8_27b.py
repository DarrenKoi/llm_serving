"""Qwen3.8-27B VLM route template."""

from .service_template import VLMServiceConfig, create_vlm_service_blueprint

SERVICE_CONFIG = VLMServiceConfig(
    blueprint_name="qwen3_8_27b_vlm",
    route_slug="qwen3.8-27b",
    display_name="Qwen3.8-27B",
    upstream_port=8006,
)
service_blueprint = create_vlm_service_blueprint(SERVICE_CONFIG)

__all__ = ["SERVICE_CONFIG", "service_blueprint"]
