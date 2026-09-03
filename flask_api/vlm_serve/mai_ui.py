"""MAI-UI VLM route template."""

from .service_template import VLMServiceConfig, create_vlm_service_blueprint

SERVICE_CONFIG = VLMServiceConfig(
    blueprint_name="mai_ui_vlm",
    route_slug="mai-ui",
    display_name="MAI-UI-8B",
    upstream_port=8002,
)
service_blueprint = create_vlm_service_blueprint(SERVICE_CONFIG)

__all__ = ["SERVICE_CONFIG", "service_blueprint"]
