"""MAI-UI-2B VLM route template."""

from .service_template import VLMServiceConfig, create_vlm_service_blueprint

SERVICE_CONFIG = VLMServiceConfig(
    blueprint_name="mai_ui_2b_vlm",
    route_slug="mai-ui-2b",
    display_name="MAI-UI-2B",
    upstream_port=8007,
)
service_blueprint = create_vlm_service_blueprint(SERVICE_CONFIG)

__all__ = ["SERVICE_CONFIG", "service_blueprint"]
