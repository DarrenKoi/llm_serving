"""GPU / 모델 상태 대시보드.

루트(/)에 붙는다 - /api 아래가 아니다. 그래서 register_flask_api 가 아니라
register_dashboard 로 따로 등록한다(register_flask_api 의 계약은 '전부 /api 아래').

페이지는 /api/health 하나만 폴링한다. 그 응답에 vlm_serve / gpu_status /
model_upload 가 모두 들어 있어 대시보드용 엔드포인트를 따로 둘 이유가 없다.
"""

from flask import Blueprint, render_template

dashboard_blueprint = Blueprint("dashboard", __name__, template_folder="templates")


@dashboard_blueprint.route("/", methods=["GET"])
def index():
    """대시보드 페이지."""
    return render_template("dashboard.html")


def register_dashboard(app) -> None:
    """앱 루트에 대시보드를 붙인다."""
    app.register_blueprint(dashboard_blueprint)


__all__ = ["dashboard_blueprint", "register_dashboard"]
