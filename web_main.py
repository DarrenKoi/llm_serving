"""Flask 앱 팩토리.

index.py(WSGI)가 여기서 app 을 가져간다.

이 저장소에서 web_main 이 하는 일은 flask_api 를 /api 아래에 얹는 것 하나뿐이다.
라우트는 전부 그 패키지 안에 있고(vlm_serve / model_upload), 등록 순서나 URL
접두어도 거기서 정한다 - 여기에 라우트를 직접 붙이지 말 것.
"""

from typing import cast

from flask import Flask
from flask.json.provider import DefaultJSONProvider

from flask_api import register_dashboard, register_flask_api


def create_app() -> Flask:
    """앱을 만들고 설정한 뒤 API 를 붙인다."""
    app = Flask(__name__)

    # 응답 JSON 을 사람이 읽을 수 있게 둔다.
    #   ensure_ascii=False : 이 저장소의 에러 메시지가 한글이다. 기본값(True)이면
    #     \uXXXX 로 이스케이프돼서 curl 로 볼 때 사실상 못 읽는다.
    #   sort_keys=False    : /api/health 페이로드는 사람이 눈으로 읽는 순서대로
    #     조립돼 있다. 알파벳 정렬하면 그 구조가 흩어진다.
    # app.json 은 JSONProvider 로만 선언돼 있어 두 속성이 타입 체커에 안 보인다.
    json_provider = cast(DefaultJSONProvider, app.json)
    json_provider.ensure_ascii = False
    json_provider.sort_keys = False

    # MAX_CONTENT_LENGTH 는 **설정하지 말 것**.
    # model_upload 의 put_chunk 는 request.stream 을 직접 읽고 한도를 스스로
    # 검사해서 ChunkTooLarge(한도를 본문에 담은 JSON)를 돌려준다. 업로드
    # 클라이언트는 그 응답을 보고 청크를 반으로 줄인다. 여기서 한도를 걸면
    # Flask 가 그 전에 밋밋한 413 으로 끊어버려 그 적응 로직이 죽는다.
    # 경계에서의 방어는 nginx 의 client_max_body_size 가 맡는다.

    register_flask_api(app)   # /api/*
    register_dashboard(app)   # / (GPU/모델 대시보드)
    return app


app = create_app()
