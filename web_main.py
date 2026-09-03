"""Flask 앱 진입점.

index.py(WSGI)가 여기서 app 을 가져간다.

이 저장소에서 web_main 이 하는 일은 flask_api 를 /api 아래에 얹는 것 하나뿐이다.
라우트는 전부 그 패키지 안에 있고(vlm_serve / model_upload), 등록 순서나 URL
접두어도 거기서 정한다 - 여기에 라우트를 직접 붙이지 말 것.
"""

from flask import Flask

from flask_api import register_flask_api

app = Flask(__name__)
register_flask_api(app)
