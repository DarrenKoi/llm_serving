"""pytest 공통 설정.

flask_api 는 import 시점에 deploy_vlms/config/site.env(개발 PC 의 실제 경로·토큰)를
os.environ 에 얹는다. 테스트가 그 값을 보면 PC 마다 결과가 달라지므로, 어떤 테스트
모듈보다 먼저 여기서 SITE_ENV 를 '파일이 아닌 경로' 로 돌려 로딩을 끈다.
site.env 로딩 자체를 검증하는 테스트는 SITE_ENV 를 지우거나 경로를 직접 넘긴다.
"""

import os

os.environ.setdefault("SITE_ENV", os.devnull)
