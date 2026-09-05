"""load_env_file 의 ${VAR} 확장과 config/site.env 로딩 테스트.

    pytest deploy_vlms/scripts/test_serve_vlm_env.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "serve_vlm", Path(__file__).with_name("serve_vlm.py")
)
assert _SPEC and _SPEC.loader  # spec_from_file_location 은 Optional 을 낸다
serve_vlm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(serve_vlm)


def _write(tmp_path: Path, body: str) -> str:
    path = tmp_path / "sample.env"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_expands_var_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", "/srv/models")
    serve_vlm.load_env_file(_write(tmp_path, "MODEL_ID=${MODEL_ROOT}/Qwen3.8-27B\n"))
    assert os.environ["MODEL_ID"] == "/srv/models/Qwen3.8-27B"


def test_unset_var_stays_literal_so_isabs_catches_it(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    serve_vlm.load_env_file(_write(tmp_path, "MODEL_ID=${MODEL_ROOT}/Qwen3.8-27B\n"))
    # 조용히 상대경로가 되면 안 된다. 리터럴로 남아야 isabs 검증이 실패시킨다.
    assert os.environ["MODEL_ID"] == "${MODEL_ROOT}/Qwen3.8-27B"
    assert not os.path.isabs(os.environ["MODEL_ID"])


def test_expansion_runs_after_quote_stripping(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", "/srv/models")
    serve_vlm.load_env_file(_write(tmp_path, 'MODEL_ID="${MODEL_ROOT}/MAI-UI-8B"\n'))
    assert os.environ["MODEL_ID"] == "/srv/models/MAI-UI-8B"


def test_values_without_dollar_are_untouched(tmp_path, monkeypatch):
    # EXTRA_VLLM_ARGS / LIMIT_MM_PER_PROMPT 같은 값이 망가지지 않는지.
    serve_vlm.load_env_file(
        _write(tmp_path, 'LIMIT_MM_PER_PROMPT={"image": 2}\nMAX_NUM_SEQS=8\n')
    )
    assert os.environ["LIMIT_MM_PER_PROMPT"] == '{"image": 2}'
    assert os.environ["MAX_NUM_SEQS"] == "8"


def test_main_reads_site_env_before_config(tmp_path, monkeypatch):
    """start_all / start_model 은 serve_vlm 을 그대로 부르므로, 여기서 site.env 를 읽어야
    셸 export 없이 ${MODEL_ROOT} 가 펼쳐진다."""
    deploy = tmp_path / "deploy_vlms"
    (deploy / "config" / "models").mkdir(parents=True)
    (deploy / "config" / "site.env").write_text("MODEL_ROOT=/srv/models\n")
    (deploy / "config" / "common.env").write_text("ALLOWED_MODEL_ROOT=${MODEL_ROOT}\n")
    (deploy / "config" / "models" / "x.env").write_text(
        "MODEL_ID=${MODEL_ROOT}/x\nSERVED_MODEL_NAME=x\nPORT=1\nGPU_ID=0\n"
    )
    # main() 은 os.environ 에 직접 쓴다 - 통째로 갈아끼워 다른 테스트로 새지 않게 한다.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop("MODEL_ROOT", None)
    os.environ.pop("SITE_ENV", None)  # conftest 가 끈 로딩을 이 테스트에서만 되살린다
    os.environ["DEPLOY_VLMS_ROOT"] = str(deploy)
    monkeypatch.setattr(sys, "argv", ["serve_vlm.py", "x"])

    # /srv/models/x 가 없어 검증에서 멈춘다. .env 는 그 전에 읽혔어야 한다.
    with pytest.raises(SystemExit):
        serve_vlm.main()

    assert os.environ["MODEL_ID"] == "/srv/models/x"
    assert os.environ["ALLOWED_MODEL_ROOT"] == "/srv/models"


def test_override_false_keeps_existing_keys(tmp_path, monkeypatch):
    """config/site.env 는 이 모드로 읽는다 - 셸 export 가 site.env 보다 우선."""
    monkeypatch.setenv("MODEL_ROOT", "/shell")
    serve_vlm.load_env_file(_write(tmp_path, "MODEL_ROOT=/file\n"), override=False)
    assert os.environ["MODEL_ROOT"] == "/shell"
