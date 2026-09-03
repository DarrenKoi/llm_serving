"""load_env_file 의 ${VAR} 확장 테스트.

    pytest deploy_vlms/scripts/test_serve_vlm_env.py
"""

import importlib.util
import os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "serve_vlm", Path(__file__).with_name("serve_vlm.py")
)
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
