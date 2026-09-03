"""PaddleOCR-VL 인스턴스 시작 (start_model.py 래퍼)."""
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.exit(subprocess.call([sys.executable, str(script_dir / "start_model.py"), "paddleocr-vl-1.5"]))
