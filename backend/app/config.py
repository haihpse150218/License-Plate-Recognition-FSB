#COnfig root path
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parents[2]

def setup_config():
    BASE_DIR = Path(__file__).resolve().parents[2]
    print(BASE_DIR)

REGEX_OTO = r'^\d{2}-?[A-Z]-?\d{3}\.?\d{2}$'
REGEX_XEMAY = r'^(?:\d{2}-?[A-Z]\d-?\d{2,3}\.?\d{2}|\d{2}-?[A-Z]{2}-?\d{2,3}\.?\d{2})$'
MODEL_PATH = (
    Path(BASE_DIR)
    / "runs/detect/latest_best.pt"
)
MODEL_PATH = MODEL_PATH.resolve()