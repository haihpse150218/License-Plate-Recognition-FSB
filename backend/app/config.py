#COnfig root path
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parents[2]

def setup_config():
    BASE_DIR = Path(__file__).resolve().parents[2]
    print(BASE_DIR)