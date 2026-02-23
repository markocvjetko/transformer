from pathlib import Path

# Project root: go up from utils/ -> src/ -> ROOT
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"