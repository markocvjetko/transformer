import os
from pathlib import Path

# Project root: go up from utils/ -> src/ -> ROOT
ROOT = Path(__file__).resolve().parents[2]

# env-var overrides so the same code works on a cluster
# where data / experiments live on a scratch filesystem.

# e.g.
#   export TRANSFORMER_DATA_DIR=/scratch/data
#   export TRANSFORMER_EXPERIMENTS_DIR=/scratch/experiments


DATA_DIR = Path(os.environ.get("TRANSFORMER_DATA_DIR", ROOT / "data"))
EXPERIMENTS_DIR = Path(os.environ.get("TRANSFORMER_EXPERIMENTS_DIR", ROOT / "experiments"))