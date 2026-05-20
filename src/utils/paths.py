import os
from pathlib import Path

from omegaconf import OmegaConf

# Project root: go up from utils/ -> src/ -> ROOT
ROOT = Path(__file__).resolve().parents[2]

# env-var overrides so the same code works on a cluster
# where data / experiments live on a scratch filesystem.

# e.g.
#   export TRANSFORMER_DATA_DIR=/scratch/data
#   export TRANSFORMER_EXPERIMENTS_DIR=/scratch/experiments

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
EXPERIMENTS_DIR = Path(os.environ.get("EXPERIMENTS_DIR", ROOT / "experiments"))

def _resolve_path(name: str) -> str:
    try:
        value = globals()[name]
    except KeyError as e:
        raise KeyError(
            f"paths resolver: unknown path '{name}'. "
            f"Known: {sorted(k for k, v in globals().items() if isinstance(v, Path))}"
        ) from e
    return str(value)


OmegaConf.register_new_resolver("paths", _resolve_path, replace=True)