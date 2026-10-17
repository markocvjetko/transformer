from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Optional

from omegaconf import MISSING, OmegaConf

from src.models.lm import DecoderLM

Config = import_module("src.scripts.gpt2-test-large.train_lm").Config


@dataclass
class GenConfig:
    experiment_dir: str = MISSING
    tokenizer_pth: str = MISSING
    prompt: str = MISSING

    model_pth: Optional[Path] = None

    seq_len: int = 256
    temp: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    seed: int = 42


def main():

    cfg = OmegaConf.structured(GenConfig)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())


if __name__ == "__main__":
    main()
