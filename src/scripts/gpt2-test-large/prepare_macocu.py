"""
Preprocessing for parallel corpus Macocu-hr-en-v2 dataset.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class TokenizerConfig:
    path: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k_hr_en"


@dataclass
class DatasetConfig:
    path: Path = paths.DATA_DIR / "MaCoCu-hr-en-v2/parsed_macocu.jsonl"
    dict_field: str = "text"


@dataclass
class Config:
    save_path: Path = paths.DATA_DIR / "gpt2-test-large/macocu_hr_en_v2"
    max_docs: int | None = None  # debug cap; None = whole corpus

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


def load_config(config_path: str | None, cli_overrides: list[str]) -> Config:
    cfg = OmegaConf.structured(Config)  # defaults from dataclass
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))  # external file overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))  # CLI overrides last
    return cfg


if __name__ == "__main__":
    rng = 42
    S = 640
    W = 16
    # usage: python <script_name>.py experiment.yaml batch_size=128 optimizer.lr=1e-3
    args = sys.argv[1:]
    yaml_path = args[0] if args and "=" not in args[0] else None
    cli_overrides = args[1:] if yaml_path else args
    config = load_config(yaml_path, cli_overrides)
    print("\n[INFO] Loaded configuration:")
    print(OmegaConf.to_yaml(config))  # log umjesto printa Logging python modul
    tokenizer = BytePairEncoding.from_file(config.tokenizer.path)

    """
    Takes a stream of text and creates an npy dataset and idx file.
    """
    idx = [0]
    tokens = []

    with open(config.dataset.path, encoding="utf-8") as file:
        for _i, line in enumerate(tqdm(file, desc="Tokenizing")):
            line = json.loads(line)
            line_hr = tokenizer.tokenize(line["hr"])
            line_en = tokenizer.tokenize(line["en"])

            # dirty way of making the parallel corpus have to translate both hr->en and en->hr.
            # simplifies usage later, at the cost of doubling the dataset size (doesn't matter much since it's tiny anyway)
            tokens.extend(
                [tokenizer._special_vocab["<HR>"]]
                + line_hr
                + [tokenizer._special_vocab["<EN>"]]
                + line_en
                + [tokenizer._special_vocab["<EOS>"]]
            )
            idx.append(len(tokens))
            tokens.extend(
                [tokenizer._special_vocab["<EN>"]]
                + line_en
                + [tokenizer._special_vocab["<HR>"]]
                + line_hr
                + [tokenizer._special_vocab["<EOS>"]]
            )
            idx.append(len(tokens))

    save_path = Path(config.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer.vocab_size <= 65535:
        np_dtype = np.uint16
    else:
        np_dtype = np.uint32

    np.save(save_path.with_name(save_path.name + "_data.npy"), np.array(tokens, dtype=np_dtype))
    np.save(save_path.with_name(save_path.name + "_idx.npy"), np.array(idx, dtype=np.int64))
