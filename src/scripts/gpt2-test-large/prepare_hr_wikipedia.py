import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets import load_dataset
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class TokenizerConfig:
    path: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k_hr_en"


@dataclass
class DatasetConfig:
    repo: str = "wikimedia/wikipedia"
    name: str = "20231101.hr"
    split: str = "train"
    cache_dir: Path = paths.DATA_DIR / "wikipedia/wikipedia"
    dict_field: str = "text"


@dataclass
class Config:
    save_dir: Path = paths.DATA_DIR / "gpt2-test-large/wikipedia_hr"
    max_docs: int | None = None  # debug cap; None = whole corpus

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


def load_config(config_path: str | None, cli_overrides: list[str]) -> Config:
    cfg = OmegaConf.structured(Config)  # defaults from dataclass
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))  # external file overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))  # CLI overrides last
    return cfg


def worker(shard, tokenizer, output_path, dtype=np.uint32):
    import os

    print(f"[PID {os.getpid()}] Worker started", flush=True)
    tokens = []

    for row in tqdm(shard, desc="Tokenizing"):
        text = row["text"]
        tokens.extend(
            [tokenizer._special_vocab["<HR>"]] + tokenizer.tokenize(text) + [tokenizer._special_vocab["<EOS>"]]
        )
    np.save(output_path, np.array(tokens, dtype=dtype))
    print(f"Saved {output_path}: {len(tokens)} tokens", flush=True)


if __name__ == "__main__":
    rng = 42
    S = 1280
    W = 20
    # usage: python <script_name>.py experiment.yaml batch_size=128 optimizer.lr=1e-3
    args = sys.argv[1:]
    yaml_path = args[0] if args and "=" not in args[0] else None
    cli_overrides = args[1:] if yaml_path else args
    config = load_config(yaml_path, cli_overrides)
    print("\n[INFO] Loaded configuration:")
    print(OmegaConf.to_yaml(config))  # log umjesto printa Logging python modul
    tokenizer = BytePairEncoding.from_file(config.tokenizer.path)

    dataset = load_dataset(
        path=config.dataset.repo,
        name=config.dataset.name,
        split=config.dataset.split,
        cache_dir=config.dataset.cache_dir,
    ).shuffle(seed=rng)
    dict_field = config.dataset.dict_field

    shard_dir = config.save_dir
    shard_dir.mkdir(parents=True, exist_ok=True)

    if tokenizer.vocab_size < 65535:
        np_dtype = np.uint16
    else:
        np_dtype = np.uint32

    jobs = (
        (
            dataset.shard(num_shards=S, index=i, contiguous=True),
            tokenizer,
            shard_dir / f"shard_{i:04d}_data.npy",
            np_dtype,
        )
        for i in range(S)
    )
    print("Jobs prepared")
    with multiprocessing.Pool(W, maxtasksperchild=1) as p:
        p.starmap(worker, jobs, chunksize=1)

    out_data = config.save_dir / "merged.npy"

    lens = []
    for i in range(S):
        path = shard_dir / f"shard_{i:04d}_data.npy"
        lens.append(np.load(path, mmap_mode="r").shape[0])

    data = np.lib.format.open_memmap(
        out_data,
        mode="w+",
        dtype=np_dtype,
        shape=(sum(lens),),
    )
    off = 0
    for i in range(S):
        shard = np.load(
            shard_dir / f"shard_{i:04d}_data.npy",
            mmap_mode="r",
        )
        print(shard.shape)
        data[off : off + shard.shape[0]] = shard
        off += len(shard)
