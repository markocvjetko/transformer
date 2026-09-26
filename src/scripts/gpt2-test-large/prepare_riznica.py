"""
Preprocessing for parallel corpus Macocu-hr-en-v2 dataset.
"""

import concurrent
import concurrent.futures
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class TokenizerConfig:
    path: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k_hr_en"


@dataclass
class DatasetConfig:
    path: Path = paths.DATA_DIR / "riznica.jsonl"
    dict_field: str = "text"


@dataclass
class Config:
    save_path: Path = paths.DATA_DIR / "gpt2-test-large/riznica"
    max_docs: int | None = None  # debug cap; None = whole corpus

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


def load_config(config_path: str | None, cli_overrides: list[str]) -> Config:
    cfg = OmegaConf.structured(Config)  # defaults from dataclass
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))  # external file overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))  # CLI overrides last
    return cfg


def worker(start_byte, end_byte, tokenizer, input_path, output_path, dtype=np.uint32):

    print(f"PID {os.getpid()}: starting shard {start_byte}", flush=True)
    tokens = []
    with open(input_path, mode="rb") as file:
        curr_byte = file.seek(start_byte)
        while curr_byte < end_byte:
            line = json.loads(file.readline())["text"]
            tokens.extend(
                [tokenizer._special_vocab["<HR>"]] + tokenizer.tokenize(line) + [tokenizer._special_vocab["<EOS>"]]
            )
            curr_byte = file.tell()

    np.save(output_path, np.array(tokens, dtype=dtype))
    print(f"Saved {output_path}: {len(tokens)} tokens", flush=True)


if __name__ == "__main__":
    rng = 42
    S = 50  # num_shards
    W = 10  # num_workers
    args = sys.argv[1:]
    yaml_path = args[0] if args and "=" not in args[0] else None
    cli_overrides = args[1:] if yaml_path else args
    config = load_config(yaml_path, cli_overrides)
    print("\n[INFO] Loaded configuration:")
    print(OmegaConf.to_yaml(config))  # log umjesto printa Logging python modul
    tokenizer = BytePairEncoding.from_file(config.tokenizer.path)

    if tokenizer.vocab_size <= 65535:
        np_dtype = np.uint16
    else:
        np_dtype = np.uint32

    """
    Takes a stream of text and creates an npy dataset and idx file.
    """
    idx = [0]
    tokens = []

    n_bytes = os.path.getsize(config.dataset.path)
    shard_len = n_bytes // S
    print(n_bytes)
    # shard

    config.save_path.mkdir(parents=True, exist_ok=True)
    with open(config.dataset.path, mode="rb") as file:
        shard_offsets = [0]
        for i in range(1, S):
            file.seek(shard_len * i)
            file.readline()
            shard_offsets.append(file.tell())
        shard_offsets.append(n_bytes)

    with concurrent.futures.ProcessPoolExecutor(max_workers=W) as executor:
        futures = []
        # for i in range(len(shard_offsets) - 1):
        for i in range(S):
            futures.append(
                executor.submit(
                    worker,
                    shard_offsets[i],
                    shard_offsets[i + 1],
                    tokenizer,
                    config.dataset.path,
                    config.save_path / str(shard_offsets[i + 1]),
                    np_dtype,
                )
            )
        print("All tasks submitted", flush=True)
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                print("A shard completed", flush=True)
            except Exception:
                traceback.print_exc()

    lens = []
    for i in range(S):
        path = config.save_path / f"{shard_offsets[i + 1]}.npy"
        lens.append(np.load(path, mmap_mode="r").shape[0])

    data = np.lib.format.open_memmap(
        config.save_path / "merged.npy",
        mode="w+",
        dtype=np_dtype,
        shape=(sum(lens),),
    )

    off = 0
    for i in range(S):
        shard = np.load(
            file=config.save_path / f"{shard_offsets[i + 1]}.npy",
            mmap_mode="r",
        )
        print(shard.shape)
        data[off : off + shard.shape[0]] = shard
        off += len(shard)

