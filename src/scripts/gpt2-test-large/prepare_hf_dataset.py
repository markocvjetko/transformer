"""
Intended for tokenizing large datasets stored as json files.
Splits a json file into shards, and creates one job per shard
that a pool of workers processes.

This only makes sense for large datasets and it is expected that
one shard is large enough to contain many dataset samples.

The script generates .npy files per shard and then merges them.
Large read and write operations in numpy are done with memmap to
avoid memory issues.
"""

import concurrent
import concurrent.futures
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


def encode_classla(doc, tok):
    sv = tok._special_vocab
    return [
        sv["<HR>"],
        *tok.tokenize(doc["text"]),
        sv["<EOS>"],
    ]


def encode_macocu(doc, tok):
    sv = tok._special_vocab
    hr = tok.tokenize(doc["hr"])
    en = tok.tokenize(doc["en"])
    return (
        [sv["<TRANSLATE>"], sv["<HR>"]]
        + hr
        + [sv["<EN>"]]
        + en
        + [sv["<EOS>"]]
        + [sv["<TRANSLATE>"], sv["<EN>"]]
        + en
        + [sv["<HR>"]]
        + hr
        + [sv["<EOS>"]]
    )


ENCODERS_REGISTRY = {"classla": encode_classla, "macocu": encode_macocu}


@dataclass
class Config:
    save_path: Path = paths.DATA_DIR / "processed/classla_hr_v2_48007_tok"
    max_docs: int | None = None  # debug cap; None = whole corpus

    n_shards: int = 500
    n_workers: int = 35
    encoder: str = "classla"

    tokenizer_path: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48007_hr_en"
    dataset_path: Path = paths.DATA_DIR / "raw/classla-web-hr-v2/CLASSLA-web.hr.2.0.jsonl"


def load_config(argv: list[str] | None = None) -> Config:
    cli = OmegaConf.from_cli(argv)  # defaults from dataclass
    cfg_path = cli.pop("config", None)
    cfg = OmegaConf.structured(Config)
    if cfg_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))
    cfg = OmegaConf.merge(cfg, cli)  # external file overrides
    return OmegaConf.to_object(cfg)


def worker(start_byte, end_byte, encode_fn, tokenizer, input_path, output_path, dtype=np.uint32):

    print(f"PID {os.getpid()}: starting shard {start_byte}", flush=True)
    tokens = []
    with open(input_path, mode="rb") as file:
        curr_byte = file.seek(start_byte)

        while curr_byte < end_byte:
            line = json.loads(file.readline())
            line_tokens = encode_fn(line, tokenizer)
            if line_tokens is not None:
                tokens.extend(line_tokens)
            curr_byte = file.tell()

    np.save(output_path, np.array(tokens, dtype=dtype))
    print(f"Saved {output_path}: {len(tokens)} tokens", flush=True)


def main():

    cfg = load_config()
    print("\n[INFO] Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    tokenizer = BytePairEncoding.from_file(cfg.tokenizer_path)

    if tokenizer.vocab_size <= 65535:
        np_dtype = np.uint16
    else:
        np_dtype = np.uint32


    n_bytes = os.path.getsize(cfg.dataset_path)
    shard_len = n_bytes // cfg.n_shards
    print(n_bytes)

    cfg.save_path.mkdir(parents=True, exist_ok=True)
    with open(cfg.dataset_path, mode="rb") as file:
        shard_offsets = [0]
        for i in range(1, cfg.n_shards):
            file.seek(shard_len * i)
            file.readline()
            shard_offsets.append(file.tell())
        shard_offsets.append(n_bytes)

    encoder = ENCODERS_REGISTRY[cfg.encoder]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
        futures = []
        # for i in range(len(shard_offsets) - 1):
        for i in range(cfg.n_shards):
            futures.append(
                executor.submit(
                    worker,
                    shard_offsets[i],
                    shard_offsets[i + 1],
                    encoder,
                    tokenizer,
                    cfg.dataset_path,
                    cfg.save_path
                    / str(
                        shard_offsets[i + 1],
                    ),
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
    # for i in range(len(shard_offsets) - 1):
    for i in range(cfg.n_shards):
        path = cfg.save_path / f"{shard_offsets[i + 1]}.npy"

        lens.append(np.load(path, mmap_mode="r").shape[0])

    data = np.lib.format.open_memmap(
        cfg.save_path / "merged.npy",
        mode="w+",
        dtype=np_dtype,
        shape=(sum(lens),),
    )
    off = 0
    for i in range(cfg.n_shards):
        shard = np.load(
            file=cfg.save_path / f"{shard_offsets[i + 1]}.npy",
            mmap_mode="r",
        )
        print(shard.shape)
        data[off : off + shard.shape[0]] = shard
        off += len(shard)


if __name__ == "__main__":
    main()
