"""
Intended for tokenizing large datasets stored either as json files 
or hf datasets.
Splits the source dataset into shards, and creates one job per shard
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

from datasets import load_from_disk
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class Config:
    max_docs: int | None = None  # debug cap; None = whole corpus

    source: str = "json"  # json or hf
    n_shards: int = 500
    n_workers: int = 35
    encoder: str = "classla" # classla fineweb_en fineweb_hr macocu

    tokenizer_path: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48007_hr_en"
    dataset_path: Path = paths.DATA_DIR / "raw/classla-web-hr-v2/CLASSLA-web.hr.2.0.jsonl"
    save_path: Path = paths.DATA_DIR / "processed/classla_hr_v2_48007_tok"


def load_config(argv: list[str] | None = None) -> Config:
    OmegaConf.register_new_resolver("paths", lambda name: str(getattr(paths, name)))
    cli = OmegaConf.from_cli(argv)  # defaults from dataclass
    cfg_path = cli.pop("config", None)
    cfg = OmegaConf.structured(Config)
    if cfg_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))
    cfg = OmegaConf.merge(cfg, cli)  # external file overrides
    return OmegaConf.to_object(cfg)


def encode_classla(doc, tok):
    sv = tok._special_vocab
    return [sv["<HR>"], *tok.tokenize(doc["text"]), sv["<EOS>"]]


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


def encode_fineweb_en(doc, tok):
    sv = tok._special_vocab
    return [sv["<EN>"], *tok.tokenize(doc["text"]), sv["<EOS>"]]


def encode_fineweb_hr(doc, tok):
    sv = tok._special_vocab
    return [sv["<EN>"], *tok.tokenize(doc["text"]), sv["<EOS>"]]


ENCODERS_REGISTRY = {
    "classla": encode_classla,
    "macocu": encode_macocu,
    "fineweb_en": encode_fineweb_en,
    "fineweb_hr": encode_fineweb_hr,
}


def json_worker(shard, cfg, encode_fn, tokenizer, output_path, dtype=np.uint32):

    start_byte, end_byte = shard

    print(f"PID {os.getpid()}: starting shard {start_byte}", flush=True)
    tokens = []
    with open(cfg.dataset_path, mode="rb") as file:
        curr_byte = file.seek(start_byte)

        while curr_byte < end_byte:
            line = json.loads(file.readline())
            line_tokens = encode_fn(line, tokenizer)
            if line_tokens is not None:
                tokens.extend(line_tokens)
            curr_byte = file.tell()

    np.save(output_path, np.array(tokens, dtype=dtype))
    print(f"Saved {output_path}: {len(tokens)} tokens", flush=True)


def json_shards(cfg):
    n_bytes = os.path.getsize(cfg.dataset_path)
    shard_len = n_bytes // cfg.n_shards
    offsets = [0]
    with open(cfg.dataset_path, mode="rb") as file:
        for i in range(1, cfg.n_shards):
            file.seek(shard_len * i)
            file.readline()
            offsets.append(file.tell())
    offsets.append(n_bytes)
    return list(zip(offsets[:-1], offsets[1:], strict=True))


def hf_worker(shard, cfg, encode_fn, tokenizer, output_path, dtype=np.uint32):
    ds = load_from_disk(cfg.dataset_path)
    ds = ds.shard(num_shards=cfg.n_shards, index=shard, contiguous=True)
    tokens = []
    for line in ds:
        line_tokenized = encode_fn(line, tokenizer)
        if line_tokenized is not None:
            tokens.extend(line_tokenized)
    np.save(output_path, np.array(tokens, dtype=dtype))


def hf_shards(cfg):
    return list(range(cfg.n_shards))


SOURCES_REGISTRY = {"json": (json_worker, json_shards), "hf": (hf_worker, hf_shards)}


def main():

    cfg = load_config()
    print("\n[INFO] Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    tokenizer = BytePairEncoding.from_file(cfg.tokenizer_path)

    if tokenizer.vocab_size <= 65535:
        np_dtype = np.uint16
    else:
        np_dtype = np.uint32

    cfg.save_path.mkdir(parents=True, exist_ok=True)
    worker, make_shards = SOURCES_REGISTRY[cfg.source]
    shards = make_shards(cfg)
    encoder = ENCODERS_REGISTRY[cfg.encoder]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
        futures = []
        for i, shard in enumerate(shards):
            futures.append(
                executor.submit(
                    worker,
                    shard,
                    cfg,
                    encoder,
                    tokenizer,
                    cfg.save_path / f"{i}.npy",
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
                executor.shutdown(cancel_futures=True)
                raise

    lens = []
    for i in range(len(shards)):
        path = cfg.save_path / f"{i}.npy"

        lens.append(np.load(path, mmap_mode="r").shape[0])

    data = np.lib.format.open_memmap(
        cfg.save_path / "merged.npy",
        mode="w+",
        dtype=np_dtype,
        shape=(sum(lens),),
    )
    off = 0
    for i in range(len(shards)):
        shard = np.load(
            file=cfg.save_path / f"{i}.npy",
            mmap_mode="r",
        )
        print(shard.shape)
        data[off : off + shard.shape[0]] = shard
        off += len(shard)


if __name__ == "__main__":
    main()
