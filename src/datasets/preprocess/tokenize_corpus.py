"""
Here I will write the code that prepares a dataset into the npz format.

The primary challenge is to figure out how to adapt this code so that it can be used
for different kinds of datasets (e.g. huggingface datasets, jsonl text files, etc.).
The idea is that each dataset should provide an iterable that returns strings, which
are then processed by the tokenizer and appended to the npy file.
"""

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.datasets.preprocess import sources
from src.tokenizers import BPE_v2
from src.utils import paths


def create_npy_dataset(
    src: Iterable[str], 
    tokenizer, 
    save_path: str | os.PathLike, 
    max_docs: int | None = None):
    """
    Takes a stream of text and creates an npy dataset and idx file.
    """
    idx = [0]
    tokens = []

    for i, text in enumerate(tqdm(src, desc="Tokenizing")):
        if max_docs is not None and i >= max_docs:
            break
        tokens.extend(tokenizer.tokenize(text) + [tokenizer._special_vocab["<EOS>"]])
        idx.append(len(tokens))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer.vocab_size <= 65535:
        np_dtype = np.uint16
    else:
        np_dtype = np.uint32
    np.save(save_path.with_name(save_path.name + "_data.npy"), np.array(tokens, dtype=np_dtype))
    np.save(save_path.with_name(save_path.name + "_idx.npy"), np.array(idx, dtype=np.int64))


if __name__ == "__main__":
    tokenizer_path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k"
    tokenizer = BPE_v2.BytePairEncoding.from_file(tokenizer_path)

    json_source = sources.JsonlSource(path=paths.DATA_DIR / "CLASSLA-web.hr.2.0.jsonl")
    save_path = paths.DATA_DIR / "gpt-2/classla_test"
    create_npy_dataset(json_source, tokenizer, save_path)
