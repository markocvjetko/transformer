from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils import paths


def shuffle_documents(input_path, output_path, eos=2, seed=42, block=1 << 26, group=1 << 28):
    """
    Shuffles documents of npy array. Assumes documents are separated
    by the 'eos' token.
    """
    x = np.load(input_path, mmap_mode="r")
    ends = np.concatenate([np.flatnonzero(x[s : s + block] == eos) + s + 1 for s in range(0, len(x), block)])

    # Include final document if it doesn't end with EOS
    ends = np.concatenate(([0], ends, [len(x)]))
    ends = np.unique(ends)

    order = np.random.default_rng(seed).permutation(len(ends) - 1)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=x.dtype, shape=x.shape)

    pos = 0
    buf, buf_len = [], 0
    for i in order:
        doc = x[ends[i] : ends[i + 1]]
        buf.append(doc)
        buf_len += len(doc)
        if buf_len >= group:  # flush ~group tokens at once
            out[pos : pos + buf_len] = np.concatenate(buf)
            pos += buf_len
            buf, buf_len = [], 0
    if buf:
        out[pos : pos + buf_len] = np.concatenate(buf)

    out.flush()


datasets = [
    paths.DATA_DIR / "processed/1B-model-vocab-48000/classla-web-bs-v1/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/classla-web-bs-v2/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/classla-web-hr-v1/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/classla-web-hr-v2/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/dgt-en-hr-v2021/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/fineweb-2-bos/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/fineweb-2-hrv/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/fineweb-edu-10BT/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/fineweb-hq-20BT/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/macocu-bs-en-sent/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/macocu-en-hr-sent/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/riznica/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/settimes-bs-en-v2/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/settimes-en-hr-v2/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/wikipedia-bs-20231101/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/wikipedia-en-20231101/tokenized.npy",
    paths.DATA_DIR / "processed/1B-model-vocab-48000/wikipedia-hr-20231101/tokenized.npy",
]

def shuffle_dataset(dataset):
    dst = dataset.parent / "tokenized_shuffled.npy"
    shuffle_documents(dataset, dst, eos=2, seed=42)
    return dataset


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=len(datasets)) as pool:  # one process per dataset
        futures = [pool.submit(shuffle_dataset, dataset) for dataset in datasets]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Shuffling datasets"):
            print(f"Done: {future.result()}")
