import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class RandomAccessMixDataset(IterableDataset):
    """
    Takes preprocessed datasets (tokenized npy arrays), and samples them.
    Weights are normalized to 1
    """

    def __init__(self, datasets, weights, seq_len):

        self.ds = []
        for dataset in datasets:
            data = np.load(dataset, mmap_mode="r")
            self.ds.append(data)

        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= np.sum(self.weights)
        self.seq_len = seq_len

    def __iter__(self):

        rng = np.random.default_rng()

        while True:
            idx = rng.choice(len(self.ds), p=self.weights)
            start = rng.integers(0, len(self.ds[idx]) - self.seq_len)
            sample = self.ds[idx][start : start + self.seq_len + 1]
            yield sample


class MixDataset(Dataset):
    def __init__(self, ds_paths, seq_len, total_chunks, weights, seed):

        self.seq_len = seq_len
        self.total_chunks = total_chunks
        self.ds = [np.load(p, mmap_mode="r") for p in ds_paths]

        self.ds_idx, self.chunk_idx = self.build_sampling_plan(
            [(len(d) - 1) // seq_len for d in self.ds],
            total_chunks,
            weights,
            seed,
        )

    def __len__(self):
        return len(self.chunk_idx)

    def __getitem__(self, idx):
        start = int(self.chunk_idx[idx]) * self.seq_len
        chunk = self.ds[int(self.ds_idx[idx])][start : start + self.seq_len + 1]
        return chunk  # uint16 memmap slice; no astype

        
    @staticmethod
    def build_sampling_plan(dataset_lens, total_chunks, weights, seed=42):

        weights = np.array(weights)

        rng = np.random.default_rng(seed)

        weights = weights / weights.sum()  # normalize weights
        chunks_per_dataset = np.round(weights * total_chunks).astype(int)

        chunks_per_dataset[np.argmax(dataset_lens)] += total_chunks - chunks_per_dataset.sum()

        dataset_indices = np.array([], dtype=np.int64)
        dataset_chunks = np.array([], dtype=np.int64)
        for i in range(len(dataset_lens)):
            epochs, mod = divmod(chunks_per_dataset[i], dataset_lens[i])
            print(epochs, mod)

            indices = np.array([], dtype=np.int64)
            while indices.size < chunks_per_dataset[i]:
                indices = np.concatenate((indices, rng.permutation(dataset_lens[i])))
            indices = indices[: chunks_per_dataset[i]]

            dataset_indices = np.concatenate((dataset_indices, np.full(indices.size, i)))
            dataset_chunks = np.concatenate((dataset_chunks, indices))

        perm = rng.permutation(len(dataset_indices))
        dataset_indices = dataset_indices.astype(np.uint8)[perm]
        dataset_chunks = dataset_chunks.astype(np.int32)[perm]

        return dataset_indices, dataset_chunks


if __name__ == "__main__":
    from tqdm import tqdm

    from src.utils import paths

    ds = [
        paths.DATA_DIR / "gpt2-test-large/classla_hr_v2/merged.npy",
        paths.DATA_DIR / "gpt2-test-large/macocu_hr_en_v2_data.npy",
    ]
    weights = [0.5, 0.5]

    dataset = RandomAccessMixDataset(ds, weights, seq_len=2048000)

    for _ in tqdm(range(1000000)):
        next(iter(dataset))
    # print(next(iter(dataset)))
    # tokenizer = BPE.BytePairEncoding.from_file(paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k_hr_en")
    # print(tokenizer.decode(next(iter(dataset)), add_special=True))
