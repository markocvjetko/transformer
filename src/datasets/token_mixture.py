import numpy as np
from torch.utils.data import IterableDataset


class TokenMixtureDataset(IterableDataset):
    """
    Takes preprocessed datasets (tokenized npy arrays), and samples them
    """

    def __init__(self, datasets, weights, seq_len):

        self.ds = []
        for dataset in datasets:
            data = np.load(dataset, mmap_mode="r")
            self.ds.append(data)

        self.weights = weights
        self.seq_len = seq_len

    def __iter__(self):

        rng = np.random.default_rng()

        while True:
            idx = rng.choice(len(self.ds), p=self.weights)
            start = rng.integers(0, len(self.ds[idx]) - self.seq_len)
            sample = self.ds[idx][start : start + self.seq_len + 1]
            yield sample


if __name__ == "__main__":
    from tqdm import tqdm

    from src.tokenizers import BPE
    from src.utils import paths

    ds = [
        paths.DATA_DIR / "gpt2-test-large/classla_hr_v2/merged.npy",
        paths.DATA_DIR / "gpt2-test-large/macocu_hr_en_v2_data.npy",
    ]
    weights = [0.5, 0.5]

    dataset = TokenMixtureDataset(ds, weights, seq_len=2048000)

    for i in tqdm(range(1000000)):
        next(iter(dataset))
    # print(next(iter(dataset)))
    # tokenizer = BPE.BytePairEncoding.from_file(paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k_hr_en")
    # print(tokenizer.decode(next(iter(dataset)), add_special=True))
