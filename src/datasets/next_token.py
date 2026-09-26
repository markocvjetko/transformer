import numpy as np
from torch.utils.data import Dataset

from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


class NextTokenPredictionDataset(Dataset):
    def __init__(self, dataset_path: str, seq_len: int = 512):
        self.dataset_path = dataset_path
        self.dataset = np.load(self.dataset_path, mmap_mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return self.dataset.shape[0] // self.seq_len

    def __getitem__(self, idx):
        tokens = self.dataset[idx * self.seq_len : (idx + 1) * self.seq_len]
        return tokens


if __name__ == "__main__":
    dataset = NextTokenPredictionDataset(
        dataset_path=paths.DATA_DIR / "gpt2-test-large/wikipedia_hr/merged.npy", seq_len=2048
    )
    tokenizer = BytePairEncoding.from_file(paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k_hr_en")
    print(len(dataset))
    print(dataset[0])
    print(tokenizer.decode(dataset[0], add_special=True))
