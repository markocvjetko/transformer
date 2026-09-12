from datasets.table import np
from torch.utils.data import Dataset

from src.tokenizers.BPE_v2 import BytePairEncoding
from src.utils import paths


class NextTokenPredictionDataset(Dataset):
    def __init__(self, dataset_path: str, seq_len: int = 512):
        self.dataset_path = dataset_path
        self.data = np.load(self.dataset_path, mmap_mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0] // self.seq_len

    def __getitem__(self, idx):
        tokens = self.data[idx * self.seq_len : (idx + 1) * self.seq_len]
        return tokens


if __name__ == "__main__":
    dataset = NextTokenPredictionDataset(
        dataset_path=paths.DATA_DIR / "gpt2-test-large/fineweb-edu_data.npy", seq_len=2048
    )
    tokenizer = BytePairEncoding.from_file(paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k")
    print(len(dataset))
    print(dataset[0])
    print(tokenizer.decode(dataset[0], add_special=True))
