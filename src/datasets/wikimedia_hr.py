import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import load_dataset
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


class WikimediaHR(Dataset):
    def __init__(
        self,
        dataset_path,
        tokenizer: BytePairEncoding,
        max_len: int = 128,
        pad_token=0,
        bos_token=1,
        eos_token=2,
        unk_token=3,
    ):
        self.hf_dataset = load_dataset("wikimedia/wikipedia", "20231101.hr", cache_dir=paths.DATA_DIR / dataset_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.corpus = []
        for data in tqdm(self.hf_dataset["train"].select(range(20000)), desc="Building WikimediaHR dataset"):
            tokens = tokenizer.tokenize(data["text"]) + [self.eos_token]
            self.corpus.extend(tokens)

        print(len(self.corpus))

    def __len__(self):
        return len(self.corpus) // self.max_len  # number of tokens

    def __getitem__(self, idx):
        """
        Returns a dictionary with keys:
          - 'text': tensor of shape (seq_len, ) containing text token ids
        Where seq_len is tokenized text length before any batching or padding.
        """
        return torch.tensor(self.corpus[idx * self.max_len : (idx + 1) * self.max_len], dtype=torch.long)


# def collate_fn(batch, pad_token=0):
#     tensors = [item["text"] for item in batch]
#     padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_token)
#     return padded


if __name__ == "__main__":
    from datasets import load_dataset
    from src.utils import paths

    # # Load Croatian Wikipedia (replace the date with your target version)
    # hf_dataset = load_dataset(
    #     "wikimedia/wikipedia", "20231101.hr", cache_dir=paths.DATA_DIR / "wikimedia/wikipedia/20231101/hr/"
    # )

    tokenizer = BytePairEncoding.from_file(paths.EXPERIMENTS_DIR / "en-hr-tokenizers/tokenizer_37000.json")

    dataset = WikimediaHR("wikimedia/wikipedia/20231101/hr/", tokenizer)

    print(len(dataset))
    for i in range(len(dataset)):
        print(dataset[i])

        print(dataset.tokenizer.decode(dataset[i], add_special=True))
