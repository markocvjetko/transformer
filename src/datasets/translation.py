from torch.utils.data import Dataset
import torch

from src.tokenizers.BPE import BytePairEncoding

class TranslationDataset(Dataset):

    def __init__(
        self, 
        dataset, 
        src_lang,
        tgt_lang,
        tokenizer_src: BytePairEncoding, 
        tokenizer_tgt: BytePairEncoding, 
        max_len: int = 128,
        pad_token = 0,
        bos_token = 1,
        eos_token = 2,
        unk_token = 3hf_dataset
    ):

        self.dataset = dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len

        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        Returns a dictionary with keys:
          - 'src': tensor of shape (src_seq_len,) containing source language token ids
          - 'tgt': tensor of shape (tgt_seq_len,) containing target language token ids
        Where src_seq_len and tgt_seq_len are the lengths of the tokenized source and target sentences,
        respectively, before any batching or padding.
        """
        tokens_src = self.tokenizer_src.tokenize(self.dataset[idx][self.src_lang], add_special=True)
        tokens_tgt = self.tokenizer_tgt.tokenize(self.dataset[idx][self.tgt_lang], add_special=True)
        
        return {
            "src": torch.tensor(tokens_src, dtype=torch.long),
            "tgt": torch.tensor(tokens_tgt, dtype=torch.long) 
        }


def collate_fn(batch, pad_token=0):
    src_tensors = [item["src"] for item in batch]
    tgt_tensors = [item["tgt"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=pad_token)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_token)

    return src_padded, tgt_padded
