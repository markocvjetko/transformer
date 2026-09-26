import math

import torch
import torch.nn.functional as F


class Perplexity:

    def compute(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: torch.Tensor (B, seq_len, vocab_size)
            src: torch.Tensor (B, seq_len)
        """
        return torch.exp(F.cross_entropy(logits.swapaxes(1, 2), targets, reduction="mean"))
    

class BitsPerByte:

    def __init__(self, vocab: dict, special_vocab: dict, device=None):
        """
        Vocabulary from which token byte lens are read
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

        self.vocab = vocab
        self.special_vocab = special_vocab
        self.token_lens = {
            idx: len(text.encode('utf-8')) for text, idx in vocab.items()
        }
        for _, idx in special_vocab.items():
            self.token_lens[idx] = 0
   
        max_id = max(self.token_lens.keys()) if self.token_lens else 0
        self.id_to_len = torch.zeros(int(max_id) + 1, dtype=torch.long, device=device)
   
        for k, v in self.token_lens.items():
            self.id_to_len[k] = v

    def compute(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: torch.Tensor (B, seq_len, vocab_size)
            src: torch.Tensor (B, seq_len)
        """
        n_bytes = self.id_to_len[torch.argmax(logits, dim=-1)]
        n_bytes = n_bytes.sum(dim=-1)

        loss = F.cross_entropy(logits.swapaxes(1, 2), targets, reduction="sum")
        bits = loss / math.log(2)
        bpb = bits / n_bytes 
        return bpb

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.compute(logits, targets)

if __name__ == "__main__":

    vocab = {"____":0, ":aa": 1, "b":2, "c":3}
    
    #or try with real tokenizer
    
    #from src.tokenizers.BPE import BytePairEncoding
    #tokenizer_path = "experiments/gpt2-test-large/BPE_vocab_48k"
    #tokenizer = BytePairEncoding.from_file(tokenizer_path)
    #vocab = tokenizer.vocab

    bpb = BitsPerByte(vocab)

    logits = torch.randn(2, 4, len(vocab))
    target = torch.randint(0, len(vocab), (2, 4))
    print(logits.shape, target.shape)
    print("Logits shape:", logits.shape)
    print("Target shape:", target.shape)

    result = bpb.compute(logits, target)
    print("Bits per byte:", result)