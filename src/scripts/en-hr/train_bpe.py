"""
Trains a english-croatian joint tokenizer.
"""

import sys
import time
from src.datasets.hrenwac_v2 import HrenWac
from src.tokenizers.BPE import BytePairEncoding
from datasets import load_dataset, tqdm
from pathlib import Path 
from src.utils import paths
from itertools import chain

CONFIG = {
    "tokenizer": {
        "vocab_size":37000,
        "min_frequency":100
    },
    "save_path": paths.EXPERIMENTS_DIR / "en-hr-tokenizers",
    "opus": paths.DATA_DIR / "Helsinki-NLP__opus-100",
    "hrenwac": paths.DATA_DIR / "hrenwac",
    "jwl_300": paths.DATA_DIR / "sentence-transformers/parallel-sentences-jw300"
}


def main():

    opus = load_dataset("Helsinki-NLP/opus-100", "en-hr", cache_dir=CONFIG["opus"])
    jwl = load_dataset("sentence-transformers/parallel-sentences-jw300", "en-hr", cache_dir=CONFIG["jwl_300"])
    jwl = jwl.rename_columns({"english":"en", "non_english":"hr"})
    hrenwac = HrenWac(path=CONFIG["hrenwac"])
    
    print(len(opus["train"]))
    print(len(jwl["train"]))
    print(len(hrenwac))

    print(opus["train"][0]["translation"]["hr"] )
    print(jwl["train"][0]["hr"])
    print(jwl["train"][0])
    print(hrenwac[0]["hr"])
    print(hrenwac[0])
    
    omega_string = str()
    translations = opus["train"]["translation"]
    omega_string = " ".join(chain.from_iterable((t["hr"], t["en"]) for t in translations))
    print("loaded")
    omega_string = " ".join(chain.from_iterable((t["hr"], t["en"]) for t in jwl["train"]))
    print("loaded")
    omega_string = " ".join(chain.from_iterable((t["hr"], t["en"]) for t in hrenwac))
    print("loaded")
    tokenizer = BytePairEncoding(**CONFIG["tokenizer"])

    tokenizer.fit(omega_string)
    tokenizer.save(CONFIG["save_path"] / "tokenizer_en.json")
    print("\nEnglish tokenizer vocab (sorted by token id):")
#     sorted_vocab_en = sorted(tokenizer_en.vocab.items(), key=lambda x: x[1])
#     for token, idx in sorted_vocab_en:
#         print(f"{idx}: {repr(token)}")

#     tokenizer_fr = BytePairEncoding(**CONFIG["tokenizer"])
#     text_de = " ".join([example["de"] for example in dataset])
#     print("Number of DE chars:", len(text_de))
#     tokenizer_fr.fit(text_de)
#     tokenizer_fr.save(Path(CONFIG["save_path"], "tokenizer_de.json"))
#     print("\nGerman tokenizer vocab (sorted by token id):")
#     sorted_vocab_de = sorted(tokenizer_fr.vocab.items(), key=lambda x: x[1])
#     for token, idx in sorted_vocab_de:
#         print(f"{idx}: {repr(token)}")

if __name__ == "__main__":
    main()