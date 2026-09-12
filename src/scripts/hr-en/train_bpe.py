"""
Trains a english-croatian joint tokenizer.
"""
from itertools import chain

from datasets import load_dataset
from src.datasets.hrenwac_v2 import HrenWac
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths

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
    corpus = ""
    translations = opus["train"]["translation"]
    corpus = " ".join(chain.from_iterable((t["hr"], t["en"]) for t in translations))
    corpus = " ".join(chain.from_iterable((t["hr"], t["en"]) for t in jwl["train"]))
    corpus = " ".join(chain.from_iterable((t["hr"], t["en"]) for t in hrenwac))
    tokenizer = BytePairEncoding(**CONFIG["tokenizer"])

    tokenizer.fit(corpus)
    tokenizer.save(CONFIG["save_path"] / "tokenizer_en.json")
    print("\nEnglish tokenizer vocab (sorted by token id):")

if __name__ == "__main__":
    main()