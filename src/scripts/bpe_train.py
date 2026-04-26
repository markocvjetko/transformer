"""
Trains the tokenizers for translation. Each language has its own tokenizer.
"""


from src.tokenizers.BPE import BytePairEncoding
from datasets import load_dataset
from pathlib import Path 
from src.utils import paths


CONFIG = {
    "tokenizer": {
        "vocab_size":37000,
    },
    "tokenizer_path": paths.EXPERIMENTS_DIR / "en-hr-tokenizers",
    "opus": paths.DATA_DIR / "Helsinki-NLP__opus-100",
    "hrenwac": paths.DATA_DIR / "hrenwac",
    "jwl_300": paths.DATA_DIR / "sentence-transformers/parallel-sentences-jw300"
}


def main():

    dataset = load_dataset("bentrevett/multi30k", split="train", cache_dir=CONFIG["dataset_dir"])#, data_dir="data/multi30k_de_en")
    

    tokenizer_en = BytePairEncoding.from_file(CONFIG["tokenizer_path"])
    text_en = " ".join([example["en"] for example in dataset])
    print("Number of EN chars:", len(text_en))
    tokenizer_en.fit(text_en)
    tokenizer_en.save(Path(CONFIG["save_path"], "tokenizer_en.json"))
    print("\nEnglish tokenizer vocab (sorted by token id):")
    sorted_vocab_en = sorted(tokenizer_en.vocab.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab_en:
        print(f"{idx}: {repr(token)}")

    tokenizer_fr = BytePairEncoding(**CONFIG["tokenizer"])
    text_de = " ".join([example["de"] for example in dataset])
    print("Number of DE chars:", len(text_de))
    tokenizer_fr.fit(text_de)
    tokenizer_fr.save(Path(CONFIG["save_path"], "tokenizer_de.json"))
    print("\nGerman tokenizer vocab (sorted by token id):")
    sorted_vocab_de = sorted(tokenizer_fr.vocab.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab_de:
        print(f"{idx}: {repr(token)}")

if __name__ == "__main__":
    main()