"""
Trains the tokenizers for translation. Each language has its own tokenizer.
"""


from src.tokenizers.BPE import BytePairEncoding
from datasets import load_dataset
from pathlib import Path 
CONFIG = {
    "tokenizer": {
        "vocab_size":500,
        "min_frequency":100
    },
    "save_path": "./experiments/multi30k/"
}


def main():

    ds = load_dataset("bentrevett/multi30k")#, data_dir="data/multi30k_de_en")
    train_ds = ds["train"]

    tokenizer_en = BytePairEncoding(**CONFIG["tokenizer"])
    text_en = " ".join([example["en"] for example in train_ds])
    print("Number of EN chars:", len(text_en))
    tokenizer_en.fit(text_en)
    tokenizer_en.save(Path(CONFIG["save_path"], "tokenizer_en.json"))
    print("\nEnglish tokenizer vocab (sorted by token id):")
    sorted_vocab_en = sorted(tokenizer_en.vocab.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab_en:
        print(f"{idx}: {repr(token)}")

    tokenizer_fr = BytePairEncoding(**CONFIG["tokenizer"])
    text_de = " ".join([example["de"] for example in train_ds])
    print("Number of DE chars:", len(text_de))
    tokenizer_fr.fit(text_de)
    tokenizer_fr.save(Path(CONFIG["save_path"], "tokenizer_de.json"))
    print("\nGerman tokenizer vocab (sorted by token id):")
    sorted_vocab_de = sorted(tokenizer_fr.vocab.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab_de:
        print(f"{idx}: {repr(token)}")

if __name__ == "__main__":
    main()