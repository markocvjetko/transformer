import json
import os
import random

from src.tokenizers.BPE_v2 import BytePairEncoding
from src.utils import paths

"""
Trains a tokenizer using the

Classla_v2 croatian and FineWeb-Edu datasets
"""


def build_classla_v2_split():

    random.seed(42)

    with open(paths.DATA_DIR / "gpt2-training/tokenizer/classla_train.txt", "w", encoding="utf-8") as out:
        with open(paths.DATA_DIR / "CLASSLA-web.hr.2.0.jsonl") as classla:
            for line in classla:
                sample = json.loads(line)
                if random.random() < 0.05:
                    out.write(sample["text"] + " ")
    # Print the size of the resulting file in gigabytes
    out_path = paths.DATA_DIR / "gpt2-training/tokenizer/classla_train.txt"
    file_size_bytes = os.path.getsize(out_path)
    file_size_gb = file_size_bytes / (1024**3)
    print(f"Size of classla_train.txt: {file_size_gb:.6f} GB")


def build_fineweb_edu_split():

    with open(paths.DATA_DIR / "gpt2-training/tokenizer/fineweb_edu_train.txt", "w", encoding="utf-8") as out:
        from datasets import load_dataset

        fineweb_edu = load_dataset(path=str(paths.DATA_DIR / "fineweb-edu"), split="train")

        print("Loaded FineWeb-Edu dataset!")
        print(f"Original dataset size: {len(fineweb_edu)}")
        
        fineweb_edu.shuffle(seed=42)
        subset_size = int(len(fineweb_edu) * 0.0217)
        fineweb_edu_subset = fineweb_edu.select(range(subset_size))

        for sample in fineweb_edu_subset:
            out.write(sample["text"] + " ")

    # Print the size of the resulting file in gigabytes
    out_path = paths.DATA_DIR / "gpt2-training/tokenizer/fineweb_edu_train.txt"
    file_size_bytes = os.path.getsize(out_path)
    file_size_gb = file_size_bytes / (1024**3)
    print(f"Size of fineweb_edu_train.txt: {file_size_gb:.6f} GB")


def train_tokenizer():

    corpus_path = paths.DATA_DIR / "gpt2-training/tokenizer/merged.txt"    
    tokenizer = BytePairEncoding(vocab_size=100000)
    tokenizer.fit(corpus_path)
    tokenizer.save(paths.EXPERIMENTS_DIR / "classla_finewebedu_2gb_tokenizer_Bytes")

if __name__ == "__main__":
    # build_classla_v2_split()
    # build_fineweb_edu_split()
    train_tokenizer()
