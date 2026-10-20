import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths

"""
Trains a tokenizer using the

Classla_v2 croatian and FineWeb-Edu datasets
"""


@dataclass
class TokenizerTrainConfig:
    train_config: str = str(paths.EXPERIMENTS_DIR / "gpt2-test-large/tokenizer_config.yaml")

    datasets: list[dict] = field(
        default_factory=lambda: [
            {
                "name": "classla_hr_v1",
                "src_path": str(paths.DATA_DIR / "raw/classla-web-hr-v2/CLASSLA-web.hr.2.0.jsonl"),
                "dst_path": str(paths.DATA_DIR / "processed/tokenizer-split/CLASSLA-web.hr.2.0.txt"),
                "size_bytes": 800_000_000,
                "type": "json",
            },
            {
                "name": "fineweb-hr",
                "src_path": str(paths.DATA_DIR / "raw/fineweb-2-hrv/data/hrv_Latn/"),
                "dst_path": str(paths.DATA_DIR / "processed/tokenizer-split/fineweb-2-hrv.txt"),
                "size_bytes": 800_000_000,
                "type": "hf",
            },
            {
                "name": "riznica",
                "src_path": str(paths.DATA_DIR / "interim/riznica/riznica.jsonl"),
                "dst_path": str(paths.DATA_DIR / "processed/tokenizer-split/riznica.txt"),
                "size_bytes": 200_000_000,
                "type": "json",
            },
            {
                "name": "classla_bs",
                "src_path": str(paths.DATA_DIR / "interim/classla-web-bs-v1/MaCoCu-Genre-DEDUPLICATED.bs.jsonl"),
                "dst_path": str(paths.DATA_DIR / "processed/tokenizer-split/MaCoCu-Genre-DEDUPLICATED.bs.txt"),
                "size_bytes": 200_000_000,
                "type": "json",
            },
            {
                "name": "fineweb_edu",
                "src_path": str(paths.DATA_DIR / "raw/fineweb-edu-10BT/sample/10BT"),
                "dst_path": str(paths.DATA_DIR / "processed/tokenizer-split/fineweb-edu.txt"),
                "size_bytes": 1_000_000_000,
                "type": "hf",
            },
            {
                "name": "fineweb_hq",
                "src_path": str(paths.DATA_DIR / "raw/fineweb-hq-20BT"),
                "dst_path": str(paths.DATA_DIR / "processed/tokenizer-split/fineweb-hq-20BT.txt"),
                "size_bytes": 1_000_000_000,
                "type": "hf",
            },
        ]
    )

    seed = 42

    vocab_size: int = 64000
    tokenizer_save_path: str = str(paths.EXPERIMENTS_DIR / "hr_en_bs_tokenizer_64000")


def load_config(argv: list[str] | None = None):
    OmegaConf.register_resolver("paths", lambda name: str(getattr(paths, name)))
    # No CLI parsing for now; just instantiate EvalConfig with defaults
    cfg = TokenizerTrainConfig()
    return cfg


def build_subset_from_json(src_path, dst_path, size_bytes):

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with (
        open(src_path, "r", encoding="utf-8") as src,
        open(dst_path, "w", encoding="utf-8") as dst,
    ):
        for line in src:
            sample = json.loads(line)
            # Process and write sample
            dst.write(sample["text"] + "\n")
            if os.path.getsize(dst_path) > size_bytes:
                break
        size_gb = os.path.getsize(dst_path) / (1024**3)
        print(f"{dst_path} \nFile size is {size_gb:.2f} GB")


def build_subset_from_hr(src_path, dst_path, seed, size_bytes):
    from datasets import load_dataset

    ds = load_dataset(path=str(src_path), split="train")
    ds = ds.shuffle(seed=seed)
    with open(dst_path, "w", encoding="utf-8") as dst:
        for sample in ds:
            dst.write(sample["text"] + "\n")
            if os.path.getsize(dst_path) > size_bytes:
                break
        size_gb = os.path.getsize(dst_path) / (1024**3)
        print(f"{dst_path} \nFile size is {size_gb:.2f} GB")


if __name__ == "__main__":
    cfg = load_config()

    for dataset in cfg.datasets:
        print(dataset)
        args = {k: dataset[k] for k in ("src_path", "dst_path", "size_bytes")}
        if dataset["type"] == "hf":
            build_subset_from_hr(**args, seed=cfg.seed)
        if dataset["type"] == "json":
            build_subset_from_json(**args)

    #merge all dataset out_paths into a single string and fit tokenizer

    # Collect dataset destination paths from cfg.datasets
    dst_paths = [dataset["dst_path"] for dataset in cfg.datasets]
    merged = Path(paths.DATA_DIR / "tokenizer-split/merged.txt")
    with open(merged, "w", encoding="utf-8") as out:
        for file_path in dst_paths:
            with open(file_path, encoding="utf-8") as fin:
                shutil.copyfileobj(fin, out)

    # # Fit the tokenizer on the concatenated string and save
    tokenizer = BytePairEncoding(vocab_size=cfg.vocab_size)
    tokenizer.fit(merged)
    tokenizer_save_path = str(cfg.tokenizer_save_path)
    tokenizer.save(tokenizer_save_path)
    print(f"Trained tokenizer saved to {tokenizer_save_path}")
