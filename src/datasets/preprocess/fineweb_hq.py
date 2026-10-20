from datasets import Dataset, load_dataset
from src.utils import paths
from tqdm import tqdm

import os

os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
print("HF_TOKEN:", os.environ.get("HF_TOKEN"))
print("HF_HOME  :", os.environ.get("HF_HOME"))
print("HF_DATASETS_OFFLINE:", os.environ.get("HF_DATASETS_OFFLINE"))
print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE"))

output_path = paths.DATA_DIR / "raw/fineweb-hq-20BT"

edu = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
edu_ids = set()
print("Collecting edu IDs with tqdm progress bar...")
for doc in tqdm(edu, desc="Collecting edu IDs", unit="docs", dynamic_ncols=True):
    edu_ids.add(doc["id"])

hq = load_dataset("epfml/FineWeb-HQ", split="train", streaming=True)

TARGET_TOKENS = 20_000_000_000

def filtered_with_tqdm():
    total = 0
    rejected = 0
    with tqdm(total=TARGET_TOKENS, unit="tokens", dynamic_ncols=True) as pbar:
        for doc in hq:
            if doc["id"] in edu_ids:
                rejected += 1
                continue
            yield doc
            total += doc["token_count"]
            pbar.update(doc["token_count"])
            if total >= TARGET_TOKENS:
                break
    print(f"Number of rejected docs: {rejected}")

Dataset.from_generator(filtered_with_tqdm).save_to_disk(output_path)
