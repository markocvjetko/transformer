# Download all presets:
#
# for preset in fineweb-2-bs fineweb-2-hr fineweb-edu-10BT; do
#     python scripts/data/huggingface_download.py $preset
# done
# for preset in wikipedia-hr wikipedia-bs wikipedia-en; do
#     python scripts/data/huggingface_download.py $preset
# done
# for preset in flores-bs flores-hr flores-en; do
#     python scripts/data/huggingface_download.py $preset
# done
# for preset in eu-mmlu-en eu-mmlu-hr belebele-hr belebele-en; do
#     python scripts/data/huggingface_download.py $preset
# done
import os
import sys

from huggingface_hub import snapshot_download

from datasets import load_dataset
from src.utils import paths

presets = {
    # FINEWEBs
    "fineweb-2-bs": ("HuggingFaceFW/fineweb-2", "data/bos_Latn/*", "raw/fineweb-2-bos"),
    "fineweb-2-hr": ("HuggingFaceFW/fineweb-2", "data/hrv_Latn/*", "raw/fineweb-2-hrv"),
    "fineweb-edu-10BT": ("HuggingFaceFW/fineweb-edu", "sample/10BT/*", "raw/fineweb-edu-10BT"),
    # WIKIPEDIA DUMPS:
    "wikipedia-hr": ("wikimedia/wikipedia", "20231101.hr/*", "raw/wikipedia-hr-20231101"),
    "wikipedia-bs": ("wikimedia/wikipedia", "20231101.bs/*", "raw/wikipedia-bs-20231101"),
    "wikipedia-en": ("wikimedia/wikipedia", "20231101.en/*", "raw/wikipedia-en-20231101"),
    # FLORES+ DEV SPLIT
    "flores-bs": ("openlanguagedata/flores_plus", "dev/bos_Latn.jsonl", "raw/flores-bs"),
    "flores-hr": ("openlanguagedata/flores_plus", "dev/hrv_Latn.jsonl", "raw/flores-hr"),
    "flores-en": ("openlanguagedata/flores_plus", "dev/eng_Latn.jsonl", "raw/flores-en"),
    # EVALS
    "eu-mmlu-hr": ("EC-DGT-AI/EU-MMLU", "data/*/HR_HR/*", "raw/eu-mmlu-hr"),
    "eu-mmlu-en": ("EC-DGT-AI/EU-MMLU", "data/*/EN_GB/*", "raw/eu-mmlu-en"),
    "belebele-en": ("facebook/belebele", "data/eng_Latn.jsonl", "raw/belebele-en"),
    "belebele-hr": ("facebook/belebele", "data/hrv_Latn.jsonl", "raw/belebele-hr"),
    "flores-bs-en": ("facebook/flores", "data/pair/bos_Latn-eng_Latn/*", "raw/flores-bs-en"),
    "flores-en-hr": ("facebook/flores", "data/pair/hrv_Latn-eng_Latn/*", "raw/flores-en-hr"),
    "copa-hr": ("classla/copa_hr", "data.zip", "raw/copa-hr"),
    "copa-en": ("pkavumba/balanced-copa", "*.csv", "raw/copa-en"),
#copa = load_dataset("aps/super_glue", "copa", "raw/copa-en")
    
}


#FOR COPA-EN, I DELETED THE TEST SET SINCE IT'S LABELS ARE MASKED

preset = sys.argv[1] if len(sys.argv) > 1 else "fineweb-2-bos"
REPO, PATTERN, SAVE_DIR = presets[preset]




if REPO == "classla/copa_hr":
    import zipfile

    data_dir = paths.DATA_DIR / SAVE_DIR

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns="data.zip",
        local_dir=data_dir,
    )

    with zipfile.ZipFile(data_dir / "data.zip") as z:
        for file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            z.extract(file, data_dir / "eval")

    dataset = load_dataset(str(data_dir / "eval"))
    print(dataset[0])


else:
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=PATTERN,
        local_dir=paths.DATA_DIR / SAVE_DIR,
    )

    #dataset = load_dataset(str(paths.DATA_DIR / SAVE_DIR / "copa"))
    dataset = load_dataset(str(paths.DATA_DIR / SAVE_DIR))

