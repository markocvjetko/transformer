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
import os
import sys

from huggingface_hub import snapshot_download

from datasets import load_dataset, load_from_disk
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
}

preset = sys.argv[1] if len(sys.argv) > 1 else "fineweb-2-bos"
REPO, PATTERN, SAVE_DIR = presets[preset]

snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=PATTERN,
    local_dir=paths.DATA_DIR / SAVE_DIR,
)
