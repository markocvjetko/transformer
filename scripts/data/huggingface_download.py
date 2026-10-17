import os

from datasets import load_dataset
from src.utils import paths

# print("Downloading FineWeb-2 hrv_Latn subset (approx. 16 GB)...")
# # 1. Download the full dataset into your standard HF cache first
# dataset = load_dataset("HuggingFaceFW/fineweb-2", name="hrv_Latn")

# # 2. Save the compiled dataset directly to a specific folder
# # Replace './fineweb_hrv_local' with whatever path or external drive you want
# local_path = paths.DATA_DIR / "raw/fineweb_2_hrv_local"
# dataset.save_to_disk(local_path)

# print(f"Success! Dataset successfully saved to: {os.path.abspath(local_path)}")


# Force Hugging Face to never look for the internet or time out trying
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import load_from_disk

# Load the exact directory you populated in Step 1
local_path = paths.DATA_DIR / "raw/fineweb_2_hrv_local"
dataset = load_from_disk(local_path)

# Verify the data works offline
print("\n--- Offline Dataset Verification ---")
print(dataset)
print("\nFirst sample text preview:")
print(dataset["train"][0]["text"][:300])
