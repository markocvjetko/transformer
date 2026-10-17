import numpy as np

from src.datasets.token_mixture import TokenMixtureDataset
from src.utils import paths

dataset_paths = [
    paths.DATA_DIR / "gpt2-test-large/classla_hr_v2/merged.npy",
    paths.DATA_DIR / "gpt2-test-large/merged_fineweb_data.npy",
    paths.DATA_DIR / "gpt2-test-large/macocu_hr_en_v2_data.npy",
]
weights = [0.47, 0.47, 0.06]
seq_len = 256

# Calculate lengths and oversample factors for a fixed token count
target_tokens = 10_000_000_000

# Prepare data for pretty tabular output
rows = []
for p, w in zip(dataset_paths, weights):
    n_tokens = len(np.load(p, mmap_mode="r"))
    oversample = (target_tokens * w) / n_tokens
    rows.append([p.name, n_tokens, f"x{oversample:.2f}"])

# Print header
print(f"{'Dataset':35} | {'# Sequences':>12} | {'Oversample':>12}")
print("-" * 65)
for row in rows:
    print(f"{row[0]:35} | {row[1]:12,} | {row[2]:>12}")


dataset = TokenMixtureDataset(dataset_paths, weights, seq_len)
