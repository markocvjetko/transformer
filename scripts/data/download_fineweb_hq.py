import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from src.utils import paths

# Ensure HF works online (override any env set elsewhere)
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# print("HF_TOKEN:", os.environ.get("HF_TOKEN"))
print("HF_HOME  :", os.environ.get("HF_HOME"))
print("HF_DATASETS_OFFLINE:", os.environ.get("HF_DATASETS_OFFLINE"))
print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE"))


output_path = paths.DATA_DIR / "raw" / "fineweb-hq-20BT"
edu_disk_path = paths.DATA_DIR / "raw" / "fineweb-edu-10BT"
tmp_dir = output_path / "_download"

print("Collecting edu IDs...")
edu_ids = pa.concat_arrays(
    [
        pq.read_table(f, columns=["id"])["id"].combine_chunks()
        for f in tqdm(sorted(edu_disk_path.rglob("*.parquet")), desc="Collecting edu IDs", unit="files")
    ]
)

hq_files = sorted(f for f in HfApi().list_repo_files("epfml/FineWeb-HQ", repo_type="dataset") if f.endswith(".parquet"))

TARGET_TOKENS = 20_000_000_000

W = 16

# Resume: keep finished output shards, redo the last one (may be partially written)
done = sorted(output_path.glob("*.parquet"))
if done:
    done.pop().unlink()
resume = len(done)
total = sum(pc.sum(pq.read_table(f, columns=["token_count"])["token_count"]).as_py() for f in done)
print(f"Resuming at shard {resume} with {total:,} tokens")


def process(f):
    local = hf_hub_download("epfml/FineWeb-HQ", f, repo_type="dataset", local_dir=tmp_dir)
    t = pq.read_table(local, columns=[c for c in pq.read_schema(local).names if c != "embeddings"])
    os.remove(local)
    n = len(t)
    t = t.filter(pc.invert(pc.is_in(t["id"], value_set=edu_ids)))
    return t, n - len(t)


rejected = 0

with tqdm(total=TARGET_TOKENS, initial=total, unit="tokens", dynamic_ncols=True) as pbar, ThreadPoolExecutor(W) as ex:
    for start in range(resume, len(hq_files), W):
        for i, (t, r) in enumerate(ex.map(process, hq_files[start : start + W]), start):
            rejected += r
            # Keep docs until the target is reached (including the one crossing it)
            keep = pc.sum(pc.less(pc.cumulative_sum(t["token_count"]), TARGET_TOKENS - total)).as_py() or 0
            t = t.slice(0, keep + (keep < len(t)))
            pq.write_table(t, output_path / f"{i:05d}.parquet")
            tokens = pc.sum(t["token_count"]).as_py() or 0
            total += tokens
            pbar.update(tokens)
            if total >= TARGET_TOKENS:
                break
        if total >= TARGET_TOKENS:
            break
