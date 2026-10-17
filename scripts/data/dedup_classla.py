import json

from tqdm import tqdm

from src.utils import paths

if __name__ == "__main__":
    # classla_v2 = paths.DATA_DIR / "raw/classla-web-hr-v2/CLASSLA-web.hr.2.0.jsonl"
    src_path = paths.DATA_DIR / "raw/classla-web-hr-v1/MaCoCu-Genre.hr.jsonl"
    dst_path = paths.DATA_DIR / "interim/classla-web-hr-v1/MaCoCu-Genre-DEDUPLICATED.hr.jsonl"
    duplicates_path = paths.DATA_DIR / "raw/CLASSLA-web.hr.2.0.duplicates_with_CLASSLA-web.hr.1.0.json"
    
    with open(duplicates_path, "r") as file:
        dup_dict = json.load(file)
    
    dup_ids = set([id for ids in dup_dict.values() for id in ids])
    print(f"total_ids to remove: {len(dup_ids)}")    
    removed = 0
    kept = 0

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(src_path, "r", encoding="utf-8") as src,
        open(dst_path, "w", encoding="utf-8") as dst
    ):
        for line in tqdm(src):
            j_content = json.loads(line)
            if j_content["id"] in dup_ids:
                removed += 1
            else:
                dst.write(line)
                kept += 1

        print(f"{removed} out of {len(dup_ids)} URLs removed")
