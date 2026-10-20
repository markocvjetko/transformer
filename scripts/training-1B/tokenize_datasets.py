import subprocess
import sys

from src.utils import paths

ROOT = paths.DATA_DIR
DST_ROOT = paths.DATA_DIR / "processed/1B-model-vocab-48000"

config = {
    "tokenizer_path": paths.EXPERIMENTS_DIR / "1B-model/hr_en_bs_tokenizer_48000",
    "n_workers": 96,
    "n_shards": 1024,
}
datasets = {
    ### BOSNIAN
    # "classla-web-bs-v1": {
    #     "input": ROOT / "interim/classla-web-bs-v1/MaCoCu-Genre-DEDUPLICATED.bs.jsonl",
    #     "output": DST_ROOT / "classla-web-bs-v1/",
    #     "encoder": "classla_bs",
    #     "type": "json",
    # },
    # "classla-web-bs-v2": {
    #     "input": ROOT / "raw/classla-web-bs-v2/CLASSLA-web.bs.2.0.jsonl",
    #     "output": DST_ROOT / "classla-web-bs-v2/",
    #     "encoder": "classla_bs",
    #     "type": "json",
    # },
    # "fineweb-2-bos": {
    #     "input": ROOT / "raw/fineweb-2-bos/data/bos_Latn/",
    #     "output": DST_ROOT / "fineweb-2-bos/",
    #     "encoder": "fineweb_bs",
    #     "type": "hf",
    # },
    # ### CROATIAN
    # "classla-web-hr-v1": {
    #     "input": ROOT / "interim/classla-web-hr-v1/MaCoCu-Genre-DEDUPLICATED.hr.jsonl",
    #     "output": DST_ROOT / "classla-web-hr-v1/",
    #     "encoder": "classla_hr",
    #     "type": "json",
    # },
    # "classla-web-hr-v2": {
    #     "input": ROOT / "raw/classla-web-hr-v2/CLASSLA-web.hr.2.0.jsonl",
    #     "output": DST_ROOT / "classla-web-hr-v2/",
    #     "encoder": "classla_hr",
    #     "type": "json",
    # },
    # "riznica": {
    #     "input": ROOT / "interim/riznica/riznica.jsonl",
    #     "output": DST_ROOT / "riznica/",
    #     "encoder": "classla_hr",
    #     "type": "json",
    # },
    # "fineweb-2-hrv": {
    #     "input": ROOT / "raw/fineweb-2-hrv/data/hrv_Latn/",
    #     "output": DST_ROOT / "fineweb-2-hrv/",
    #     "encoder": "fineweb_hr",
    #     "type": "hf",
    # },
    # ### ENGLISH
    # "fineweb_edu": {
    #     "input": ROOT / "raw/fineweb-edu-10BT/sample/10BT",
    #     "output": DST_ROOT / "fineweb-edu-10BT/",
    #     "encoder": "fineweb_en",
    #     "type": "hf",
    # },
    # "fineweb_hq": {
    #     "input": ROOT / "raw/fineweb-hq-20BT",
    #     "output": DST_ROOT / "fineweb-hq-20BT/",
    #     "encoder": "fineweb_en",
    #     "type": "hf",
    # },
    # ### BOSNIAN-ENGLISH
    # "macocu-bs-en-sent": {
    #     "input": ROOT / "interim/macocu-bs-en-sent/MacoCu-bs-en.jsonl",
    #     "output": DST_ROOT / "macocu-bs-en-sent/",
    #     "encoder": "macocu_bs_en",
    #     "type": "json",
    # },
    # "settimes-bs-en-v2": {
    #     "input": ROOT / "interim/settimes-bs-en-v2/SETIMES-bs-en.jsonl",
    #     "output": DST_ROOT / "settimes-bs-en-v2/",
    #     "encoder": "macocu_bs_en",
    #     "type": "json",
    # },
    # ### CROATIAN-ENGLISH
    # "settimes-en-hr-v2": {
    #     "input": ROOT / "interim/settimes-en-hr-v2/SETIMES-en-hr.jsonl",
    #     "output": DST_ROOT / "settimes-en-hr-v2/",
    #     "encoder": "macocu_hr_en",
    #     "type": "json",
    # },
    # "dgt-en-hr-v2021": {
    #     "input": ROOT / "interim/dgt-en-hr-v2021/DGT-en-hr.jsonl",
    #     "output": DST_ROOT / "dgt-en-hr-v2021/",
    #     "encoder": "macocu_hr_en",
    #     "type": "json",
    # },
    # "macocu-en-hr-sent": {
    #     "input": ROOT / "interim/macocu-en-hr-sent/MacoCu-en-hr.jsonl",
    #     "output": DST_ROOT / "macocu-en-hr-sent/",
    #     "encoder": "macocu_hr_en",
    #     "type": "json",
    # },


    ### EVALS
    "wikipedia-hr": {
        "input": ROOT / "raw/wikipedia-hr-20231101/20231101.hr",
        "output": DST_ROOT / "wikipedia-hr-20231101",
        "encoder": "fineweb_hr",
        "type": "hf",
    },
    "wikipedia-bs": {
        "input": ROOT / "raw/wikipedia-bs-20231101/20231101.bs",
        "output": DST_ROOT / "wikipedia-bs-20231101",
        "encoder": "fineweb_bs",
        "type": "hf",
    },
    "wikipedia-en": {
        "input": ROOT / "raw/wikipedia-en-20231101/20231101.en",
        "output": DST_ROOT / "wikipedia-en-20231101",
        "encoder": "fineweb_en",
        "type": "hf",
    },
}


def main():
    for name, d in datasets.items():
        print(f"=== {name} ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.datasets.preprocess.tokenize_dataset",
                f"n_shards={config['n_shards']}",
                f"n_workers={config['n_workers']}",
                f"dataset_path={d['input']}",
                f"save_path={d['output']}",
                f"encoder={d['encoder']}",
                f"source={d['type']}",
                f"tokenizer_path={config['tokenizer_path']}",
            ],
            check=True,
            cwd=paths.ROOT,
        )


if __name__ == "__main__":
    main()
