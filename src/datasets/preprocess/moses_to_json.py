"""
Merges two line-aligned parallel files (Moses format) into one jsonl,
one {lang1: ..., lang2: ...} object per line.

Usage:
    python -m src.datasets.preprocess.moses_to_json preset=settimes_hr_en
    python -m src.datasets.preprocess.moses_to_json \
        config=src/datasets/preprocess/moses_to_json.yaml [key=value ...]

Precedence: defaults < preset < config file < cli.
"""
# The following bash script (example usage) runs this script for several presets:
#
# for preset in settimes_hr_en settimes_bs_en macocu_hr_en macocu_bs_en dgt_hr_en; do
#     python -m src.datasets.preprocess.moses_to_json preset=$preset
# done

import json
from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING, OmegaConf

from src.utils import paths

RAW = paths.DATA_DIR / "raw"
INTERIM = paths.DATA_DIR / "interim"

PRESETS = {
    "settimes_hr_en": dict(
        file1=RAW / "settimes-hr-en-v2/SETIMES.en-hr.hr",
        file2=RAW / "settimes-hr-en-v2/SETIMES.en-hr.en",
        lang1="hr",
        lang2="en",
        output=INTERIM / "settimes-en-hr-v2/SETIMES-en-hr.jsonl",
    ),
    "settimes_bs_en": dict(
        file1=RAW / "settimes-bs-en-v2/SETIMES.bs-en.bs",
        file2=RAW / "settimes-bs-en-v2/SETIMES.bs-en.en",
        lang1="bs",
        lang2="en",
        output=INTERIM / "settimes-bs-en-v2/SETIMES-bs-en.jsonl",
    ),
    "macocu_hr_en": dict(
        file1=RAW / "macocu-hr-en-sent/MaCoCu.en-hr.hr",
        file2=RAW / "macocu-hr-en-sent/MaCoCu.en-hr.en",
        lang1="hr",
        lang2="en",
        output=INTERIM / "macocu-en-hr-sent/MacoCu-en-hr.jsonl",
    ),
    "macocu_bs_en": dict(
        file1=RAW / "macocu-bs-en-sent/MaCoCu.bs_Latn-en.bs_Latn",
        file2=RAW / "macocu-bs-en-sent/MaCoCu.bs_Latn-en.en",
        lang1="bs",
        lang2="en",
        output=INTERIM / "macocu-bs-en-sent/MacoCu-bs-en.jsonl",
    ),
    "dgt_hr_en": dict(
        file1=RAW / "dgt-hr-en-v2021/DGT.en-hr.hr",
        file2=RAW / "dgt-hr-en-v2021/DGT.en-hr.en",
        lang1="hr",
        lang2="en",
        output=INTERIM / "dgt-en-hr-v2021/DGT-en-hr.jsonl",
    ),
}


@dataclass
class Config:
    file1: Path = MISSING
    file2: Path = MISSING
    lang1: str = MISSING
    lang2: str = MISSING
    output: Path = MISSING


def load_config(argv: list[str] | None = None) -> Config:
    OmegaConf.register_resolver("paths", lambda name: str(getattr(paths, name)))
    cli = OmegaConf.from_cli(argv)
    cfg_path = cli.pop("config", None)
    preset = cli.pop("preset", None)
    cfg = OmegaConf.structured(Config)
    if preset:
        cfg = OmegaConf.merge(cfg, PRESETS[preset])
    if cfg_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))
    cfg = OmegaConf.merge(cfg, cli)  # cli overrides config file
    return OmegaConf.to_object(cfg)


def main():
    cfg = load_config()
    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(cfg.file1, encoding="utf-8") as f1,
        open(cfg.file2, encoding="utf-8") as f2,
        open(cfg.output, "w", encoding="utf-8") as out,
    ):
        for l1, l2 in zip(f1, f2, strict=True):
            doc = {cfg.lang1: l1.rstrip("\n"), cfg.lang2: l2.rstrip("\n")}
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
