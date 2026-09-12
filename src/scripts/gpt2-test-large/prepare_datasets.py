import sys
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from datasets import load_dataset
from src.datasets.preprocess.sources import HFSource, JsonlSource
from src.datasets.preprocess.tokenize_corpus import create_npy_dataset
from src.tokenizers.BPE_v2 import BytePairEncoding
from src.utils import paths


@dataclass
class TokenizerConfig:
    path: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k"

@dataclass
class ClasslaConfig:
    path: Path = paths.DATA_DIR / "CLASSLA-web.hr.2.0.jsonl"
    field: str = "text"


@dataclass
class FinewebConfig:
    repo: str = "HuggingFaceFW/fineweb-edu"
    name: str = "sample-10BT"
    split: str = "train"
    cache_dir: Path = paths.DATA_DIR / "fineweb-edu"
    field: str = "text"

@dataclass
class Config:
    save_dir: Path = paths.DATA_DIR / "gpt2-test-large"
    max_docs: int | None = None  # debug cap; None = whole corpus

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    classla_v2: ClasslaConfig = field(default_factory=ClasslaConfig)
    fineweb_edu: FinewebConfig = field(default_factory=FinewebConfig)
 

def load_config(config_path: str | None, cli_overrides: list[str]) -> Config:
    cfg = OmegaConf.structured(Config)  # defaults from dataclass
    if config_path:
        cfg = OmegaConf.merge(
            cfg, OmegaConf.load(config_path)
        )  # external file overrides
    cfg = OmegaConf.merge(
        cfg, OmegaConf.from_dotlist(cli_overrides)
    )  # CLI overrides last
    return cfg



def main(config_path: str | None = None, overrides: list[str] | None = None):
    # usage: python <script_name>.py experiment.yaml batch_size=128 optimizer.lr=1e-3
    args = sys.argv[1:]
    yaml_path = args[0] if args and "=" not in args[0] else None
    cli_overrides = args[1:] if yaml_path else args
    config = load_config(yaml_path, cli_overrides)
    print(OmegaConf.to_yaml(config)) #log umjesto printa Logging python modul

    classla_v2 = JsonlSource(**config.classla_v2)
    
    fineweb_edu = HFSource(
        load_dataset(
            path=config.fineweb_edu.repo,
            name=config.fineweb_edu.name,
            split=config.fineweb_edu.split,
            cache_dir=config.fineweb_edu.cache_dir
        ),
        field=config.fineweb_edu.field
    )

    tokenizer = BytePairEncoding.from_file(config.tokenizer.path)

    create_npy_dataset(classla_v2, tokenizer, config.save_dir / "classla_v2")
    create_npy_dataset(fineweb_edu, tokenizer, config.save_dir / "fineweb-edu")

if __name__ == "__main__":
    
    main()