import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import II, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from datasets import load_dataset
from src.datasets.hrenwac_v2 import HrenWac
from src.datasets.translation import TranslationDataset, collate_fn
from src.models.transformer import Transformer
from src.tokenizers.BPE import BytePairEncoding
from src.train.lit_transformer import LitTransformer
from src.utils import paths


@dataclass
class TokenizerConfig:
    path: Path = paths.EXPERIMENTS_DIR / "en-hr-tokenizers/tokenizer_37000.json"


@dataclass
class ModelConfig:
    vocab_size: int = II("..vocab_size")
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 8
    N: int = 6
    max_len: int = II("..seq_len")


@dataclass
class OptimizerConfig:
    name: str = "muon"
    lr: float = 1e-3
    weight_decay: float = 0.1


### TRAINER CALLBACKS
@dataclass
class CheckpointConfig:
    save_dir: str = II("..save_dir")
    save_top_k: int = 3
    save_last: bool = True
    every_n_epochs: int = 1
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "en-hr-translation"
    name: str = "default"


### TRAINER
@dataclass
class TrainerConfig:
    max_epochs: int = 200
    accelerator: str = "gpu"
    devices: int = -1  # -1 uses all available
    num_nodes: int = 1
    strategy: str = "ddp"
    log_every_n_steps: int = 100
    check_val_every_n_epoch: int = 5
    early_stopping_patience: int = 5

    resume_from: str | None = None  # None, path to .ckpt, or "last" to auto-pick


@dataclass
class DatasetConfig:
    opus: Path = paths.DATA_DIR / "Helsinki-NLP__opus-100"
    hrenwac: Path = paths.DATA_DIR / "hrenwac"
    jwl_300: Path = paths.DATA_DIR / "sentence-transformers/parallel-sentences-jw300"
    seq_len: int = II("..seq_len")
    src_lang: str = "en"
    tgt_lang: str = "hr"
    num_workers: int = 10
    batch_size: int = 2


@dataclass
class Config:
    project_name: str = "en-hr"
    save_dir: Path = paths.EXPERIMENTS_DIR / "en-hr"

    seq_len: int = 256
    vocab_size: int = 37000

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


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


def main():

    # usage: python <script_name>.py experiment.yaml batch_size=128 optimizer.lr=1e-3
    args = sys.argv[1:]
    yaml_path = args[0] if args and "=" not in args[0] else None
    cli_overrides = args[1:] if yaml_path else args
    cfg = load_config(yaml_path, cli_overrides)
    print(OmegaConf.to_yaml(cfg))

    opus = load_dataset("Helsinki-NLP/opus-100", "en-hr", cache_dir=cfg.dataset.opus)
    jwl = load_dataset(
        "sentence-transformers/parallel-sentences-jw300",
        "en-hr",
        cache_dir=cfg.dataset.jwl_300,
    )
    jwl = jwl.rename_columns({"english": "en", "non_english": "hr"})
    hrenwac = HrenWac(path=cfg.dataset.hrenwac)

    print(len(opus["train"]))
    print(len(jwl["train"]))
    print(len(hrenwac))

    print(opus["train"]["translation"])
    print(jwl["train"])
    print(hrenwac.data[0].keys())
    ds_train = ConcatDataset([opus["train"]["translation"], jwl["train"], hrenwac])
    ds_val = ConcatDataset([opus["validation"]["translation"]])
    print(len(ds_train))
    print(len(ds_val))

    tokenizer = BytePairEncoding.from_file(cfg.tokenizer.path)

    dataset_train = TranslationDataset(
        ds_train,
        src_lang=cfg.dataset.src_lang,
        tgt_lang=cfg.dataset.tgt_lang,
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        max_len=cfg.dataset.seq_len,
    )

    dataset_val = TranslationDataset(
        ds_val,
        src_lang=cfg.dataset.src_lang,
        tgt_lang=cfg.dataset.tgt_lang,
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        max_len=cfg.dataset.seq_len,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = Transformer(
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.model.max_len,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        n_heads=cfg.model.n_heads,
        N=cfg.model.N,
        # pad_token_id=cfg.model.pad_token_id,
        # eos_token_id=cfg.model.eos_token_id,
        # p_dropout=cfg.model.dropout,
    )

    lit_model = LitTransformer(
        transformer=model,
        optim=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=cfg.checkpoint.save_dir,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            every_n_epochs=cfg.checkpoint.every_n_epochs,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
        )
    )
    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.early_stopping.monitor,
                mode=cfg.early_stopping.mode,
                patience=cfg.early_stopping.patience,
            )
        )

    loggers = []
    loggers.append(
        CSVLogger(save_dir=cfg.save_dir),
    )
    if cfg.wandb.enabled:
        loggers.append(
            WandbLogger(
                name=cfg.project_name,
                save_dir=cfg.save_dir,
                project=cfg.project_name,
                offline=True,
                # offline=os.environ.get("WANDB_MODE", "online") == "offline",
            )
        )

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=loggers,
    )
    try:
        # near the end of main(), replacing the trainer.fit line
        ckpt_path = cfg.trainer.resume_from
        if ckpt_path == "last":
            ckpt_path = (
                "last"  # Lightning resolves this against ModelCheckpoint dirpath
            )
        elif ckpt_path:
            ckpt_path = str(Path(ckpt_path).expanduser().resolve())

        trainer.fit(lit_model, dataloader_train, dataloader_val, ckpt_path=ckpt_path)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        run = wandb.run
        if run is not None and os.environ.get("WANDB_MODE", "online") == "online":
            wandb.Api().run(f"{run.entity}/{run.project}/{run.id}").delete()
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
