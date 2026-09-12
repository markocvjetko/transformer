import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import II, OmegaConf
from src.tokenizers.BPE_v2 import BytePairEncoding
from torch.utils.data import ConcatDataset, DataLoader

from src.datasets.next_token import NextTokenPredictionDataset
from src.metrics.metrics import BitsPerByte
from src.models.gpt2 import GPT2
from src.train.lit_gpt import LitGPT
from src.utils import paths

print("lightning version:", L.__version__)
torch.set_float32_matmul_precision('medium')


@dataclass
class TokenizerConfig:
    path: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k"


@dataclass
class ModelConfig:
    vocab_size: int = II("..vocab_size")
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 8
    N: int = 12
    max_len: int = II("..seq_len")
    compile_model: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adamw" # only adamw supported atm
    lr: float = 1e-4
    weight_decay: float = 0.01


### TRAINER CALLBACKS
@dataclass
class CheckpointConfig:
    enabled: bool = False
    save_dir: str = II("..save_dir")
    save_top_k: int = 3
    save_last: bool = True
    every_n_epochs: int = 1
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "gpt-test-large"
    name: str = "default"


### TRAINER
@dataclass
class TrainerConfig:
    max_epochs: int = 1000
    accelerator: str = "gpu"
    precision: str = "bf16-mixed"
    devices: int = -1  # -1 uses all available
    num_nodes: int = 1
    strategy: str = "auto" # auto, ddp
    log_every_n_steps: int = 100
    check_val_every_n_epoch: int = 1
    early_stopping_patience: int = 5
    resume_from: str | None = None  # None, path to .ckpt, or "last" to auto-pick


@dataclass
class DatasetConfig:
    classla_v2_hr: Path = paths.DATA_DIR / "gpt2-test-large/classla_v2_data.npy"
    fineweb_edu: Path = paths.DATA_DIR / "gpt2-test-large/fineweb-edu_data.npy"
    seq_len: int = II("..seq_len")

    shuffle: bool = False
    num_workers: int = 16
    batch_size: int = 32


@dataclass
class Config:
    project_name: str = "gpt2-test-large"
    save_dir: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large"

    seq_len: int = 256
    vocab_size: int = 48004

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


# resolves ${paths:DATA_DIR} / ${paths:EXPERIMENTS_DIR} in yaml configs
OmegaConf.register_new_resolver("paths", lambda name: str(getattr(paths, name)), replace=True)


def load_config(config_path: str | None, cli_overrides: list[str]) -> Config:
    cfg = OmegaConf.structured(Config)  # defaults from dataclass
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))  # external file overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))  # CLI overrides last
    return cfg


def main():
    
    # usage: python <script_name>.py experiment.yaml batch_size=128 optimizer.lr=1e-3
    args = sys.argv[1:]
    yaml_path = args[0] if args and "=" not in args[0] else None
    cli_overrides = args[1:] if yaml_path else args
    cfg = load_config(yaml_path, cli_overrides)
    print(OmegaConf.to_yaml(cfg))  # log umjesto printa Logging python modul

    dataset_fineweb = NextTokenPredictionDataset(dataset_path=cfg.dataset.fineweb_edu, seq_len=cfg.dataset.seq_len)

    dataset_classla_v2_hr = NextTokenPredictionDataset(
        dataset_path=cfg.dataset.classla_v2_hr, seq_len=cfg.dataset.seq_len
    )

    from torch.utils.data import Subset

    ds_train_full = ConcatDataset([dataset_fineweb, dataset_classla_v2_hr])
    # Use 100,000 tokens for train, next 100,000 for val
    ds_train = Subset(ds_train_full, list(range(min(70000, len(ds_train_full)))))
    ds_val = Subset(ds_train_full, list(range(70000, min(210_000, len(ds_train_full)))))
    print(len(ds_train))
    print(len(ds_val))
    def collate_fn(batch: np.array):
        return torch.tensor(np.stack(batch, axis=0), dtype=torch.long)
   

    dataloader_train = DataLoader(
        ds_train,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=cfg.dataset.shuffle,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    dataloader_val = DataLoader(
        ds_val,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    model = GPT2(
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

    if cfg.model.compile_model:
        model = torch.compile(model)

    tokenizer = BytePairEncoding.from_file(cfg.tokenizer.path)
    bits_per_byte = BitsPerByte(tokenizer.vocab)

    lit_model = LitGPT(
        transformer=model,
        optim=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        bits_per_byte = bits_per_byte,
        weight_decay=cfg.optimizer.weight_decay,
    )
    callbacks = []
    if cfg.checkpoint.enabled:
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
                #offline=True,
                offline=os.environ.get("WANDB_MODE", "online") == "offline",
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
        enable_checkpointing=cfg.checkpoint.enabled,
        callbacks=callbacks,
        logger=loggers,
    )
    try:
        # near the end of main(), replacing the trainer.fit line
        ckpt_path = cfg.trainer.resume_from
        if ckpt_path == "last":
            ckpt_path = "last"  # Lightning resolves this against ModelCheckpoint dirpath
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
