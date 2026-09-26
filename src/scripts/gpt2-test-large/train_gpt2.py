import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from omegaconf import II, OmegaConf
from torch.profiler import ProfilerActivity, schedule
from torch.utils.data import DataLoader

import lightning as L
from src.datasets.next_token import NextTokenPredictionDataset
from src.datasets.token_mixture import TokenMixtureDataset
from src.lightning.lit_gpt import LitGPT
from src.metrics.metrics import BitsPerByte
from src.models.gpt2 import GPT2
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths

print("lightning version:", L.__version__)
torch.set_float32_matmul_precision("medium")


@dataclass
class TokenizerConfig:
    path: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k_hr_en"


@dataclass
class ModelConfig:
    vocab_size: int = II("..vocab_size")
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 8
    N: int = 6
    seq_len: int = II("..seq_len")
    compile_model: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # adamw only
    kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-4,
            "weight_decay": 0.01,
        }
    )


@dataclass
class LRSchedulerConfig:
    name: str = "wsd"  # wsd or constant
    kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "max_iters": -1,  ### THIS IS COMPUTED ONCE THE EFFECTIVE BATCH SIZE IS KNOWN
            "warmup_ratio": 0.15,
            "decay_ratio": 0.15,
            "min_lr_ratio": 0.1,
        }
    )


### TRAINER CALLBACKS
@dataclass
class CheckpointConfig:
    enabled: bool = False
    save_dir: str = II("..save_dir")
    save_top_k: int = 3
    save_last: bool = True
    every_n_train_steps: int = 1
    train_step_ratio: float = 0.25
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
    enabled: bool = True  # Enable wandb for debug runs
    project: str = "gpt2-test-large"
    name: str = "debug-run"
    log_freq: int = 1  # Log every step for debugging


### TRAINER
@dataclass
class TrainerConfig:
    max_tokens: int = 1_000_000
    accelerator: str = "gpu"
    precision: str = "16-mixed"
    devices: int = -1  # -1 uses all available
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    gradient_clip_val: float | None = 0.5
    strategy: str = "auto"  # auto, ddp
    log_every_n_steps: int = 1
    val_check_interval: float = 0.25  # max_steps * val_check_interval
    limit_val_batches: int | None = 100
    resume_from: str | None = None  # None, path to .ckpt, or "last" to auto-pick
    profiler: str | None = None


@dataclass
class TrainDatasetConfig:
    paths: list[Path] = field(
        default_factory=lambda: [
            paths.DATA_DIR / "gpt2-test-large/classla_hr_v2/merged.npy",
            paths.DATA_DIR / "gpt2-test-large/fineweb-edu_data.npy",
            paths.DATA_DIR / "gpt2-test-large/macocu_hr_en_v2_data.npy",
        ]
    )
    weights: list[float] = field(
        default_factory=lambda: [
            0.48,
            0.48,
            0.04,
        ]
    )
    seq_len: int = II("..seq_len")

    shuffle: bool = False
    num_workers: int = 16
    batch_size: int = 32


@dataclass
class ValDatasetConfig:
    paths: list[Path] = field(
        default_factory=lambda: [
            paths.DATA_DIR / "gpt2-test-large/wikipedia_hr/merged.npy",
            paths.DATA_DIR / "gpt2-test-large/wikipedia_en/merged.npy",
        ]
    )
    names: list[str] = field(default_factory=lambda: ["wikipedia_hr", "wikipedia_en"])
    seq_len: int = II("..seq_len")


@dataclass
class Config:
    project_name: str = "gpt2-test-large"
    save_dir: Path = paths.EXPERIMENTS_DIR / "gpt2-test-large"

    seq_len: int = 256
    vocab_size: int = 48006

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    train_dataset: TrainDatasetConfig = field(default_factory=TrainDatasetConfig)
    val_dataset: ValDatasetConfig = field(default_factory=ValDatasetConfig)


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

    ds_train = TokenMixtureDataset(cfg.train_dataset.paths, cfg.train_dataset.weights, cfg.train_dataset.seq_len)
    ds_val = [NextTokenPredictionDataset(path, cfg.val_dataset.seq_len) for path in cfg.val_dataset.paths]

    def collate_fn(batch: np.array):
        return torch.tensor(np.stack(batch, axis=0), dtype=torch.long)

    dataloader_train = DataLoader(
        ds_train,
        batch_size=cfg.train_dataset.batch_size,
        num_workers=cfg.train_dataset.num_workers,
        shuffle=cfg.train_dataset.shuffle,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # Sample a single element from the DataLoader and print its shape
    sample_batch = next(iter(dataloader_train))
    print(f"Sample batch shape: {sample_batch.shape}")

    dataloader_val = [
        DataLoader(
            ds,
            batch_size=cfg.train_dataset.batch_size,
            num_workers=cfg.train_dataset.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=False,
        )
        for ds in ds_val
    ]

    model = GPT2(
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.model.seq_len,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        n_heads=cfg.model.n_heads,
        N=cfg.model.N,
    )

    if cfg.model.compile_model:
        model = torch.compile(model)

    tokenizer = BytePairEncoding.from_file(cfg.tokenizer.path)
    bits_per_byte = BitsPerByte(tokenizer.vocab, tokenizer._special_vocab)

    num_devices = cfg.trainer.devices if cfg.trainer.devices > 0 else torch.cuda.device_count()
    print(f"[DEBUG] batch_size: {cfg.train_dataset.batch_size}")
    print(f"[DEBUG] seq_len: {cfg.train_dataset.seq_len}")
    print(f"[DEBUG] num_devices: {num_devices} (max(1, num_devices) = {max(1, num_devices)})")
    print(f"[DEBUG] accumulate_grad_batches: {cfg.trainer.accumulate_grad_batches}")

    tokens_per_batch = cfg.train_dataset.batch_size * cfg.train_dataset.seq_len * max(1, num_devices)
    tokens_per_optim_step = tokens_per_batch * cfg.trainer.accumulate_grad_batches
    print(f"[DEBUG] eff_tok_per_batch: {tokens_per_optim_step}")
    max_steps = cfg.trainer.max_tokens // tokens_per_optim_step
    cfg.lr_scheduler.kwargs["max_iters"] = max_steps

    lit_model = LitGPT(
        transformer=model,
        optim_name=cfg.optimizer.name,
        optim_kwargs=cfg.optimizer.kwargs,
        lr_scheduler_name=cfg.lr_scheduler.name,
        lr_scheduler_kwargs=cfg.lr_scheduler.kwargs,
        bits_per_byte=bits_per_byte,
        val_dataset_names=cfg.val_dataset.names,
    )
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    if cfg.checkpoint.enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint.save_dir,
                save_top_k=cfg.checkpoint.save_top_k,
                save_last=cfg.checkpoint.save_last,
                save_on_train_epoch_end=False,
                every_n_epochs=1,
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
        wandb_logger = WandbLogger(
            name=cfg.wandb.name,
            save_dir=cfg.save_dir,
            project=cfg.wandb.project,
            offline=os.environ.get("WANDB_MODE", "online") == "offline",
        )
        wandb_logger.watch(
            lit_model.transformer,
            log_freq=cfg.wandb.log_freq * cfg.trainer.accumulate_grad_batches,
            log="gradients",
            log_graph=False,
        )
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        loggers.append(wandb_logger)

    if cfg.trainer.profiler == "pytorch":
        profiler = PyTorchProfiler(
            dirpath=cfg.save_dir,
            filename="perf",
            export_to_chrome=True,
            record_module_names=True,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        )
    else:
        profiler = None

    print(f"training for {max_steps} batches across {num_devices} device(s)")
    print(
        f"Effective token count per batch: {tokens_per_optim_step} "
        f"(batch_size={cfg.train_dataset.batch_size} × "
        f"seq_len={cfg.train_dataset.seq_len} × "
        f"num_devices={num_devices} × "
        f"grad_accum={cfg.trainer.accumulate_grad_batches})"
    )

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        max_steps=max_steps,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=int(max_steps * cfg.trainer.val_check_interval) * cfg.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.trainer.limit_val_batches,
        enable_checkpointing=cfg.checkpoint.enabled,
        callbacks=callbacks,
        logger=loggers,
        profiler=profiler,
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
