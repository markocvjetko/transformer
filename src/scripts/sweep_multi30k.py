"""
A simple hyperparameter sweep script with Optuna.
"""
import json
import os
import subprocess
import torch
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_from_disk
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import wandb

from src.datasets.translation import TranslationDataset, collate_fn
from src.models.transformer import Transformer
from src.tokenizers.BPE import BytePairEncoding
from src.train.lit_transformer import LitTransformer
from src.utils import paths


def objective(trial: optuna.Trial, config):
    #suggest params,

    vocab_size_exp =  trial.suggest_int("vocab_size_exp", 8, 14)
    vocab_size = 2 ** vocab_size_exp
    d_model_exp = trial.suggest_int("d_model_exp", 5, 8) 
    d_model = 2 ** d_model_exp
    d_ff_ratio = trial.suggest_int("d_ff_ration", 1, 4)
    d_ff = d_model * d_ff_ratio

    n_heads_exp = trial.suggest_int("n_heads", 1, 5)
    n_heads = 2 ** n_heads_exp
    N = trial.suggest_int("N", 1, 6)
    lr = trial.suggest_float("lr", 10e-5, 10e-3)
    batch_size_exp = trial.suggest_int("batch_size_exp", 6, 11)
    batch_size = 2 ** batch_size_exp

    dataset = load_from_disk(str(paths.DATA_DIR / "multi30k"))
    ds_train = dataset["train"]
    ds_val = dataset["validation"]

    text_en = " ".join([example["en"] for example in ds_train])
    text_de = " ".join([example["de"] for example in ds_train])
    text = " ".join([text_en, text_de])

    #train tokenizer
    tokenizer = BytePairEncoding(vocab_size)
    tokenizer.fit(text)
    tokenizer.save(config["sweep_root"] / str(trial.number) / "tokenizer.json")
    
    dataset_train = TranslationDataset(
        ds_train, 
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        max_len=config["seq_len"],
        )

    dataset_val = TranslationDataset(
        ds_val, 
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        max_len=config["seq_len"])

    # Minimal dataloader
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn)

    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn)

    model = Transformer(
        vocab_size=vocab_size,
        seq_len=config["seq_len"],
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        N=N,
        pad_token_id=0,
        eos_token_id=2,
        p_dropout=0.1
    )

    lit_model = LitTransformer(model, lr)

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=config["max_epochs"],
        log_every_n_steps=10,    
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        callbacks=[
            ModelCheckpoint(
                dirpath=config["sweep_root"] / str(trial.number), 
                save_last=True, 
                every_n_epochs=1, 
                save_top_k=3,
                monitor="val_loss"),
            EarlyStopping(monitor="val_loss", mode="min"),
            PyTorchLightningPruningCallback(trial, monitor="val_loss")
        ],
        logger=[
            CSVLogger(save_dir=config["sweep_root"] / str(trial.number)), 
            WandbLogger(
                name=str(trial.number),
                save_dir=config["sweep_root"] /str(trial.number),
                project=config["sweep_name"],
                offline=True)
            #TensorBoardLogger(save_dir=config["sweep_root"] / str(trial.number))
        ],
    )
    try:
        trainer.fit(lit_model, dataloader_train, dataloader_val)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()
    finally:
        wandb.finish()
        
    return trainer.callback_metrics["val_loss"].item()
    
def main(git_hash: str):
    config = {
        "sweep_name": "multi30k",
        "sweep_root": paths.EXPERIMENTS_DIR / "sweeps" / "multi30k",
        "dataset_root": "bentrevett/multi30k",
        "seq_len": 256,
        "check_val_every_n_epoch": 10,
        "max_epochs": 500,
        "num_workers": 16,
        "early_stopping_patience": 25,
    }
    os.makedirs(config["sweep_root"], exist_ok=True)
    with open(config["sweep_root"] / "git_hash", mode="w") as f:
        f.write(git_hash)
    with open(config["sweep_root"] / "config.json", "w") as f:
        json.dump(config, f, default=str)

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(config["sweep_root"] / "optuna.log"))
    )
    study = optuna.create_study(
        study_name="multi30k_test",
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
        load_if_exists=True,
    )
    study.optimize(partial(objective, config=config), n_trials=100, timeout=None, n_jobs=1)


if __name__ == "__main__":
    main(git_hash=os.environ.get("GIT_HASH", "unknown"))