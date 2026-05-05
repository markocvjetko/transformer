import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from optuna.integration import PyTorchLightningPruningCallback
import wandb
from torch.utils.data import ConcatDataset

from src.datasets.hrenwac_v2 import HrenWac
from src.datasets.translation import TranslationDataset, collate_fn
from src.models.transformer import Transformer
from src.tokenizers.BPE import BytePairEncoding
from src.train.lit_transformer import LitTransformer
from src.utils import paths

def main():

    config = {
        "tokenizer": {
            "vocab_size":37000,
            "min_frequency":100
        },
        "project_name": "en-hr",
        "save_dir": paths.EXPERIMENTS_DIR / "en-hr",
        "tokenizer_path": paths.EXPERIMENTS_DIR / "en-hr-tokenizers" / "tokenizer_37000.json",
        "opus": paths.DATA_DIR / "Helsinki-NLP__opus-100",
        "hrenwac": paths.DATA_DIR / "hrenwac",
        "jwl_300": paths.DATA_DIR / "sentence-transformers/parallel-sentences-jw300",
        "seq_len": 256,
        "src_lang": "en",
        "tgt_lang": "hr",

        "check_val_every_n_epoch": 1,
        "max_epochs": 3,
        "num_workers": 10,
        "early_stopping_patience": 5,
    }


    vocab_size = 37000
    d_model = 256
    d_ff = 512
    n_heads = 8
    N = 6
    lr = 3e-4
    batch_size = 128

    opus = load_dataset("Helsinki-NLP/opus-100", "en-hr", cache_dir=config["opus"])
    jwl = load_dataset("sentence-transformers/parallel-sentences-jw300", "en-hr", cache_dir=config["jwl_300"])
    jwl = jwl.rename_columns({"english":"en", "non_english":"hr"})
    hrenwac = HrenWac(path=config["hrenwac"])

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

    # hrenwac = HrenWac(path=config["hrenwac"])
    # ds_train = ConcatDataset([hrenwac])
    # ds_val = ConcatDataset([hrenwac])
   

    tokenizer = BytePairEncoding.from_file(config["tokenizer_path"])

    dataset_train = TranslationDataset(
        ds_train, 
        src_lang=config["src_lang"],
        tgt_lang=config["tgt_lang"],
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        max_len=config["seq_len"],
        )

    dataset_val = TranslationDataset(
        ds_val, 
        src_lang=config["src_lang"],
        tgt_lang=config["tgt_lang"],
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        max_len=config["seq_len"])


    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn)


    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=config["num_workers"],
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
        devices=-1,        # or just devices=-1
        num_nodes=1,
        strategy="ddp",
        limit_train_batches=5, 
        limit_val_batches=5,
        log_every_n_steps=100,    
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        callbacks=[
            ModelCheckpoint(
                dirpath=config["save_dir"], 
                save_last=True, 
                every_n_epochs=1, 
                save_top_k=3,
                monitor="val_loss"),
            EarlyStopping(monitor="val_loss", mode="min"),
        ],
        logger=[
            CSVLogger(save_dir=config["save_dir"]), 
            WandbLogger(
                name=str("project_name"),
                save_dir=config["save_dir"],
                project=config["project_name"],
                offline=os.environ.get("WANDB_MODE", "online") == "offline")
        ],
    )
    try:
        trainer.fit(lit_model, dataloader_train, dataloader_val)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        run = wandb.run
        if run is not None and os.environ.get("WANDB_MODE", "online") == "online":
            wandb.Api().run(f"{run.entity}/{run.project}/{run.id}").delete()
        raise
    except OSError as e:
        if e.errno == 28:
            import subprocess
            save_dir = config["save_dir"]
            print(f"=== ENOSPC during training. Capturing filesystem state for {save_dir} ===", flush=True)
            for cmd in [
                ["ls", "-la", str(save_dir)],
                ["lfs", "quota", "-h", "-u", os.environ["USER"], str(save_dir)],
                ["lfs", "quota", "-h", "-g", "imi", str(save_dir)],
                ["df", "-h", "/tmp", str(save_dir)],
                # ENOSPC fix: stripe layout — if stripe_count is 1, the file was bound
                # to a single OST and ENOSPC is likely a per-OST project quota hit.
                ["lfs", "getstripe", "-d", str(save_dir)],
                ["id"],
            ]:
                print(f"$ {' '.join(cmd)}", flush=True)
                subprocess.run(cmd, check=False)
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main(git_hash=os.environ.get("GIT_HASH", "unknown"))