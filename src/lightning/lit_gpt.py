from dataclasses import dataclass

import lightning as L
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from datasets import load_dataset
from src.datasets.next_token import NextTokenPredictionDataset
from src.models.gpt2 import GPT2
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class Args:
    experiment_name: str = "gpt2-test-large"
    dataset_root: str = "bentrevett/multi30k"
    tokenizer: str = "tokenizer_en.json"
    vocab_size: int = 37000
    seq_len: int = 128
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    N: int = 3
    p_dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 1
    n_epochs: int = 200
    eval_freq: int = 10
    max_epochs: int = 200
    num_workers: int = 16
    early_stopping_patience: int = 10


class LitGPT(L.LightningModule):
    def __init__(self, transformer, optim="adamw", lr=3e-4, weight_decay=0.01,
        bits_per_byte=None, val_dataset_names=None):
        super().__init__()
        self.transformer = transformer
        self.optim = optim
        self.lr = lr
        self.weight_decay = 0.1
        self.save_hyperparameters(ignore=["transformer"])
        self.bits_per_byte = bits_per_byte
        self.val_dataset_names = val_dataset_names
        

    def training_step(self, batch, batch_idx):
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]

        outputs = self.transformer(input_seq)

        loss = nn.functional.cross_entropy(
            input=outputs.view(-1, self.transformer.vocab_size), 
            target=target_seq.reshape(-1),
            ignore_index=0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]

        outputs = self.transformer(input_seq)
   
        loss = nn.functional.cross_entropy(
            input=outputs.view(-1, self.transformer.vocab_size), 
            target=target_seq.reshape(-1),
            ignore_index=0)
        
        bpb = self.bits_per_byte(outputs, target_seq)
        # Logging to TensorBoard (if installed) by defaultexperiment_name
        
        #if there is self.val_dataset_names print per dataset metrics, otherwise aggregated
        if self.val_dataset_names:
            self.log(f"val_loss {self.val_dataset_names[dataloader_idx]}", loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val_bpb {self.val_dataset_names[dataloader_idx]}", bpb, on_step=False, on_epoch=True, sync_dist=True)
        else:
            self.log("val_loss", loss.mean(), on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_bpb", bpb.mean(), on_step=False, on_epoch=True, sync_dist=True)
    
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        if self.optim == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim == "muon":
            raise NotImplementedError("Muon optimizer is not yet supported.")

        else:
            raise ValueError(f"Unsupported optimizer: {self.optim}")
        return optimizer


if __name__ == "__main__":
    args = Args()

    print(args.experiment_name)
    experiment_root = paths.EXPERIMENTS_DIR / args.experiment_name

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", cache_dir=paths.DATA_DIR / "fineweb-edu")

    hf_dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.hr", cache_dir=paths.DATA_DIR / "wikimedia/wikipedia/20231101/hr/"
    )

    hf_ds_train = hf_dataset["train"]

    tokenizer = BytePairEncoding.from_file(paths.EXPERIMENTS_DIR / "en-hr-tokenizers/tokenizer_37000.json")

    dataset_train = NextTokenPredictionDataset(
        hf_ds_train,
        tokenizer=tokenizer,
        max_len=args.seq_len,
        lang="text",
    )

    dataset_val = NextTokenPredictionDataset(hf_ds_train, tokenizer=tokenizer, max_len=args.seq_len, lang="text")

    # Minimal dataloader
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = GPT2(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        N=args.N,
        p_dropout=args.p_dropout,
        pad_token_id=0,
        eos_token_id=2,
    )

    lit_model = LitGPT(model)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=10,
        overfit_batches=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=experiment_root,
                save_last=True,
            ),
            # EarlyStopping(
            #     monitor="val_loss", mode="min", patience=args.early_stopping_patience
            # ),
        ],
        logger=[CSVLogger(save_dir=experiment_root)],
    )

    trainer.fit(lit_model, dataloader_train, dataloader_val)
