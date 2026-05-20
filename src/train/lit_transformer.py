from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from datasets import load_dataset
from src.datasets.translation import TranslationDataset, collate_fn
from src.models.transformer import Transformer
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class Args:
    experiment_name: str = "multi30k"
    dataset_root: str = "bentrevett/multi30k"
    tokenizer_src: str = "tokenizer_en.json"
    tokenizer_tgt: str = "tokenizer_de.json"
    vocab_size: int = 500
    seq_len: int = 128
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    N: int = 3
    p_dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 200
    eval_freq: int = 10
    max_epochs: int = 200
    num_workers: int = 16
    early_stopping_patience: int = 10


class LitTransformer(L.LightningModule):
    def __init__(self, transformer, optim="adamw", lr=3e-4, weight_decay=0.01):
        super().__init__()
        self.transformer = transformer
        self.optim = optim
        self.lr = lr
        self.weight_decay = 0.1
        self.save_hyperparameters(ignore=["transformer"])

    def training_step(self, batch, batch_idx):
        input_seq, target_seq = batch

        decoder_input = target_seq[:, :-1]  # All but last token
        target_labels = target_seq[:, 1:]  # All but first token (shifted)

        outputs = self.transformer(
            input_seq, decoder_input
        )  # (batch, seq_len-1, vocab_size)

        loss = nn.functional.cross_entropy(
            outputs.view(-1, self.transformer.vocab_size), target_labels.reshape(-1)
        )
        # Logging to TensorBoard (if installed) by defaultexperiment_name
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        input_seq, target_seq = batch

        decoder_input = target_seq[:, :-1]  # All but last token
        target_labels = target_seq[:, 1:]  # All but first token (shifted)

        outputs = self.transformer(
            input_seq, decoder_input
        )  # (batch, seq_len-1, vocab_size)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, self.transformer.vocab_size), target_labels.reshape(-1)
        )
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        if self.optim == "adamw":
            optimizer = optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optim == "muon":
            raise NotImplementedError("Muon optimizer is not yet supported.")

        else:
            raise ValueError(f"Unsupported optimizer: {self.optim}")
        return optimizer


if __name__ == "__main__":
    args = Args()

    print(args.experiment_name)
    experiment_root = paths.EXPERIMENTS_DIR / args.experiment_name

    # Load from cached local files if they exist, else download and save to data/multi30k
    hf_dataset = load_dataset("bentrevett/multi30k", cache_dir=paths.DATA_DIR)

    hf_ds_train = hf_dataset["train"]
    hf_ds_val = hf_dataset["validation"]

    tokenizer_src = BytePairEncoding.from_file(experiment_root / args.tokenizer_src)
    tokenizer_tgt = BytePairEncoding.from_file(experiment_root / args.tokenizer_tgt)

    dataset_train = TranslationDataset(
        hf_ds_train,  # .select(range(batch_size)),
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        max_len=args.seq_len,
    )

    dataset_val = TranslationDataset(
        hf_ds_val,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        max_len=args.seq_len,
    )

    # Minimal dataloader
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = Transformer(
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

    lit_model = LitTransformer(model)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=10,
        callbacks=[
            ModelCheckpoint(
                dirpath=experiment_root,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_loss", mode="min", patience=args.early_stopping_patience
            ),
        ],
        logger=[CSVLogger(save_dir=experiment_root)],
    )

    trainer.fit(lit_model, dataloader_train, dataloader_val)
