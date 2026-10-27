from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

import lightning as L
from datasets import load_dataset
from src.datasets.next_token import NextTokenPredictionDataset
from src.models.gpt2 import GPT2
from src.optim.scheduler import warmup_stable_decay_scheduler
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


def ce_loss(logits, y):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)


class LitGPT(L.LightningModule):
    def __init__(
        self,
        transformer,
        optim_name="adamw",
        optim_kwargs=None,
        lr_scheduler_name=None,
        lr_scheduler_kwargs=None,
        bits_per_byte=None,
        val_dataset_names=None,
        compile_loss=False,
        compile_mode="default",
        fused_ce=False,
    ):
        super().__init__()
        self.transformer = transformer
        self.optim_name = optim_name
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.bits_per_byte = bits_per_byte
        self.val_dataset_names = val_dataset_names

        self.compile_loss = compile_loss
        self.fused_ce = fused_ce

        if fused_ce:
            self._flce = LigerFusedLinearCrossEntropyLoss(ignore_index=0)
        else:
            self._ce = torch.compile(ce_loss, mode=compile_mode) if compile_loss else ce_loss

        self.save_hyperparameters(ignore=["transformer", "bits_per_byte"])

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, 1:]
        if self.fused_ce:
            h = self.transformer(x, return_hidden=True)
            loss = self._flce(self.transformer.ff_output.weight, h.reshape(-1, h.size(-1)), y.reshape(-1))
        else:
            loss = self._ce(self.transformer(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=False)

        return loss

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # one pass: compute the global grad norm, clip, and log the pre-clip norm
        max_norm = gradient_clip_val if gradient_clip_val is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm, foreach=True)
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]

        outputs = self.transformer(input_seq)

        loss = nn.functional.cross_entropy(
            input=outputs.view(-1, self.transformer.vocab_size), target=target_seq.reshape(-1), ignore_index=0
        )

        bpb = self.bits_per_byte(outputs, target_seq)
        # Logging to TensorBoard (if installed) by defaultexperiment_name

        # if there is self.val_dataset_names print per dataset metrics, otherwise aggregated
        if self.val_dataset_names is not None and dataloader_idx is not None:
            self.log(
                f"val_loss {self.val_dataset_names[dataloader_idx]}",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                add_dataloader_idx=False,
            )
            self.log(
                f"val_bpb {self.val_dataset_names[dataloader_idx]}",
                bpb.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        else:
            self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=False, add_dataloader_idx=False)
            self.log("val_bpb", bpb.mean(), on_step=False, on_epoch=True, sync_dist=False, add_dataloader_idx=False)

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def on_load_checkpoint(self, checkpoint):
        prefix = "transformer._orig_mod." if hasattr(self.transformer, "_orig_mod") else "transformer."
        checkpoint["state_dict"] = {
            (
                prefix + key.removeprefix("transformer.").removeprefix("_orig_mod.")
                if key.startswith("transformer.")
                else key
            ): value
            for key, value in checkpoint["state_dict"].items()
        }

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return batch.long()

    def configure_optimizers(self):
        if self.optim_name == "adamw":
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                [
                    {"params": [p for p in params if p.ndim >= 2]},
                    {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
                ],
                **self.optim_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim_name}")

        if self.lr_scheduler_name is None:
            return [optimizer]
        elif self.lr_scheduler_name == "wsd":
            scheduler = warmup_stable_decay_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")


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
            )
        ],
        logger=[CSVLogger(save_dir=experiment_root)],
    )

    trainer.fit(lit_model, dataloader_train, dataloader_val)
