import os
from sacrebleu.metrics.bleu import BLEU
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from tqdm import tqdm
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
        "tokenizer": {"vocab_size": 37000, "min_frequency": 100},
        "project_name": "en-hr",
        "save_dir": paths.EXPERIMENTS_DIR / "en-hr",
        "tokenizer_path": paths.EXPERIMENTS_DIR
        / "en-hr-tokenizers"
        / "tokenizer_37000.json",
        "opus": paths.DATA_DIR / "Helsinki-NLP__opus-100",
        "hrenwac": paths.DATA_DIR / "hrenwac",
        "jwl_300": paths.DATA_DIR / "sentence-transformers/parallel-sentences-jw300",
        "seq_len": 256,
        "src_lang": "en",
        "tgt_lang": "hr",
        "check_val_every_n_epoch": 1,
        "max_epochs": 200,
        "num_workers": 10,
        "early_stopping_patience": 5,
    }

    vocab_size = 37000
    d_model = 256
    d_ff = 512
    n_heads = 8
    N = 6
    lr = 3e-4
    batch_size = 32

    opus = load_dataset(
        "Helsinki-NLP/opus-100", "en-hr", cache_dir=config["opus"], split="test"
    )
    # jwl = load_dataset("sentence-transformers/parallel-sentences-jw300", "en-hr", split="test", cache_dir=config["jwl_300"])

    ds_test = ConcatDataset([opus["translation"]])  # jwl])
    print(len(ds_test))

    tokenizer = BytePairEncoding.from_file(config["tokenizer_path"])

    ds_test = TranslationDataset(
        ds_test,
        src_lang=config["src_lang"],
        tgt_lang=config["tgt_lang"],
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        max_len=config["seq_len"],
    )

    dataloader_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = Transformer(
        vocab_size=vocab_size,
        seq_len=config["seq_len"],
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        N=N,
        pad_token_id=0,
        eos_token_id=2,
        p_dropout=0.1,
    )

    # lit_model = LitTransformer.load_from_checkpoint(paths.EXPERIMENTS_DIR / "en-hr/last.ckpt", transformer=model)

    lit_model = LitTransformer.load_from_checkpoint(
        paths.EXPERIMENTS_DIR / "en-hr" / "last.ckpt",
        transformer=model,
    )
    model = lit_model.transformer
    model = model.cuda()
    out_strs = []
    tgt_strs = []
    bleu = BLEU()
    for input_seq, target_seq in tqdm(dataloader_test, desc="Evaluating", unit="batch"):
        input_seq = input_seq.cuda()

        output = model.translate_beam_search(
            input_seq,
            y=torch.full((input_seq.shape[0], 1), 1, dtype=torch.long).to("cuda"),
        )
        for o, t in zip(output, target_seq):
            out_strs.append(tokenizer.decode(o[0]))
            tgt_strs.append(tokenizer.decode(t))
            # print(out_strs[-1], tgt_strs[-1])

    print(bleu.corpus_score(out_strs, tgt_strs))


if __name__ == "__main__":
    main()
