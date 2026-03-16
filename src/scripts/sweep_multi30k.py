"""
A simple hyperparameter sweep script with Optuna.
"""

import optuna
import torch



if __name__ == "__main__":
    
    experiment_name = "multi30k"
    experiment_root = paths.EXPERIMENTS_DIR / experiment_name

    #fixed params
    seq_len = 128
    n_epochs = 200

    eval_freq = 10

    #optim params
    d_model = 128
    d_ff = d_model * 4
    vocab_size = 500
    n_heads = 4
    N = 3

    lr = 3e-4
    batch_size = 256


    # Load from cached local files if they exist, else download and save to data/multi30k
    hf_dataset = load_dataset("bentrevett/multi30k", cache_dir=paths.DATA_DIR)

    hf_ds_train = hf_dataset["train"]
    hf_ds_val = hf_dataset["validation"]

    tokenizer_src = BytePairEncoding.from_file(experiment_root / "tokenizer_en.json")
    tokenizer_tgt = BytePairEncoding.from_file(experiment_root / "tokenizer_de.json")

    dataset_train = TranslationDataset(
        hf_ds_train,#.select(range(batch_size)), 
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        max_len=128,
        )

    dataset_val = TranslationDataset(
        hf_ds_val, 
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        max_len=128)

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
        collate_fn=collate_fn, 
        drop_last=True)


    model = Transformer(
        vocab_size=500,
        seq_len=128,
        d_model=128,
        d_ff=256,
        n_heads=4,
        N=3,
        pad_token_id=0,
        eos_token_id=2,
        p_dropout=0.1
    )
    lit_model = LitTransformer(model)

    trainer = L.Trainer(
    max_epochs=200,
    check_val_every_n_epoch=10,
    callbacks=[
        ModelCheckpoint(dirpath=experiment_root, save_last=True, ),
        EarlyStopping(monitor="val_loss", mode="min")],
    logger=[CSVLogger(save_dir=experiment_root)],
    )

    trainer.fit(lit_model, dataloader_train, dataloader_val)