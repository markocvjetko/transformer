from datasets import load_dataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch import optim, nn
from torch.utils.data import DataLoader

from src.models.transformer import Transformer
from src.datasets.translation import TranslationDataset, collate_fn
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths

class LitTransformer(L.LightningModule):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        input_seq, target_seq = batch

        decoder_input = target_seq[:, :-1]   # All but last token
        target_labels = target_seq[:, 1:]    # All but first token (shifted)
        
        outputs = self.transformer(input_seq, decoder_input)  # (batch, seq_len-1, vocab_size)

            #loss = criterion(outputs.view(-1, vocab_size), target_labels.reshape(-1))
            
        loss = nn.functional.cross_entropy(outputs.view(-1, vocab_size), target_labels.reshape(-1))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        input_seq, target_seq = batch

        decoder_input = target_seq[:, :-1]   # All but last token
        target_labels = target_seq[:, 1:]    # All but first token (shifted)
        
        outputs = self.transformer(input_seq, decoder_input)  # (batch, seq_len-1, vocab_size)
        print(outputs.shape)
        loss = nn.functional.cross_entropy(outputs.view(-1, vocab_size), target_labels.reshape(-1))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     # this is the validation loop
    #     input_seq, target_seq = batch
    #     generated = self.model.translate(input_seq)
    #     self._val_preds

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

        


if __name__ == "__main__":

    experiment_name = "multi30k"
    experiment_root = paths.EXPERIMENTS_DIR / experiment_name

    vocab_size = 500 
    seq_len = 128
    d_model = 128
    d_ff = 256
    n_heads = 4
    N = 3
    lr = 3e-4

    n_epochs = 200
    eval_freq = 10
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