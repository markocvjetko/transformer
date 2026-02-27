
import os
import torch

from torch.utils.data import DataLoader
from src.datasets.translation import TranslationDataset, collate_fn
from src.models.transformer import Transformer

from datasets import load_dataset

from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths

from tqdm import tqdm
from sacrebleu.metrics import BLEU

def evaluate(model: Transformer, dataloader: DataLoader, device: torch.device):
    # Fetch the underlying dataset from the dataloader
    dataset = dataloader.dataset
    bos_token = dataset.bos_token

    batch_size = dataloader.batch_size if dataloader.batch_size is not None else 1

    # No teacher forcing, the model decoder always starts with the <BOS> token 
    decoder_input = torch.full((batch_size, 1), bos_token, dtype=torch.long).to(device)
    
    targets = [] #expected to be a list of strings
    candidates = [] #expected to be a list of strings
    bleu = BLEU()
    
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            input_seq = input_seq.to(device)
            target_labels = target_seq[:, 1:]    # All but first token (shifted)
  
            # Single forward pass - model predicts all positions in parallel
            outputs = model.translate(input_seq, decoder_input)  # (batch, seq_len-1, vocab_size)
    
            decoded_targets = [dataset.tokenizer_tgt.decode(target) for target in target_labels]
            decoded_outputs = [dataset.tokenizer_tgt.decode(output) for output in outputs]
            targets.extend([*decoded_targets])
            candidates.extend([*decoded_outputs])
        print(candidates[0])    #HACK This is just a debug print
        print(bleu.corpus_score(candidates, [targets]))
    return


if __name__ == "__main__":
    vocab_size = 500 
    seq_len = 128
    d_model = 512
    d_ff = 2048
    n_heads = 8
    N = 6

    n_epochs = 1000
    batch_size = 128

    data_dir = paths.DATA_DIR / "multi30k"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer = Transformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        N=N
    ).to(device)

    #print num parameters
    print(f"Number of parameters: {sum(p.numel() for p in transformer.parameters())}")

    # Load from cached local files if they exist, else download and save to data/multi30k
    data_dir = "data/multi30k"
    hf_dataset = load_dataset("bentrevett/multi30k", cache_dir=data_dir)

    hf_ds_train = hf_dataset["train"]
    hf_ds_val = hf_dataset["validation"]

    tokenizer_src = BytePairEncoding.from_file(paths.EXPERIMENTS_DIR / "multi30k/tokenizer_en.json")
    tokenizer_tgt = BytePairEncoding.from_file(paths.EXPERIMENTS_DIR / "multi30k/tokenizer_de.json")

    dataset_train = TranslationDataset(
        hf_ds_train.select(range(2048)), 
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
        collate_fn=collate_fn)

    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn, 
        drop_last=True)


    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=3e-4)

    # Minimal training loop
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader_train):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            
            decoder_input = target_seq[:, :-1]   # All but last token
            target_labels = target_seq[:, 1:]    # All but first token (shifted)
            
            # Create padding masks (True where padding)
            
            # Single forward pass - model predicts all positions in parallel
            outputs = transformer(input_seq, decoder_input)  # (batch, seq_len-1, vocab_size)

            # Compute loss on ALL positions at once
            loss = criterion(outputs.view(-1, vocab_size), target_labels.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # if (batch_idx) % 500 == 0 or (batch_idx+1) == len(dataloader):
            #     print(f"  Batch {batch_idx+1}/{len(dataloader)}: Batch Loss = {loss.item():.4f}")
            #     #print softmaxed model outputs for the last batch
            #     #print("Softmaxed model outputs:", F.softmax(outputs, dim=-1))
        avg_loss = total_loss / len(dataloader_val)
        print(f"Epoch {epoch+1} Average Loss = {avg_loss:.4f}")

        torch.save(transformer.state_dict(), paths.EXPERIMENTS_DIR / "multi30k/latest.pth")
        
        if epoch % 50 == 0:
            evaluate(transformer, dataloader_train, device)