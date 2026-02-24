
import os
import torch

from torch.utils.data import DataLoader
from src.datasets.translation import TranslationDataset, collate_fn
from src.models.transformer import Transformer

from datasets import load_dataset

from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths

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
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.to(device)
            target_labels = target_seq[:, 1:]    # All but first token (shifted)
  
            # Single forward pass - model predicts all positions in parallel
            outputs = model.translate(input_seq, decoder_input)  # (batch, seq_len-1, vocab_size)
    
            decoded_targets = [dataset.tokenizer_tgt.decode(target) for target in target_labels]
            decoded_outputs = [dataset.tokenizer_tgt.decode(output) for output in outputs]
            
            targets.extend([*decoded_targets])
            candidates.extend([*decoded_outputs])
        print(bleu.corpus_score(decoded_outputs, [decoded_targets]))
    return


if __name__ == "__main__":
    vocab_size = 500 
    seq_len = 128
    d_model = 128
    d_ff = 128
    n_heads = 8
    N = 6

    n_epochs = 1000
    batch_size = 1

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
        hf_ds_train.select(range(1)), 
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

        # Take a single batch from the validation dataloader
        test_example = next(iter(dataloader_val))

        # Print length of source and target sentences in token space
        src_len = (test_example[0][0] != 0).sum().item()
        tgt_len = (test_example[1][0] != 0).sum().item()
        # print(f"Source sentence token length: {src_len}")
        # print(f"Target sentence token length: {tgt_len}")

        #print(tokenizer_src.decode(test_example[0][0], add_special=True))
        #print(tokenizer_tgt.decode(test_example[1][0], add_special=True))
        # Run the transformer on this batch (src->tgt), simulating inference
        # For simplicity, here, just pass src as both args
        decoder_input = test_example[1][:, :-1].to(device)
        src_input = test_example[0].to(device)
        src_padding_mask = (src_input == 0)
        tgt_padding_mask = (decoder_input == 0)
        test_output = transformer(src_input, decoder_input)

        # Take argmax to get predicted token IDs (greedy decoding for demonstration)
        predicted_ids = test_output.argmax(dim=-1)
        # For batch_size=1, get the first batch example
        predicted_tokens = predicted_ids[0]


        torch.save(transformer.state_dict(), paths.EXPERIMENTS_DIR / "multi30k/latest.pth")
        

        #print(test_output)
        #print("model output argmax (predicted token ids):", predicted_tokens)
        #print("model output decoded:", tokenizer_tgt.decode(predicted_tokens))

        evaluate(transformer, dataloader_train, device)