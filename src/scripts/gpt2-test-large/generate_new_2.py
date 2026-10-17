import argparse
from pathlib import Path

import torch

from src.lightning.lit_gpt import LitGPT
from src.models.lm import DecoderLM
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


# TODO add a config support as seen in the train script, the dataset tokenize script....

def main():
    experiment_dir = paths.EXPERIMENTS_DIR / "best-gpt2-test-large/1024-dim-resumed-20261013-thick-diamond"
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--prompt",
        default="",
    )
    parser.add_argument("--ckpt", type=Path, default=experiment_dir / "last.ckpt")
    parser.add_argument(
        "--tokenizer", type=Path, default=paths.EXPERIMENTS_DIR / "gpt2-test-large" / "BPE_vocab_48k_hr_en"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = BytePairEncoding.from_file(args.tokenizer)
    # Get <HR> and <EN> token ids from tokenizer special vocab. Fallback to 3 if not found.
    hr_token_id = tokenizer._special_vocab.get("<HR>", 3)
    en_token_id = tokenizer._special_vocab.get("<EN>", 3)
    prompt_ids = tokenizer.tokenize(args.prompt, add_special=False)
    # Only wrap prompt_ids with language tokens if both <HR> and <EN> are found, and prompt_ids is nonempty
    if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
        raise SystemExit("tokenizer produced an empty prompt")
    # prompt_ids = [hr_token_id] + prompt_ids + [en_token_id]
    prompt_ids = [en_token_id] + prompt_ids + [hr_token_id]


    gpt2 = DecoderLM(
        vocab_size=48006,
        seq_len=512,
        d_model=1024,
        d_ff=2816,
        n_heads=16,
        N=30,
        pad_token_id=0,
        eos_token_id=2,
        p_dropout=0.1,
        pos_emb_type="rope",
        rope_base=10000,
        norm_type="rms",
        ff_type="swiglu",
        use_bias=False,
    )
    lit = LitGPT.load_from_checkpoint(
        str(args.ckpt),
        transformer=gpt2,
        map_location=args.device,
        weights_only=False,
    )
    model = lit.transformer.to(args.device).eval()

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    with torch.no_grad():
        tokens = model.generate(
            tokens,
            temperature=0.3,
            seq_len=128,
            top_k=100,
            top_p=0.9,
        )

    full = tokenizer.decode(tokens[0].tolist(), add_special=True)
    # Find the index of the first occurrence of <EOS>
    eos_index = full.find("<EOS>")

    print("\n")
    print("\n")
    print("\n")
    if eos_index != -1:
        full = full[:eos_index]
    print(full)
    # print("--- continuation ---")
    # print(continuation)


if __name__ == "__main__":
    main()
