import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from src.lightning.lit_gpt import LitGPT
from src.models.lm import DecoderLM
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class GenConfig:
    # train_config: str = str(paths.EXPERIMENTS_DIR / "1B-model/test-2-20261021-proper-velocity/config.resolved.yaml")
    # ckpt: str = str(paths.EXPERIMENTS_DIR / "1B-model/test-2-20261021-proper-velocity/last.ckpt")
    train_config: str = str(paths.EXPERIMENTS_DIR / "1B-model/test-2-20261021-acute-swimmer/config.resolved.yaml")
    ckpt: str = str(paths.EXPERIMENTS_DIR / "1B-model/test-2-20261021-acute-swimmer/last.ckpt")
    tokenizer: str = str(paths.EXPERIMENTS_DIR / "1B-model/hr_en_bs_tokenizer_48000")
    prompt: str = "Test Prompt"
    max_new_tokens: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(argv: list[str] | None = None):
    from omegaconf import OmegaConf

    # Register the "paths" resolver for path interpolation if needed
    OmegaConf.register_resolver("paths", lambda name: str(getattr(paths, name)))
    cli = OmegaConf.from_cli(argv)
    cfg = OmegaConf.structured(GenConfig)
    cfg = OmegaConf.merge(cfg, cli)  # external file overrides
    # Always load train_config from (potentially overridden) cfg.train_config
    cfg_train = OmegaConf.load(cfg.train_config)
    return OmegaConf.to_object(cfg), cfg_train


def main():
    cfg, cfg_train = load_config()
    tokenizer = BytePairEncoding.from_file(cfg.tokenizer)
    # Get <HR> and <EN> token ids from tokenizer special vocab. Fallback to 3 if not found.
    translate_token_id = tokenizer._special_vocab.get("<TRANSLATE>", 3)
    hr_token_id = tokenizer._special_vocab.get("<HR>", 3)
    en_token_id = tokenizer._special_vocab.get("<EN>", 3)
    prompt_ids = tokenizer.tokenize(cfg.prompt, add_special=False)
    # Only wrap prompt_ids with language tokens if both <HR> and <EN> are found, and prompt_ids is nonempty
    if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
        raise SystemExit("tokenizer produced an empty prompt")
    #prompt_ids = [hr_token_id] + prompt_ids + [en_token_id]
    #prompt_ids = [translate_token_id] + [en_token_id] + prompt_ids + [hr_token_id]
    prompt_ids = [hr_token_id] + prompt_ids
    print(cfg.prompt)
    tokenizer = BytePairEncoding.from_file(cfg.tokenizer)

    model = DecoderLM(**cfg_train.model)
    lit = LitGPT.load_from_checkpoint(
        transformer=model,
        checkpoint_path=cfg.ckpt,
        map_location="cpu",
        weights_only=False,
        strict=False,
    )
    model = lit.transformer.to(cfg.device).eval()
    del lit

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)
    print(tokens.shape)
    with torch.no_grad():
        tokens = model.generate(
            tokens,
            temperature=0.5,
            seq_len=96,
            top_k=32,
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
