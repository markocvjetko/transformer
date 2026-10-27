import os
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf

from datasets import load_from_disk
from src.lightning.lit_gpt import LitGPT
from src.models.lm import DecoderLM
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class EvalConfig:
    # train_config: str = str(
    #     paths.EXPERIMENTS_DIR
    #     / "architecture-tests/all-changes-lr-5e-4-20261005-amicable-tangerine/config.resolved.yaml"
    # )
    # model_path: str = str(
    #     paths.EXPERIMENTS_DIR / "architecture-tests/all-changes-lr-5e-4-20261005-amicable-tangerine/last.ckpt"
    # )

    # train_config: str = str(
    #     paths.EXPERIMENTS_DIR
    #     / "architecture-tests/all-changes-lr-5e-4-20261005-amicable-tangerine/config.resolved.yaml"
    # )
    # model_path: str = str(
    #     paths.EXPERIMENTS_DIR / "architecture-tests/all-changes-lr-5e-4-20261005-amicable-tangerine/last.ckpt"
    # )
    train_config: str = str(paths.EXPERIMENTS_DIR / "1B-model/as-240B-20261025-breezy-sound/config.resolved.yaml")
    model_path: str = str(paths.EXPERIMENTS_DIR / "1B-model/as-240B-20261025-breezy-sound/last.ckpt")
    tokenizer_path: str = str(paths.EXPERIMENTS_DIR / "1B-model/hr_en_bs_tokenizer_48000")
    # train_config: str = str(paths.EXPERIMENTS_DIR / "1B-model/test-2-20261021-acute-swimmer/config.resolved.yaml")
    # model_path: str = str(paths.EXPERIMENTS_DIR / "1B-model/test-2-20261021-acute-swimmer/last.ckpt")
    # tokenizer_path: str = str(paths.EXPERIMENTS_DIR / "1B-model/hr_en_bs_tokenizer_48000")

    dataset_path: str = str(paths.DATA_DIR / "raw/belebele-hr")
    device: str = "cuda"


def load_config(argv: list[str] | None = None):
    OmegaConf.register_resolver("paths", lambda name: str(getattr(paths, name)))
    # No CLI parsing for now; just instantiate EvalConfig with defaults
    cfg_eval = EvalConfig()
    cfg_train = OmegaConf.load(cfg_eval.train_config)
    return cfg_eval, cfg_train


def main():
    # Load configs
    cfg_eval, cfg_train = load_config()
    tokenizer = BytePairEncoding.from_file(cfg_eval.tokenizer_path)

    # model = DecoderLM(**cfg_train.model)
    # lit = LitGPT.load_from_checkpoint(
    #     transformer=model,
    #     checkpoint_path=cfg_eval.model_path,
    #     map_location="cpu",
    #     weights_only=False,
    #     strict=False,
    # )
    # model = lit.transformer.to(cfg_eval.device).eval()
    # del lit

    dataset = load_from_disk(cfg_eval.dataset_path)

    # Verify the data works offline
    # print("\n--- Offline Dataset Verification ---")
    # print(dataset)
    total_correct = 0
    for idx in range(900):
        # print("\nFirst sample text preview:")
        # print(dataset["test"][idx])

        text = (
            dataset["test"][idx]["flores_passage"]
            + "\n"
            + "Pitanje: "
            + dataset["test"][idx]["question"]
            + "\n"
            + "A: "
            + dataset["test"][idx]["mc_answer1"]
            + "\n"
            + "B: "
            + dataset["test"][idx]["mc_answer2"]
            + "\n"
            + "C: "
            + dataset["test"][idx]["mc_answer3"]
            + "\n"
            + "D: "
            + dataset["test"][idx]["mc_answer4"]
            + "\n"
            + "Odgovor:"
        )
        correct = dataset["test"][idx]["correct_answer_num"]
        print("correct:", correct)
        target_tokens = [
            tokenizer.tokenize(" A", add_special=False)[0],
            tokenizer.tokenize(" B", add_special=False)[0],
            tokenizer.tokenize(" C", add_special=False)[0],
            tokenizer.tokenize(" D", add_special=False)[0],
        ]
        print(target_tokens)

        tokens = [tokenizer._special_vocab["<HR>"]] + tokenizer.tokenize(text)

        # print(len(tokens))
        tokens = torch.tensor([tokens], dtype=torch.long, device=cfg_eval.device)
        with torch.no_grad():
            tokens_new = model.forward(tokens)
            # tokens_new = model.generate(
            #     tokens,
            #     temperature=0.8,
            #     seq_len=170,
            #     top_k=100,
            #     top_p=0.9,
            # )
        log_probs = tokens_new[0, -1, :][target_tokens]
        print(log_probs)
        if log_probs.argmax().item() == (int(correct) - 1):
            total_correct += 1
    print(total_correct / 900)
    # print(tokens_new)
    # print(len(tokens_new[0]))
    # full = tokenizer.decode(tokens_new[0].tolist(), add_special=True)
    # print(full)


if __name__ == "__main__":
    main()
