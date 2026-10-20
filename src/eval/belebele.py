import os
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets import load_from_disk
from src.lightning.lit_gpt import LitGPT
from src.models.lm import DecoderLM
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths
'''
add param for logp normalization

'''

@dataclass
class EvalConfig:
    train_config: str = str(
        paths.EXPERIMENTS_DIR
        #/ "architecture-tests/all-changes-lr-5e-4-20261005-amicable-tangerine/config.resolved.yaml"
        #/ "best-gpt2-test-large/fineweb-only-20261012-ternary-reef/config.resolved.yaml"
        / "best-gpt2-test-large/1024-dim-20M-token-20261012-delicious-electron/config.resolved.yaml"
    )
    model_path: str = str(
        paths.EXPERIMENTS_DIR
        #/ "architecture-tests/all-changes-lr-5e-4-20261005-amicable-tangerine/last.ckpt"
        #/ "best-gpt2-test-large/fineweb-only-20261012-ternary-reef/lightning_logs/version_0/checkpoints/last.ckpt"
        / "best-gpt2-test-large/1024-dim-20M-token-20261012-delicious-electron/last.ckpt"
    )
    tokenizer_path: str = str(paths.EXPERIMENTS_DIR / "gpt2-test-large/BPE_vocab_48k_hr_en")
    dataset_path: str = str(paths.DATA_DIR / "raw/belebele-hr")
    device: str = "cuda"


def load_config(argv: list[str] | None = None):
    OmegaConf.register_resolver("paths", lambda name: str(getattr(paths, name)))
    # No CLI parsing for now; just instantiate EvalConfig with defaults
    cfg_eval = EvalConfig()
    cfg_train = OmegaConf.load(cfg_eval.train_config)
    return cfg_eval, cfg_train


def assemble_belebele_hr_sample(sample, tokenizer, mode: str = "letter"):
    # mode letter or full, standardized to match the English version's logic and structure
    question = (
        f"{sample['flores_passage']}\n"
        f"Pitanje: {sample['question']}\n"
        # f"A: {sample['mc_answer1']}\n"
        # f"B: {sample['mc_answer2']}\n"
        # f"C: {sample['mc_answer3']}\n"
        # f"D: {sample['mc_answer4']}\n"
        "Odgovor:"
    )
    question = [tokenizer._special_vocab["<HR>"]] + tokenizer.tokenize(question, add_special=False)
    if mode == "letter":
        answers = [" A", " B", " C", " D"]
        answers = [tokenizer.tokenize(answer, add_special=False)[0] for answer in answers]
    elif mode == "full":
        answers = [
            tokenizer.tokenize(" " + sample["mc_answer1"], add_special=False),
            tokenizer.tokenize(" " + sample["mc_answer2"], add_special=False),
            tokenizer.tokenize(" " + sample["mc_answer3"], add_special=False),
            tokenizer.tokenize(" " + sample["mc_answer4"], add_special=False),
        ]
        answers = [question + a for a in answers]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    correct = int(sample["correct_answer_num"]) - 1
    return question, answers, correct


def assemble_belebele_en_sample(sample, tokenizer, mode: str = "letter"):
    # mode letter or full
    question = (
        f"{sample['flores_passage']}\n"
        f"Question: {sample['question']}\n"
        # f"A: {sample['mc_answer1']}\n"
        # f"B: {sample['mc_answer2']}\n"
        # f"C: {sample['mc_answer3']}\n"
        # f"D: {sample['mc_answer4']}\n"
        "Answer:"
    )
    question = [tokenizer._special_vocab["<EN>"]] + tokenizer.tokenize(question, add_special=False)
    if mode == "letter":
        answers = [" A", " B", " C", " D"]
        answers = [tokenizer.tokenize(answer, add_special=False)[0] for answer in answers]
    elif mode == "full":
        answers = [
            tokenizer.tokenize(" " + sample["mc_answer1"], add_special=False),
            tokenizer.tokenize(" " + sample["mc_answer2"], add_special=False),
            tokenizer.tokenize(" " + sample["mc_answer3"], add_special=False),
            tokenizer.tokenize(" " + sample["mc_answer4"], add_special=False),
        ]

        answers = [question + a for a in answers]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    correct = int(sample["correct_answer_num"]) - 1
    return question, answers, correct


def eval_multiple_answer(
    dataset,
    model: torch.nn.Module,
    tokenizer: BytePairEncoding,
    mode="full",
    assembler=assemble_belebele_hr_sample,
):
    # mode letter or full
    device = next(model.parameters()).device
    total_correct = 0

    # for idx in range(len(dataset)):
    with torch.no_grad():
        for idx in tqdm(range(900), desc="Evaluating"):
            question, answers, correct = assembler(dataset[idx], tokenizer, mode)
            question = torch.tensor([question], dtype=torch.long, device=device)
            answers = [torch.tensor(answer, dtype=torch.long, device=device) for answer in answers]

            if mode == "letter":
                logits = model.forward(question)
                logits = logits[0, -1, :]
                logits = logits[answers]
                pred = logits.argmax().item()
                if pred == correct:
                    total_correct += 1

            if mode == "full":
                Q = question.shape[1]
                scores = []
                for answer in answers:
                    logp = model(answer.unsqueeze(0))[0, Q - 1 : -1].log_softmax(-1)
                    target = answer[Q:]
                    scores.append(logp.gather(1, target[:, None]).sum())
                pred = torch.tensor(scores).argmax().item()

                if pred == correct:
                    total_correct += 1

        return total_correct / len(dataset)


def main():
    # Minimal placeholder for main; main2 contains the old procedure (to be deprecated)
    cfg_eval, cfg_train = load_config()
    tokenizer = BytePairEncoding.from_file(cfg_eval.tokenizer_path)
    transformer = DecoderLM(**cfg_train.model)
    lit = LitGPT.load_from_checkpoint(
        transformer=transformer, checkpoint_path=cfg_eval.model_path, map_location="cuda", weights_only=False
    )
    model = lit.transformer.to(cfg_eval.device).eval()
    dataset = load_from_disk(cfg_eval.dataset_path)
    # The main evaluation/procedure is to be implemented or migrated from main2.
    print("Loaded model, tokenizer, and dataset. Evaluation logic goes here.")
    score = eval_multiple_answer(dataset["test"], model, tokenizer, mode="full")
    print(score)


if __name__ == "__main__":
    main()
