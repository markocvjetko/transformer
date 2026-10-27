import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.datasets.evals import eval_parsing
from src.lightning.lit_gpt import LitGPT, load_dataset
from src.models.lm import DecoderLM
from src.tokenizers.BPE import BytePairEncoding
from src.utils import paths


@dataclass
class EvalConfig:
    train_config: str = str(paths.EXPERIMENTS_DIR / "1B-model/as-240B-20261025-breezy-sound/config.resolved.yaml")
    model_path: str = str(paths.EXPERIMENTS_DIR / "1B-model/as-240B-20261025-breezy-sound/last.ckpt")
    tokenizer_path: str = str(paths.EXPERIMENTS_DIR / "1B-model/hr_en_bs_tokenizer_48000")

    # train_config: str = str(paths.EXPERIMENTS_DIR / "1B-model/test-2-20261021-proper-velocity/config.resolved.yaml")
    # model_path: str = str(paths.EXPERIMENTS_DIR / "1B-model/test-2-20261021-proper-velocity/last.ckpt")
    # tokenizer_path: str = str(paths.EXPERIMENTS_DIR / "1B-model/hr_en_bs_tokenizer_48000")

    dataset_path: str = str(paths.DATA_DIR / "raw/belebele-hr")
    # dataset_parse_fn: Callable = parse_belebele
    device: str = "cuda"


def load_config(argv: list[str] | None = None):
    OmegaConf.register_resolver("paths", lambda name: str(getattr(paths, name)))
    cfg_eval = EvalConfig()
    cfg_train = OmegaConf.load(cfg_eval.train_config)
    return cfg_eval, cfg_train


def load_model(model_path, model_cfg):

    model = DecoderLM(**model_cfg)
    lit = LitGPT.load_from_checkpoint(
        transformer=model,
        checkpoint_path=model_path,
        map_location="cpu",
        weights_only=False,
        strict=False,
    )
    model = lit.transformer.to().eval()
    del lit
    return model


def main():

    cfg_eval, cfg_train = load_config()
    tokenizer = BytePairEncoding.from_file(cfg_eval.tokenizer_path)
    model = load_model(cfg_eval.model_path, cfg_train.model)
    model.eval()
    model.to(device=cfg_eval.device)

    presets = {
        # "eu-mmlu-hr": ("EC-DGT-AI/EU-MMLU", "data/*/HR_HR/*", "raw/eu-mmlu-hr"),
        # "eu-mmlu-en": ("EC-DGT-AI/EU-MMLU", "data/*/EN_GB/*", "raw/eu-mmlu-en"),
        # "belebele-en": ("facebook/belebele", "data/eng_Latn.jsonl", "raw/belebele-en"),
        # "belebele-hr": ("facebook/belebele", "data/hrv_Latn.jsonl", "raw/belebele-hr"),
        # "flores-bs-en": ("facebook/flores", "data/pair/bos_Latn-eng_Latn.jsonl", "raw/flores-bs-en"),
        # "flores-en-hr": ("facebook/flores", "data/pair/hrv_Latn-eng_Latn.jsonl", "raw/flores-en-hr"),
        # "copa-hr": ("classla/copa_hr", "data.zip", "raw/copa-hr"),
        "copa-en": ("pkavumba/balanced-copa", "*.csv", "raw/copa-en"),
    }

    parsers = {
        "eu-mmlu-hr": eval_parsing.parse_eu_mmlu_hr,
        "eu-mmlu-en": eval_parsing.parse_eu_mmlu_en,
        "belebele-en": eval_parsing.parse_belebele_en,
        "belebele-hr": eval_parsing.parse_belebele_hr,
        "flores-bs-en": eval_parsing.parse_flores_bs_en,
        "flores-en-hr": eval_parsing.parse_flores_en_hr,
        "copa-hr": eval_parsing.parse_copa_hr,
        "copa-en": eval_parsing.parse_copa_en,
    }
    for preset in presets:
        dataset = load_dataset(str(paths.DATA_DIR / presets[preset][2]))
        parse_fn = parsers.get(preset)
        total_correct = 0
        split = "test" if "test" in dataset else "validation" if "validation" in dataset else "train"
        print(f"Evaluating {preset} on {split} split with {len(dataset[split])} samples")
        for idx in tqdm(range(len(dataset[split]))):
            question, answers, correct = parse_fn(dataset[split][idx], tokenizer)
            # print(tokenizer.decode(question))
            # print([tokenizer.decode(a) for a in answers])
            # print(tokenizer.decode(correct))

            question = torch.tensor(question, device=cfg_eval.device)
            answers = [torch.tensor(a, device=cfg_eval.device) for a in answers]
            answers = pad_sequence(answers, batch_first=True, padding_value=tokenizer._special_vocab["<PAD>"])
            answer_lens = torch.where(answers[:] != tokenizer._special_vocab["<PAD>"], 1.0, 0.0).sum(dim=1)
            # print(answer_lens)
            querry = torch.cat((question.unsqueeze(0).repeat(answers.size(0), 1), answers), dim=1)

            with torch.no_grad():
                logits = model.forward(querry)

            log_probs = logits.log_softmax(dim=-1)
            selected_log_probs = (
                log_probs[:, question.shape[0] - 1 : -1, :]
                .gather(dim=-1, index=querry[:, question.shape[0] :, None].long())
                .squeeze(dim=-1)
            )
            prob_sum = (
                torch.where(answers != tokenizer._special_vocab["<PAD>"], selected_log_probs, 0).sum(dim=1)
                / answer_lens
            )
            # print(prob_sum)
            # print(selected_log_probs)
            if prob_sum.argmax().item() == int(correct):
                total_correct += 1

        print(total_correct / (idx + 1))
        # print(tokens_new)
        # print(len(tokens_new[0]))
        # full = tokenizer.decode(tokens_new[0].tolist(), add_special=True)
        # print(full)


if __name__ == "__main__":
    main()
