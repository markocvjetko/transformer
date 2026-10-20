import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile

from src.models.lm import DecoderLM

cfg = OmegaConf.load("scripts/training-1B/config_test.yaml").model
model = DecoderLM(**OmegaConf.to_container(cfg, resolve=True)).cuda()  # **kwargs absorbs compile_model
model = torch.compile(model)

x = torch.randint(0, cfg.vocab_size, (32, 1024), device="cuda")  # batch_size, data seq_len


def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x).float().mean().backward()


step(); step()  # compile + warmup
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    step()
    torch.cuda.synchronize()

ops = {e.key for e in prof.key_averages() if "scaled_dot_product" in e.key or "attention" in e.key.lower()}
print("\n".join(sorted(ops)) or "no SDPA op -> math fallback (bmm + softmax)")