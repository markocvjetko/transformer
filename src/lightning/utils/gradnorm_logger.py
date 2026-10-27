import subprocess

import torch

import lightning as L


class GradNormLogger(L.Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        pl_module.log("grad_norm", norm)
