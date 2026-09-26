import torch


def warmup_stable_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    max_iters,
    warmup_ratio=0.1,
    decay_ratio=0.1,
    min_lr_ratio=0.1,
    last_epoch: int = -1,
    # warmup_type="linear",
    # decay_type="cosine",
):

    def lambda_lr(step: int) -> float:
        progress = step / max_iters

        if progress < warmup_ratio:
            return min_lr_ratio + (1 - min_lr_ratio) * (progress / warmup_ratio)
        if progress > 1 - decay_ratio:
            return 1.0 - (1 - min_lr_ratio) * ((progress - (1 - decay_ratio)) / decay_ratio)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    # Use a real optimizer (e.g., SGD) for correct scheduler functionality
    max_iters = 100
    warmup_ratio = 0.1
    decay_ratio = 0.1
    min_lr_ratio = 0.1

    # Create a dummy parameter for the optimizer
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy_param], lr=0.1)

    scheduler = warmup_stable_decay_scheduler(
        optimizer=optimizer,
        max_iters=max_iters,
        warmup_ratio=warmup_ratio,
        decay_ratio=decay_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    # Retrieve the lambda function for plotting
    lambda_lr = scheduler.lr_lambdas[0]
    steps = range(max_iters + 1)
    lrs = [lambda_lr(step) for step in steps]

    plt.plot(steps, lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate (relative)")
    plt.title("Warmup-Stable-Decay Learning Rate Schedule")
    plt.show()
