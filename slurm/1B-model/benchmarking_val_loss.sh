#!/usr/bin/env bash
# One line per benchmark: "<variant> <overrides...>"
set -euo pipefail

CONFIG=scripts/training-1B/config_1B_mfu_bench.yaml
COMMON=(
    trainer.max_tokens=5000000000
    trainer.log_every_n_steps=10
    trainer.val_check_interval=1.0
    trainer.limit_val_batches=2
    checkpoint.enabled=false
)

BENCHMARKS=(

)

for bench in "${BENCHMARKS[@]}"; do
    read -r variant overrides <<< "$bench"
    python3 slurm/new_submit.py --cluster adastra --batch 1B-model-adastra-mfu-bs12 --variant "$variant" \
    src.scripts.gpt2-test-large.train nodes=1 gpus_per_node=8 tasks_per_node=8 timeout_min=60 -- \
    "$CONFIG" "${COMMON[@]}" $overrides
done
