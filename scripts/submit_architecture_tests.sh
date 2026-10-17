#!/usr/bin/env bash
# Usage: ./scripts/submit_architecture_tests.sh BATCH
set -euo pipefail

batch=${1:?Usage: $0 BATCH}
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
config=src/scripts/gpt2-test-large/config_large_next_gen.yaml

submit() {
    local variant=$1
    shift
    python3 slurm/new_submit.py src.scripts.gpt2-test-large.train_lm \
        --batch "$batch" --variant "${variant}-lr-${lr}" \
        nodes=1 gpus_per_node=4 tasks_per_node=4 \
        -- "$config" \
        trainer.devices=4 trainer.resume_from=null \
        "optimizer.kwargs.lr=$lr" \
        'checkpoint.save_dir=${save_dir}' \
        model.pos_emb_type=learned model.norm_type=layer \
        model.ff_type=gelu model.d_ff=2112 model.use_bias=true \
        "$@"
}

# Six architectures at each learning rate: 12 runs total.
for lr in 5e-3 5e-4; do
    # Each single-change run starts from the same baseline.
    submit gpt2-baseline
    submit rope model.pos_emb_type=rope
    submit rmsnorm model.norm_type=rms
    submit swiglu model.ff_type=swiglu model.d_ff=1408
    # use_bias controls attention/FF linear layers; LayerNorm retains its bias.
    submit no-linear-bias model.use_bias=false
    submit all-changes model.pos_emb_type=rope model.norm_type=rms \
        model.ff_type=swiglu model.d_ff=1408 model.use_bias=false
done
