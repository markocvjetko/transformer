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
        "optimizer.kwargs.lr=5e-3" \
        'checkpoint.save_dir=${save_dir}' \
        model.pos_emb_type=learned model.norm_type=layer \
        model.ff_type=gelu model.d_ff=2112 model.use_bias=true \
        train_dataset.paths=${paths:DATA_DIR}/gpt2-test-large/merged_fineweb_data.npy \
        train_dataset.weights=1.0

        "$@"
}

    submit all-changes model.pos_emb_type=rope model.norm_type=rms \
        model.ff_type=swiglu model.d_ff=1408 model.use_bias=false \ 
