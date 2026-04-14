#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

# ── Paths ────────────────────────────────────────────────────────────
# Point data / experiments to scratch so we don't fill the home quota.
# Adjust these to match your cluster's filesystem layout.
export TRANSFORMER_DATA_DIR=/scratch/${USER}/data
export TRANSFORMER_EXPERIMENTS_DIR=/scratch/${USER}/experiments

# HuggingFace also caches to ~/.cache by default — redirect it too.
export HF_HOME=/scratch/${USER}/.cache/huggingface

# ── Environment ──────────────────────────────────────────────────────
module load python/3.10
module load cuda/12.1

# cd to the project root (where pyproject.toml lives)
cd /scratch/${USER}/transformer

# Activate the project's virtual environment
source .venv/bin/activate

# ── Create directories if needed ─────────────────────────────────────
mkdir -p "$TRANSFORMER_DATA_DIR"
mkdir -p "$TRANSFORMER_EXPERIMENTS_DIR"
mkdir -p logs

# ── Run ──────────────────────────────────────────────────────────────
python -m src.train.train_transformer
