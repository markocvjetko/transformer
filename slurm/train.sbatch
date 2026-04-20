#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH --account=imi@v100                                                                                                                                                              
#SBATCH --partition=gpu_p13                                                                                                                                                             
#SBATCH --gres=gpu:1                                                                                                                                                                    
##SBATCH --qos=qos_gpu-dev                                                                                                                                                               
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:59:00          
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ── Paths ────────────────────────────────────────────────────────────
# Point data / experiments to scratch so we don't fill the home quota.
# Adjust these to match your cluster's filesystem layout.
export DATA_DIR=/${SCRATCH}/transformer/data
export EXPERIMENTS_DIR=/${SCRATCH}/transformer/experiments

# HuggingFace also caches to ~/.cache by default — redirect it too.
export HF_HOME=/${SCRATCH}/.cache/huggingface

# ── Environment ──────────────────────────────────────────────────────
module purge
module load python/3.11.5
module load cuda/12.8.0
conda activate transformer


# cd to the project root (where pyproject.toml lives)
cd /${WORK}/proj/transformer

# ── Create directories if needed ─────────────────────────────────────
mkdir -p "$DATA_DIR"
mkdir -p "$EXPERIMENTS_DIR"
mkdir -p logs

# ── Run ──────────────────────────────────────────────────────────────
python -m scripts.sweep_multi30k.py
