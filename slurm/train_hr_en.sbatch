#!/bin/bash
#SBATCH --job-name=train_hr_en
#SBATCH --account=imi@v100
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:59:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ── Paths ────────────────────────────────────────────────────────────
# Everything on SCRATCH: WORK kept ENOSPC'ing on checkpoint writes (likely
# per-OST project-quota interaction); SCRATCH is the IDRIS-recommended location
# for active training output anyway.
export DATA_DIR=${SCRATCH}/proj/transformer/data
export EXPERIMENTS_DIR=${SCRATCH}/proj/transformer/experiments
export HF_HOME=${SCRATCH}/.cache/huggingface

# ── Offline mode (JZ compute nodes have no internet) ─────────────────
export WANDB_MODE=offline
unset WANDB_API_KEY
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ENOSPC fix: JZ compute /tmp is a 150 MB tmpfs. Anything in the Python stack
# (torch.save, fsspec autocommit=False, wandb, HF datasets, etc.) that uses
# tempfile.mkstemp() lands there and can ENOSPC mid-checkpoint even though
# the destination filesystem has space. Redirect to JOBSCRATCH (per-job
# Lustre scratch) so temp writes have terabytes of room.
export TMPDIR=${JOBSCRATCH:-$SCRATCH/tmp}
mkdir -p "$TMPDIR"

# ── Environment ──────────────────────────────────────────────────────
module purge
module load python/3.11.5
module load cuda/12.8.0
conda activate transformer
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Project root (where pyproject.toml lives — needed for the `src.` package)
cd ${WORK}/proj/transformer

# ── Create directories if needed ─────────────────────────────────────
mkdir -p "$DATA_DIR" "$EXPERIMENTS_DIR" logs

# ENOSPC fix: stripe new files in EXPERIMENTS_DIR across 4 OSTs.
# Default JZ stripe_count=1 binds files to a single OST, which can ENOSPC
# even when the project as a whole has space. No-op if dir is non-empty.
lfs setstripe -c 4 "$EXPERIMENTS_DIR" 2>/dev/null || true

# ── Run ──────────────────────────────────────────────────────────────
python -m src.scripts.train_hr_en
