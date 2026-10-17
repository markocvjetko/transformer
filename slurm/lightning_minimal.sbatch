#!/bin/bash
# Submit from the project root: sbatch slurm/lightning_minimal.sbatch
#SBATCH --job-name=lightning-minimal
#SBATCH --account=bzb@v100
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=lightning-minimal-%j.out
#SBATCH --error=lightning-minimal-%j.err

set -e
module purge
module load pytorch-gpu/py3/2.8.0

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_MODE=offline
export RUN_DIR="${SCRATCH:?}/lightning-minimal/${SLURM_JOB_ID}"
export TMPDIR="$RUN_DIR/tmp"
mkdir -p "$TMPDIR"

# One process per GPU; Lightning reads ranks and rendezvous details from Slurm.
srun --kill-on-bad-exit=1 python -u train_lightning_minimal.py \
    --nodes 4 --devices 4 --steps 200
