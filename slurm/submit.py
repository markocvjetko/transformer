import datetime
import shutil
import subprocess
from pathlib import Path

import submitit

from src.scripts.sweep_multi30k import main

git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = Path("experiments/submitit") / f"{stamp}-{git_hash[:8]}"
snapshot = run_dir / "code"
snapshot.mkdir(parents=True, exist_ok=True)
rsync_excludes = [
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=*.egg-info",
    "--exclude=.pytest_cache",
    "--exclude=.mypy_cache",
    "--exclude=.ruff_cache",
]
subprocess.run(
    ["rsync", "-a", *rsync_excludes, "src/", f"{snapshot}/src/"],
    check=True,
)
shutil.copy("pyproject.toml", snapshot / "pyproject.toml")

executor = submitit.AutoExecutor(folder=str(run_dir / "logs" / "%j"))
executor.update_parameters(
    name="transformer",
    slurm_account="imi@v100",
    slurm_partition="gpu_p2s",
    slurm_gres="gpu:1",
    cpus_per_task=20,
    timeout_min=18 * 60 + 59,
    slurm_additional_parameters={"hint": "nomultithread"},
    slurm_setup=[
        "module purge",
        "module load python/3.11.5",
        "module load cuda/12.8.0",
        "conda activate transformer",
        "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH",
        "export DATA_DIR=${SCRATCH}/transformer/data",
        "export EXPERIMENTS_DIR=${SCRATCH}/transformer/experiments",
        "export HF_HOME=${SCRATCH}/.cache/huggingface",
        f"export PYTHONPATH={snapshot.resolve()}:$PYTHONPATH",
        "mkdir -p $DATA_DIR $EXPERIMENTS_DIR",
    ],
)

job = executor.submit(main, git_hash=git_hash)
print(f"Submitted {job.job_id}  →  {run_dir}")