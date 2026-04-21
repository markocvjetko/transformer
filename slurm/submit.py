import datetime
import importlib
import os
import shutil
import subprocess
from pathlib import Path

import submitit


def _entry(git_hash: str):
    import sys
    print("=== _entry diagnostics ===", flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}", flush=True)
    print("sys.path:", flush=True)
    for p in sys.path:
        print(f"  {p}", flush=True)
    mod = importlib.import_module("src.scripts.sweep_multi30k")
    print(f"loaded module __file__: {mod.__file__}", flush=True)
    print(f"module attrs: {sorted(a for a in dir(mod) if not a.startswith('_'))}", flush=True)
    print("==========================", flush=True)
    return mod.main(git_hash=git_hash)


project_root = Path(__file__).resolve().parent.parent
log_root = Path(os.environ.get("SCRATCH", project_root)) / "transformer" / "submitit"
git_hash = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True, cwd=project_root
).strip()
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = log_root / f"{stamp}-{git_hash[:8]}"
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
    ["rsync", "-a", *rsync_excludes, f"{project_root}/src/", f"{snapshot}/src/"],
    check=True,
)
shutil.copy(project_root / "pyproject.toml", snapshot / "pyproject.toml")

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

job = executor.submit(_entry, git_hash=git_hash)
print(f"Submitted {job.job_id}  →  {run_dir}")