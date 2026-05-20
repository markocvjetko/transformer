"""Submit a training/eval script to SLURM via submitit.

Examples:
    python slurm/submit.py src.scripts.train_hr_en
    python slurm/submit.py src.scripts.sweep_multi30k --partition gpu_p2s --cpus 20
    python slurm/submit.py src.scripts.train_hr_en --name hren --log-subdir submitit_hren

The target module must expose a `main` callable. If `main` accepts a
`git_hash` keyword argument it will be passed; otherwise the git hash is
available to the script via the `GIT_HASH` environment variable.
"""

import argparse
import datetime
import importlib
import inspect
import os
import subprocess
from pathlib import Path

import submitit


def _entry(module_path: str, git_hash: str):
    os.environ["GIT_HASH"] = git_hash
    module = importlib.import_module(module_path)
    main = module.main
    if "git_hash" in inspect.signature(main).parameters:
        return main(git_hash=git_hash)
    return main()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("module", help="Module path, e.g. src.scripts.train_hr_en")
    parser.add_argument("--name", default="transformer", help="SLURM job name")
    parser.add_argument("--account", default="imi@v100")
    parser.add_argument("--partition", default="gpu_p13")
    parser.add_argument("--cpus", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--timeout-min", type=int, default=18 * 60 + 59)
    parser.add_argument(
        "--log-subdir",
        default="submitit",
        help="Subdir under $SCRATCH/proj/transformer for logs/snapshots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    scratch = Path(os.environ["SCRATCH"])
    log_root = scratch / "proj" / "transformer" / args.log_subdir

    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=project_root
    ).strip()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = log_root / f"{stamp}-{git_hash[:8]}"
    snapshot = run_dir / "code"
    snapshot.mkdir(parents=True, exist_ok=True)

    files = subprocess.check_output(
        ["git", "ls-files", "src", "pyproject.toml"], text=True, cwd=project_root
    )
    subprocess.run(
        ["rsync", "-a", "--files-from=-", str(project_root) + "/", str(snapshot) + "/"],
        input=files,
        text=True,
        check=True,
    )

    executor = submitit.AutoExecutor(folder=str(run_dir / "logs" / "%j"))
    executor.update_parameters(
        name=args.name,
        slurm_account=args.account,
        slurm_partition=args.partition,
        slurm_gres=f"gpu:{args.gpus}",
        cpus_per_task=args.cpus,
        timeout_min=args.timeout_min,
        slurm_additional_parameters={"hint": "nomultithread"},
        slurm_setup=[
            "module purge",
            "module load python/3.11.5",
            "module load cuda/12.8.0",
            "conda activate transformer",
            "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH",
            "export DATA_DIR=${SCRATCH}/proj/transformer/data",
            "export EXPERIMENTS_DIR=${SCRATCH}/proj/transformer/experiments",
            "export HF_HOME=${SCRATCH}/.cache/huggingface",
            "export ON_JZ=TRUE",
            f"export GIT_HASH={git_hash}",
            f"export PYTHONPATH={snapshot}:$PYTHONPATH",
            "mkdir -p $DATA_DIR $EXPERIMENTS_DIR",
        ],
    )

    job = executor.submit(_entry, module_path=args.module, git_hash=git_hash)
    print(f"Submitted {job.job_id}  ->  {run_dir}")


if __name__ == "__main__":
    main()
