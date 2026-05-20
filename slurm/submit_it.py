from argparse import ArgumentParser
import datetime
import importlib
import os
import subprocess
from pathlib import Path

import submitit


def _entry(module_path: str):
    module = importlib.import_module(module_path)
    return module.main()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("script", help="Module path, e.g. src.scripts.train_hr_en")
    parser.add_argument("--name", default="transformer")
    parser.add_argument("--account", default="imi@v100")
    parser.add_argument("--partition", default="gpu_p13")
    parser.add_argument("--qos", default="qos_gpu-t3",
                        help="qos_gpu-t3 (≤20h), qos_gpu-t4 (≤100h), qos_gpu-dev (≤2h)")
    parser.add_argument("--constraint", default="v100-32g",
                        help="e.g. v100-32g, v100-16g")
    parser.add_argument("--cpus-per-task", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--timeout-min", type=int, default=19 * 60 + 59)
    parser.add_argument("--log-subdir", default="submitit")
    args = parser.parse_args()

    root = "SCRATCH"
    project_root = Path(__file__).resolve().parent.parent
    log_root = Path(os.environ[root]) / "proj" / "transformer" / args.log_subdir
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
        input=files, text=True, check=True,
    )

    executor = submitit.AutoExecutor(folder=str(run_dir / "logs" / "%j"))
    slurm_extra = {"hint": "nomultithread"}
    if args.constraint:
        slurm_extra["constraint"] = args.constraint
    executor.update_parameters(
        name=args.name,
        slurm_account=args.account,
        slurm_partition=args.partition,
        slurm_qos=args.qos,
        slurm_gres=f"gpu:{args.gpus}",
        tasks_per_node=args.gpus,                  
        cpus_per_task=args.cpus_per_task,      
        nodes=1,                                   
        timeout_min=args.timeout_min,
        slurm_additional_parameters=slurm_extra,
        slurm_setup=[
            "module purge",
            "module load python/3.11.5",
            "module load cuda/12.8.0",
            "conda activate transformer",
            "export WANDB_MODE=offline",
            "unset WANDB_API_KEY",
            "export HF_DATASETS_OFFLINE=1",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH",
            "export DATA_DIR=${" + root + "}/proj/transformer/data",
            "export EXPERIMENTS_DIR=${" + root + "}/proj/transformer/experiments",
            "export HF_HOME=${SCRATCH}/.cache/huggingface",
            "export TMPDIR=${JOBSCRATCH:-$SCRATCH/tmp}",
            "export ON_JZ=TRUE",
            f"export GIT_HASH={git_hash}",
            f"export PYTHONPATH={snapshot}:$PYTHONPATH",
            "mkdir -p $DATA_DIR $EXPERIMENTS_DIR",
        ],
    )

    job = executor.submit(_entry, module_path=args.script)

    # Stable "latest" symlink: $log_root/latest -> run_dir
    latest = log_root / "latest"
    tmp = log_root / f".latest.{os.getpid()}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(run_dir, target_is_directory=True)
    os.replace(tmp, latest)

    stdout = Path(str(job.paths.stdout))
    stderr = Path(str(job.paths.stderr))
    print(f"Submitted {job.job_id}  →  {run_dir}")
    print(f"  stdout: {stdout}")
    print(f"  stderr: {stderr}")
    print(f"  latest: {latest}  (-> {run_dir.name})")
    print(f"  tail:   tail -F {stdout} {stderr}")
