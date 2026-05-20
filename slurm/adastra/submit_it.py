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


def _to_module_path(s: str) -> str:
    if s.endswith(".py") or "/" in s:
        p = Path(s).with_suffix("")
        parts = [x for x in p.parts if x not in ("", ".", "..")]
        return ".".join(parts)
    return s


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("script", help="Module path, e.g. slurm.adastra.test")
    parser.add_argument("--name", default="transformer")
    parser.add_argument("--account", default="iso1996")
    parser.add_argument("--job-name", default="transformer")
    parser.add_argument("--constraint", default="MI250")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--exclusive", action="store_true", default=True)
    parser.add_argument(
        "--timeout-min",
        type=int,
        default=60,
        help="time in minutes, e.g. 60 for 1:00:00",
    )
    parser.add_argument("--ntasks-per-node", type=int, default=8)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--gpus-per-task", type=int, default=1)
    parser.add_argument("--gpu-bind", default="closest")
    parser.add_argument("--log-subdir", default="submitit")
    args = parser.parse_args()

    root = "SCRATCH"
    project_root = Path(__file__).resolve().parent.parent
    log_root = Path(os.environ[root]) / "proj" / "transformer" / args.log_subdir
    # For demonstration, use fixed values for git_hash and stamp; update if you want real usage.
    git_hash = "a"
    stamp = "b"
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

    slurm_extra = {}
    if args.constraint:
        slurm_extra["constraint"] = args.constraint
    if args.exclusive:
        slurm_extra["exclusive"] = ""

    # Remove deprecated/invalid arguments, use slurm_ntasks_per_node, slurm_cpus_per_task, slurm_gpus_per_task, and slurm_job_name
    # Do not pass slurm_threads_per_core or threads_per_core
    executor.update_parameters(
        name=args.job_name,
        slurm_account=args.account,
        nodes=args.nodes,
        timeout_min=args.timeout_min,
        slurm_ntasks_per_node=args.ntasks_per_node,
        slurm_cpus_per_task=args.cpus_per_task,
        slurm_gpus_per_task=args.gpus_per_task,
        slurm_additional_parameters=slurm_extra,
        slurm_setup=[
            "module purge",
            # Add modules relevant to Adastra as needed.
            "source $WORK/proj/transformer/venv/bin/activate",
            "export WANDB_MODE=offline",
            "unset WANDB_API_KEY",
            "export HF_DATASETS_OFFLINE=1",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH",
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

    job = executor.submit(_entry, module_path=_to_module_path(args.script))

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
