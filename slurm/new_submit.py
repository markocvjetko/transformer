"""
This script is intented to be used as an entry point for all slurm jobs on Jean Zay.


"""

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import randomname
import submitit
from omegaconf import OmegaConf

from src.utils import paths

CLUSTER_PROFILES = {
    "jz": {
        "config": {},
        "setup": [
            "module purge",
            "module load pytorch-gpu/py3/2.8.0",
        ],
    },
    "adastra": {
        # 1 node = 4 MI250X = 8 GCDs, each seen as one 64 GB GPU.
        "config": {
            "slurm_account": "iso1996",
            "slurm_constraint": "MI250",
            "slurm_qos": None,
            "nodes": 8,
            "gpus_per_node": 8,
            "tasks_per_node": 8,
            "cpus_per_task": 8,
            "timeout_min": 24 * 60,
        },
        "setup": [
            "module purge",
            "module use /opt/software/gaia-external/IA/venv_AI/modulefiles/Core/",
            "module load AI-6.0.0",
            "export AI_VENV_PATH=$WORK/venv/transformer",
            "module load venv/0.8.0+torch2.11.0+rocm7.2.0",
            'source "$AI_VENV_PATH/bin/activate"',
            "export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1",
            'export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}"',
            'export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"',
        ],
    },
}


def _detect_cluster() -> str:
    return "adastra" if Path("/opt/software/gaia-external").exists() else "jz"


@dataclass
class ClusterConfig:
    name: str = "transformer"  # --job-name        (generic)
    slurm_account: str = "imi@v100"  # --account imi@v100 imi@cpu
    # slurm_partition: str = ""  # --partition (uncomment/set as needed)
    slurm_constraint: str = "v100-32g"
    slurm_qos: str | None = "qos_gpu-t3"  # --qos   qos_gpu-t3 qos_cpu-dev
    nodes: int = 1
    gpus_per_node: int = 4
    tasks_per_node: int = 4
    cpus_per_task: int = 10  # --cpus-per-task   (generic)
    timeout_min: int = 19 * 60 + 59  # --time=18:59:00 → 1139 min (generic)

    # name: str = "transformer"  # --job-name        (generic)
    # slurm_account: str = "imi@v100"  # --account imi@v100 imi@cpu
    # # slurm_partition: str = ""  # --partition (uncomment/set as needed)
    # slurm_qos: str = "qos_gpu-dev"  # --qos   qos_gpu-t3 qos_cpu-dev
    # nodes: int = 1
    # gpus_per_node: int = 1
    # tasks_per_node: int = 1
    # cpus_per_task: int = 10  # --cpus-per-task   (generic)
    # timeout_min: int = 60
    # timeout_min: incat = 19 * 60 + 59  # --time=18:59:00 → 1139 min (generic)


# @dataclass
# class ClusterConfig:
#     name: str = "transformer"  # --job-name        (generic)
#     slurm_account: str = "imi@cpu"  # --account
#     # slurm_partition: str = ""  # --partition (uncomment/set as needed)
#     slurm_qos: str = "qos_cpu-dev"  # --qos
#     # gpus_per_node: int = 1  # --gres            (or: gpus_per_node=1)
#     cpus_per_task: int = 20  # --cpus-per-task   (generic)
#     timeout_min: int = 1 * 60 + 59  # --time=18:59:00 → 1139 min (generic)


def _run_entry(module_path: str, argv: list[str]):
    module = importlib.import_module(module_path)
    print(f"Training module: {module.__file__}", flush=True)
    old_argv = sys.argv
    try:
        sys.argv = [module_path, *argv]
        return module.main()
    finally:
        sys.argv = old_argv


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # split off the pass-through tail before argparse sees it
    if "--" in argv:
        i = argv.index("--")
        launcher_argv, script_argv = argv[:i], argv[i + 1 :]
    else:
        launcher_argv, script_argv = argv, []

    p = argparse.ArgumentParser(prog="new_submit.py")
    p.add_argument("script", help="entry module, e.g. src.scripts.train_hr_en")
    p.add_argument("--batch", type=str, required=True, help="Batch name for experiment directory (required)")
    p.add_argument("--variant", type=str, required=True, help="Variant name for the run (required)")
    p.add_argument(
        "--cluster",
        choices=sorted(CLUSTER_PROFILES),
        default=_detect_cluster(),
        help="Cluster profile (default: auto-detected)",
    )
    p.add_argument(
        "cluster_overrides",
        nargs="*",
        help="ClusterConfig dotlist, e.g. slurm_qos=... gpus_per_node=2",
    )

    ns = p.parse_intermixed_args(launcher_argv)
    return ns, script_argv


def snapshot_code(project_root: Path, dest: Path) -> None:

    dest.mkdir(parents=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--relative",
            "--exclude=__pycache__",
            "--exclude=*.egg-info",
            "--exclude=*.pyc",
            "src",
            "pyproject.toml",
            str(dest) + "/",
        ],
        cwd=project_root,
        check=True,
    )


def main(argv=None):

    args, script_argv = parse_args(argv)

    profile = CLUSTER_PROFILES[args.cluster]

    cfg_cluster = OmegaConf.merge(
        OmegaConf.structured(ClusterConfig),
        profile["config"],
        OmegaConf.from_dotlist(args.cluster_overrides),
    )

    ### RUN_DIR GENERATION
    while True:
        run_name = f"{args.variant}-{datetime.now():%Y%m%d}-{randomname.get_name()}"
        run_dir = paths.EXPERIMENTS_DIR / args.batch / run_name

        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        break
    cfg_cluster.name = run_name

    snapshot_code(paths.ROOT, run_dir / "code")

    log_folder = run_dir / "slurm"
    executor = submitit.AutoExecutor(folder=log_folder, cluster=None)
    params = {k: v for k, v in OmegaConf.to_container(cfg_cluster, resolve=True).items() if v is not None}
    print(params)
    executor.update_parameters(
        **params,
        # slurm_setup=[
        #     "module purge",
        #     "module load python/3.11.5",
        #     "module load cuda/12.8.0",
        #     "conda activate transformer",
        #     f"export PYTHONPATH={run_dir}:$PYTHONPATH",
        #     "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH",
        slurm_setup=[
            *profile["setup"],
            f"export PYTHONPATH={run_dir / 'code'}:$PYTHONPATH",
            'export TMPDIR="${SCRATCH:?}/transformer/tmp/${SLURM_JOB_ID}"',
            "export WANDB_MODE=offline",
            'mkdir -p "$TMPDIR"',
        ],
    )

    if script_argv and "=" not in script_argv[0]:
        config_copy = (run_dir / "config.input.yaml").resolve()
        shutil.copy2(script_argv[0], config_copy)
        script_argv[0] = str(config_copy)

    submitted_argv = [
        *script_argv,
        f"save_dir={run_dir}",
        f"wandb.group={args.batch}",
        f"wandb.name={run_name}",
        f"trainer.num_nodes={cfg_cluster.nodes}",
    ]
    print(args.cluster)
    job = executor.submit(
        _run_entry,
        args.script,
        submitted_argv,
    )
    meta = {
        "job_id": str(job.job_id),
        "run_dir": str(run_dir),
        "batch": str(args.batch),
        "variant": str(args.variant),
        "run_name": run_name,
        "stdout": str(job.paths.stdout),
        "stderr": str(job.paths.stderr),
        "cluster": args.cluster,
        "cluster_config": OmegaConf.to_container(cfg_cluster, resolve=True),
        "submitted_script": str(args.script),
        "script_args": submitted_argv,
    }

    print(meta)
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(json.dumps(meta, indent=4))

    # output = job.result()
    # print(output)


if __name__ == "__main__":
    """
    Should be run with something like: python slurm/submit.it <script_path> <cluster_config> <scriptargs> 
    
    $EXPERIMENTS_DIR/<run_name>/<id>/          # e.g. hren-baseline/0042/
    ├── code/                # code snapshot           — submitter writes (fresh only)
    ├── slurm/<jobid>/       # submitit logs + sbatch  — submitit writes (each job)
    ├── checkpoints/         # last.ckpt etc.          — script writes / appends
    ├── csv/  wandb/         # Lightning + wandb        — script writes
    ├── config.yaml          # resolved script config  — script writes, immutable
    └── meta.json            # provenance ledger        — submitter appends each job
    """
    main()
