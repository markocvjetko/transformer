"""
This script is intented to be used as an entry point for all slurm jobs on Jean Zay.


"""

import argparse
import importlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import submitit
from omegaconf import OmegaConf

from src.utils import paths

# @dataclass
# class ClusterConfig:
#     name: str = "transformer"  # --job-name        (generic)
#     slurm_account: str = "imi@v100"  # --account
#     # slurm_partition: str = ""  # --partition (uncomment/set as needed)
#     slurm_qos: str = "qos_gpu-dev"  # --qos
#     gpus_per_node: int = 1  # --gres            (or: gpus_per_node=1)
#     cpus_per_task: int = 20  # --cpus-per-task   (generic)
#     timeout_min: int = 1 * 60 + 59  # --time=18:59:00 → 1139 min (generic)

@dataclass
class ClusterConfig:
    name: str = "transformer"  # --job-name        (generic)
    slurm_account: str = "imi@cpu"  # --account
    # slurm_partition: str = ""  # --partition (uncomment/set as needed)
    slurm_qos: str = "qos_cpu-dev"  # --qos
    # gpus_per_node: int = 1  # --gres            (or: gpus_per_node=1)
    cpus_per_task: int = 20  # --cpus-per-task   (generic)
    timeout_min: int = 1 * 60 + 59  # --time=18:59:00 → 1139 min (generic)


def _entry(module_path: str):
    module = importlib.import_module(module_path)
    return module.main


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
    p.add_argument(
        "cluster_overrides",
        nargs="*",
        help="ClusterConfig dotlist, e.g. slurm_qos=... gpus_per_node=2",
    )
    ns = p.parse_args(launcher_argv)
    return ns.script, ns.cluster_overrides, script_argv


def snapshot_code(project_root: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
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


    script, cluster_overrides, script_argv = parse_args(argv)

    cfg_cluster = OmegaConf.merge(
        OmegaConf.structured(ClusterConfig),
        OmegaConf.from_dotlist(cluster_overrides),
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = paths.EXPERIMENTS_DIR / cfg_cluster.name / run_id
    snapshot_code(paths.ROOT, run_dir / "code")

    log_folder = run_dir / "slurm"
    executor = submitit.AutoExecutor(folder=log_folder, cluster=None)

    executor.update_parameters(
        **OmegaConf.to_container(cfg_cluster, resolve=True),
        # slurm_setup=[
        #     "module purge",
        #     "module load python/3.11.5",
        #     "module load cuda/12.8.0",
        #     "conda activate transformer",
        #     f"export PYTHONPATH={run_dir}:$PYTHONPATH",
        #     "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH",
        # ],
        slurm_setup=[
            "module purge",
            "module load pytorch-gpu/py3/2.8.0",  # check `module avail pytorch-gpu`
            f"export PYTHONPATH={run_dir / 'code'}:$PYTHONPATH",
        ],
    )

    yaml_path = script_argv[0] if script_argv and "=" not in script_argv[0] else None
    overrides = script_argv[1:] if yaml_path else script_argv
    job = executor.submit(
        _entry(script), 
        config_path=yaml_path, 
        overrides=[*overrides, f"save_dir={run_dir}"])

    print(f"Job submitted! Job ID: {job.job_id}")
    print(f"Run dir: {run_dir}")
    print(f"stdout: {job.paths.stdout}")
    print(f"stderr: {job.paths.stderr}")
    print(f"Cluster config: {cfg_cluster}")
    print(f"Submitted script: {script}")
    print(f"Script args: {script_argv}")

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
