from argparse import ArgumentParser
import datetime
import os
import subprocess
from pathlib import Path

import submitit


def _entry():
    from src.scripts.train_hr_en import main
    return main()


if __name__ == "__main__":


    root = "SCRATCH"
    project_root = Path(__file__).resolve().parent.parent
    log_root = Path(os.environ[root]) / "proj" / "transformer" / "submitit_hren"
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
    executor.update_parameters(
        name="transformer",
        slurm_account="imi@v100",
        slurm_partition="gpu_p13",
        slurm_gres="gpu:1",
        cpus_per_task=10,
        timeout_min=18 * 60 + 59,
        slurm_additional_parameters={"hint": "nomultithread"},
        slurm_setup=[
            "module purge",
            "module load python/3.11.5",
            "module load cuda/12.8.0",
            "conda activate transformer",
            "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH",
            "export DATA_DIR=${" + root + "}/proj/transformer/data",
            "export EXPERIMENTS_DIR=${" + root + "}/proj/transformer/experiments",
            "export HF_HOME=${" + root + "}/.cache/huggingface",
            "export ON_JZ=TRUE",
            f"export PYTHONPATH={snapshot}:$PYTHONPATH",
            "mkdir -p $DATA_DIR $EXPERIMENTS_DIR",
        ],
    )

    job = executor.submit(_entry, git_hash=git_hash)
    print(f"Submitted {job.job_id}  →  {run_dir}")