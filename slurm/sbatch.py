# scripts/submit.py
import subprocess, sys, datetime, pathlib, os

exp_name = sys.argv[1] if len(sys.argv) > 1 else "run"
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
exp_dir = pathlib.Path(os.environ["SCRATCH"]) / "transformer/experiments"# / f"{exp_name}-{stamp}"
log_dir = exp_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    "sbatch",
    f"--job-name={exp_name}",
    f"--output={log_dir}/%j.out",
    f"--error={log_dir}/%j.err",
    f"--export=ALL,EXP_DIR={exp_dir}",   # pass path into the job's env
    "slurm/train.sbatch",
]
subprocess.run(cmd, check=True)
