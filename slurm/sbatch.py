# scripts/submit.py
import subprocess, sys, datetime, pathlib, os

SBATCH = pathlib.Path(__file__).parent / "train.sbatch"

exp_name = sys.argv[1] if len(sys.argv) > 1 else "run"
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
exp_dir = pathlib.Path(os.environ["SCRATCH"]) / "transformer/experiments"# / f"{exp_name}-{stamp}"
log_dir = exp_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

git_hash = git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

cmd = [
    "sbatch",
    f"--job-name={exp_name}",
    f"--output={log_dir}/%j.out",
    f"--error={log_dir}/%j.err",
    f"--export=ALL,EXP_DIR={exp_dir},GIT_HASH={git_hash}",   # pass path into the job's env
    str(SBATCH),
]
subprocess.run(cmd, check=True)
