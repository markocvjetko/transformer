#!/bin/bash
# Activate the env on the login node, then submit a job.
# Usage: ./slurm/submit.sh src.scripts.train_hr_en [--partition ... --cpus ...]
set -euo pipefail
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/activate_env.sh"
python "$SCRIPT_DIR/submit_it.py" "$@"
