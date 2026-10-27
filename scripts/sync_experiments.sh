#!/usr/bin/env bash
# Sync downloaded offline W&B logs for a batch or one run.
set -euo pipefail

usage() {
    echo "Usage: $0 BATCH [RUN]"
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# < 1 || $# > 2 )); then
    usage >&2
    exit 1
fi
for name in "$@"; do
    if [[ ! "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
        echo "Invalid batch or run name: $name" >&2
        exit 1
    fi
done

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
target="$project_root/experiments/$1"
if (( $# == 2 )); then
    target+="/$2"
fi
if [[ ! -d "$target" ]]; then
    echo "Download directory not found: $target" >&2
    exit 1
fi

# Only inspect each run's top-level W&B logs, not its code snapshot.
shopt -s nullglob
if (( $# == 2 )); then
    runs=("$target"/wandb/offline-run-*)
else
    runs=("$target"/*/wandb/offline-run-*)
fi
if (( ${#runs[@]} == 0 )); then
    echo "No offline W&B runs found in $target"
    exit 0
fi

wandb_command=wandb
if ! command -v wandb >/dev/null 2>&1; then
    wandb_command="$project_root/.venv/bin/wandb"
    if [[ ! -x "$wandb_command" ]]; then
        echo "W&B CLI not found. Activate an environment containing wandb." >&2
        exit 1
    fi
fi

# Revisit previously synced runs after pulling newer logs from ongoing jobs.
# Sync runs one at a time: a run pulled mid-write ends in a partial record,
# and one unreadable run would otherwise abort the whole batch.
failed=()
for run in "${runs[@]}"; do
    "$wandb_command" sync --include-synced "$run" || failed+=("$run")
done
if (( ${#failed[@]} > 0 )); then
    echo "Incomplete sync for ${#failed[@]} run(s) (still running or killed mid-write; pull and sync again later):" >&2
    printf '  %s\n' "${failed[@]}" >&2
fi
