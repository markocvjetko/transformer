#!/usr/bin/env bash
# Download a batch or one run from Jean Zay or Adastra. Checkpoints are optional.
set -euo pipefail

usage() {
    echo "Usage: $0 BATCH [RUN] [--cluster jz|adastra] [--checkpoints | --last-checkpoint] [--no-logs]"
    echo "  Cluster defaults to jz; --cluster=NAME also works."
    echo "  --last-checkpoint downloads only last.ckpt, for all runs if RUN is omitted."
    echo "  --last-experiment is an alias for --last-checkpoint."
    echo "  --no-logs skips slurm .out/.err files."
}

names=()
exclude_args=(--exclude='*.ckpt')
log_args=()
checkpoints=false
last_checkpoint=false
cluster=jz

while (( $# )); do
    case "$1" in
        --cluster)
            if (( $# < 2 )) || [[ $2 == -* ]]; then
                echo "--cluster requires jz or adastra." >&2
                exit 1
            fi
            cluster=$2
            shift
            ;;
        --cluster=*) cluster="${1#*=}" ;;
        --checkpoints) checkpoints=true ;;
        --last-checkpoint|--last-experiment) last_checkpoint=true ;;
        --no-logs) log_args=(--exclude='*.out' --exclude='*.err') ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
        *) names+=("$1") ;;
    esac
    shift
done

if (( ${#names[@]} < 1 || ${#names[@]} > 2 )); then
    usage >&2
    exit 1
fi

if $last_checkpoint; then
    if $checkpoints; then
        echo "--last-checkpoint/--last-experiment cannot be combined with --checkpoints." >&2
        exit 1
    fi
    exclude_args=(--include='*/' --include='last.ckpt' --exclude='*' --prune-empty-dirs)
elif $checkpoints; then
    exclude_args=()
fi

# Each argument names one directory, not a path or remote shell expression.
for name in "${names[@]}"; do
    if [[ ! "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
        echo "Invalid batch or run name: $name" >&2
        exit 1
    fi
done

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "$cluster" in
    jz)
        remote_host=${JZ_HOST:-jz}
        remote_root=${JZ_ROOT:-/lustre/fsn1/projects/rech/imi/uix29qp/proj/transformer}
        remote_root=${JZ_EXPERIMENTS_DIR:-${remote_root%/}/experiments}
        ;;
    adastra)
        remote_host=${ADASTRA_HOST:-adastra-agent}
        remote_root=${ADASTRA_ROOT:-/lus/scratch/CT10/iso1996/mcvjetko/proj/transformer}
        remote_root=${ADASTRA_EXPERIMENTS_DIR:-${remote_root%/}/experiments}
        ;;
    *) echo "Unknown cluster: $cluster (expected jz or adastra)." >&2; exit 1 ;;
esac
relative_path=${names[0]}
if (( ${#names[@]} == 2 )); then
    relative_path+=/${names[1]}
fi

local_path="$project_root/experiments/$relative_path"
mkdir -p -- "$local_path"
rsync -avP --protect-args "${log_args[@]}" "${exclude_args[@]}" \
    "$remote_host:${remote_root%/}/$relative_path/" \
    "$local_path/"
