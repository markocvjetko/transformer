#!/usr/bin/env bash
# Pull tokenized datasets (and tokenizer) from Jean Zay or Adastra, keeping the project layout.
# Usage: ./scripts/pull_data.sh [--cluster jz|adastra] [PATH...] [-- extra rsync flags, e.g. -- -n]
#   PATH is relative to the remote project root (e.g. data/gpt2-test-large/wikipedia_en/merged.npy).
#   Without paths, pulls everything config_large_next_gen.yaml needs.
#   Destination defaults to /media/mcvjetko/marko/transformer; override with DEST=...
set -euo pipefail

cluster=jz
dest=${DEST:-/media/mcvjetko/marko/transformer}

usage() {
    echo "Usage: $0 [--cluster jz|adastra] [PATH...] [-- extra rsync flags]"
    echo "  Cluster defaults to jz; --cluster=NAME also works."
    echo "  Without paths, pulls the default datasets and tokenizer."
}

default_files=(
    experiments/gpt2-test-large/BPE_vocab_48k_hr_en
    data/gpt2-test-large/wikipedia_en/merged.npy
    data/gpt2-test-large/wikipedia_hr/merged.npy
    data/gpt2-test-large/macocu_hr_en_v2_data.npy
    data/gpt2-test-large/classla_hr_v2/merged.npy
    data/gpt2-test-large/merged_fineweb_data.npy
)

files=()
rsync_extra=()
while (( $# )); do
    case "$1" in
        --) shift; rsync_extra=("$@"); break ;;
        --cluster)
            if (( $# < 2 )) || [[ $2 == -* ]]; then
                echo "--cluster requires jz or adastra." >&2
                exit 1
            fi
            cluster=$2
            shift
            ;;
        --cluster=*) cluster="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
        *) files+=("$1") ;;
    esac
    shift
done
(( ${#files[@]} )) || files=("${default_files[@]}")

case "$cluster" in
    jz)
        remote_host=${JZ_HOST:-jz}
        remote_root=${JZ_ROOT:-/lustre/fsn1/projects/rech/imi/uix29qp/proj/transformer}
        ;;
    adastra)
        remote_host=${ADASTRA_HOST:-adastra-agent}
        remote_root=${ADASTRA_ROOT:-/lus/scratch/CT10/iso1996/mcvjetko/proj/transformer}
        ;;
    *) echo "Unknown cluster: $cluster (expected jz or adastra)." >&2; exit 1 ;;
esac

for rel in "${files[@]}"; do
    if [[ ! "$rel" =~ ^[A-Za-z0-9._/-]+$ || "$rel" == *..* || "$rel" == /* ]]; then
        echo "Invalid path: $rel" >&2
        exit 1
    fi
done

mkdir -p -- "$dest"
# --files-from keeps each path relative to remote_root under dest; --partial lets reruns resume.
printf '%s\n' "${files[@]}" | rsync -a --partial --info=progress2 --human-readable \
    --files-from=- "${rsync_extra[@]}" \
    "$remote_host:${remote_root%/}/" "$dest/"
