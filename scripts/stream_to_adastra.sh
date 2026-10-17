#!/usr/bin/env bash
# Stream files from Jean Zay to ADASTRA through this machine, one at a time,
# without touching local disk. Each file is hashed in flight, re-hashed on
# ADASTRA, and only renamed into place (with a .sha256 sidecar) if both match.
# Re-running skips files that already arrived verified.
#
# Usage: ./scripts/stream_to_adastra.sh [PATH...]
#   PATH is relative to the project root on both clusters (e.g. data/gpt2-test-large/wikipedia_en/merged.npy).
#   Without arguments, sends everything config_large_next_gen.yaml needs.
#   ADASTRA only accepts passwords, so open the master connection first: ssh adastra-agent
set -euo pipefail

src_host=${JZ_HOST:-jz}
dst_host=${ADASTRA_HOST:-adastra-agent}
src_root=${JZ_ROOT:-/lustre/fsn1/projects/rech/imi/uix29qp/proj/transformer}
dst_root=${ADASTRA_ROOT:-/lus/scratch/CT10/iso1996/mcvjetko/proj/transformer}

# Smallest first, so problems show up before the 21 GB file.
default_files=(
    experiments/gpt2-test-large/BPE_vocab_48k_hr_en
    data/gpt2-test-large/wikipedia_en/merged.npy
    data/gpt2-test-large/wikipedia_hr/merged.npy
    data/gpt2-test-large/macocu_hr_en_v2_data.npy
    data/gpt2-test-large/classla_hr_v2/merged.npy
    data/gpt2-test-large/merged_fineweb_data.npy
)
if (( $# )); then files=("$@"); else files=("${default_files[@]}"); fi

# Paths are interpolated into remote shell commands, so keep them boring.
for rel in "${files[@]}"; do
    if [[ ! "$rel" =~ ^[A-Za-z0-9._/-]+$ || "$rel" == *..* || "$rel" == /* ]]; then
        echo "Invalid path: $rel" >&2
        exit 1
    fi
done

# Copies stdin to stdout, hashing as it goes and drawing a progress line on stderr.
relay='
import hashlib, sys, time
total, name, hash_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
inp, out = sys.stdin.buffer, sys.stdout.buffer
h, n, t0, last = hashlib.sha256(), 0, time.time(), 0.0

def show(end=""):
    dt = max(time.time() - t0, 1e-9)
    rate = n / dt
    eta = (total - n) / rate if rate else 0
    pct = 100 * n / total if total else 100
    sys.stderr.write(f"\r  {n/1e9:7.2f}/{total/1e9:.2f} GB {pct:5.1f}%  {rate/1e6:6.1f} MB/s  "
                     f"elapsed {int(dt)//60:02d}:{int(dt)%60:02d}  ETA {int(eta)//60:02d}:{int(eta)%60:02d} {end}")
    sys.stderr.flush()

while chunk := inp.read(1 << 22):
    out.write(chunk)
    h.update(chunk)
    n += len(chunk)
    if time.time() - last >= 0.5:
        show()
        last = time.time()
out.flush()
show("\n")
if n != total:
    sys.exit(f"  short read: got {n} of {total} bytes")
open(hash_path, "w").write(h.hexdigest())
'

if ! ssh -n -o BatchMode=yes "$dst_host" true 2>/dev/null; then
    echo "Cannot reach $dst_host. Open the master connection first: ssh $dst_host" >&2
    exit 1
fi

hash_file=$(mktemp)
trap 'rm -f "$hash_file"' EXIT
failed=()

for i in "${!files[@]}"; do
    rel=${files[$i]}
    src=$src_root/$rel
    dst=$dst_root/$rel
    echo "[$((i + 1))/${#files[@]}] $rel"

    size=$(ssh -n -o BatchMode=yes "$src_host" "stat -c %s $src")
    if ssh -n -o BatchMode=yes "$dst_host" \
        "test -f $dst.sha256 && test \"\$(stat -c %s $dst 2>/dev/null)\" = $size"; then
        echo "  already on $dst_host and verified, skipping"
        continue
    fi

    if ! ssh -o BatchMode=yes "$src_host" "cat $src" \
        | python3 -I -c "$relay" "$size" "$rel" "$hash_file" \
        | ssh -o BatchMode=yes "$dst_host" "mkdir -p \"\$(dirname $dst)\" && cat > $dst.part"; then
        echo "  transfer failed" >&2
        failed+=("$rel")
        continue
    fi

    sent=$(<"$hash_file")
    echo "  verifying on $dst_host..."
    got=$(ssh -n -o BatchMode=yes "$dst_host" "sha256sum $dst.part" | cut -d' ' -f1)
    if [[ "$sent" != "$got" ]]; then
        echo "  checksum mismatch: sent $sent, got $got (left as $dst.part)" >&2
        failed+=("$rel")
        continue
    fi
    ssh -n -o BatchMode=yes "$dst_host" \
        "mv $dst.part $dst && echo '$sent  $(basename "$dst")' > $dst.sha256"
    echo "  ok  sha256 $sent"
done

if (( ${#failed[@]} )); then
    echo "Failed (re-run to retry): ${failed[*]}" >&2
    exit 1
fi
echo "All ${#files[@]} files on $dst_host."
