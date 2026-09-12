#!/usr/bin/env bash
#
# count_extensions.sh — count files by extension in a directory tree.
#
# Usage:
#   ./count_extensions.sh [DIRECTORY]    # defaults to current directory
#
# Notes:
#   - Recurses into all subdirectories.
#   - Extensions are lowercased so .PDF, .Pdf and .pdf are counted together.
#   - Files with no extension (e.g. "Makefile") and dotfiles (e.g. ".bashrc")
#     are grouped under "(no extension)".
#   - Output is sorted by count, descending, with a TOTAL row.
#   - Requires GNU find (standard on Linux) for the -printf option.

set -euo pipefail

DIR="${1:-.}"

if [[ ! -d "$DIR" ]]; then
    echo "Error: '$DIR' is not a directory." >&2
    exit 1
fi

find "$DIR" -type f -printf '%f\n' \
| awk '
    {
        name = $0
        # locate the last dot, ignoring a leading dot (dotfiles)
        dot = 0
        for (i = length(name); i > 1; i--) {
            if (substr(name, i, 1) == ".") { dot = i; break }
        }
        if (dot > 1 && dot < length(name))
            ext = tolower(substr(name, dot + 1))
        else
            ext = "(no extension)"
        count[ext]++
    }
    END { for (e in count) printf "%d\t%s\n", count[e], e }
' \
| sort -rn \
| awk -F'\t' '
    { c[NR]=$1; n[NR]=$2; total+=$1; if (length($2)>w) w=length($2) }
    END {
        for (i = 1; i <= NR; i++) printf "  %-*s  %d\n", w, n[i], c[i]
        printf "  %-*s  %d\n", w, "TOTAL", total
    }
'