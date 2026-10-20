cd /home/mcvjetko/phd/projects/transformer
files=(
    data/interim/settimes-bs-en-v2/SETIMES-bs-en.jsonl
    data/interim/settimes-en-hr-v2/SETIMES-en-hr.jsonl
    data/interim/macocu-bs-en-sent/MacoCu-bs-en.jsonl
    data/interim/dgt-en-hr-v2021/DGT-en-hr.jsonl
    data/interim/riznica/riznica.jsonl
    data/interim/macocu-en-hr-sent/MacoCu-en-hr.jsonl
    data/interim/classla-web-bs-v1/MaCoCu-Genre-DEDUPLICATED.bs.jsonl
    data/raw/classla-web-bs-v2/CLASSLA-web.bs.2.0.jsonl
    data/interim/classla-web-hr-v1/MaCoCu-Genre-DEDUPLICATED.hr.jsonl
    data/raw/classla-web-hr-v2/CLASSLA-web.hr.2.0.jsonl
)
until rsync -ah --relative -z \
    --partial-dir=.rsync-partial \
    --info=progress2,name1 \
    "${files[@]}" \
    adastra-agent:/lus/scratch/CT10/iso1996/mcvjetko/proj/transformer/
do
    echo "rsync failed (exit $?), retrying in 30s..." >&2
    sleep 30
done
