import json
import re

from src.utils import paths

riznica_path = paths.DATA_DIR / "raw/Riznica/riznica.vert"

ATTRS = re.compile(r'(\w+)="([^"]*)"')


def iter_texts(path):
    meta, paras, cur, glue = {}, [], [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t")

            if len(parts) == 3:  # token line
                if cur and not glue:
                    cur.append(" ")
                cur.append(parts[0])
                glue = False
            elif line == "<g/>":
                glue = True
            elif line.startswith("<p") or line in ("</p>", "</text>"):
                if cur:  # flush paragraph
                    paras.append("".join(cur))
                    cur = []
                    glue = False
                if line == "</text>":
                    yield meta, "\n".join(paras)
                    meta, paras = {}, []
            elif line.startswith("<text "):
                meta = dict(ATTRS.findall(line))
    if cur:
        paras.append("".join(cur))
    if paras:
        yield meta, "\n".join(paras)  # unterminated tail


with open(paths.DATA_DIR / "riznica.jsonl", "w", encoding="utf-8") as out:
    for meta, text in iter_texts(riznica_path):
        out.write(json.dumps({"text": text, **meta}, ensure_ascii=False) + "\n")
