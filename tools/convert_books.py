"""Convert OneDrive book PDFs/DOCs to cleaned Croatian-only text.

Usage:
    python tools/convert_books.py [--workers N] [--limit N] [--force]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import fasttext
# numpy>=2 + fasttext-wheel: predict() calls np.array(..., copy=False) which is
# now disallowed. Replace with np.asarray.
def _predict_patched(self, text, k=1, threshold=0.0, on_unicode_error='strict'):
    # Replacement that avoids np.array(..., copy=False) (banned in numpy>=2).
    if "\n" in text:
        text = text.replace("\n", " ")
    predictions = self.f.predict(text, k, threshold, on_unicode_error)
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = ([], ())
    return labels, np.asarray(probs, dtype=np.float32)
fasttext.FastText._FastText.predict = _predict_patched

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "OneDrive_343_5-3-2026 (2)"
DST = ROOT / "data" / "books_txt"
LID_PATH = ROOT / "tools" / "models" / "lid.176.bin"

# Per-process language identifier (initialized in worker).
_LID = None


def _init_worker():
    global _LID
    # Silence fasttext warning.
    fasttext.FastText.eprint = lambda *a, **k: None
    _LID = fasttext.load_model(str(LID_PATH))


def extract_pdf(src: Path) -> str:
    """Run pdftotext; return UTF-8 text or empty on failure."""
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", "-nopgbrk", str(src), "-"],
            capture_output=True, timeout=300, check=False,
        )
        return out.stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_doc(src: Path) -> str:
    """Use libreoffice headless to convert .doc -> .txt."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "txt:Text (encoded):UTF8",
                 "--outdir", td, str(src)],
                capture_output=True, timeout=600, check=False,
            )
            for p in Path(td).glob("*.txt"):
                return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return ""


# Lines that are page-numbers, dotted TOC entries, repeated separators, etc.
_PAGE_NUM_RE = re.compile(r"^\s*[ivxlcdm]*\s*\d{1,4}\s*[ivxlcdm]*\s*$", re.IGNORECASE)
_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$")
_NON_LETTER_RE = re.compile(r"[^\w]", re.UNICODE)
_WS_RE = re.compile(r"[ \t]+")


def strip_repeated_lines(text: str) -> str:
    """Remove lines that look like running headers/footers (appear many times)."""
    lines = text.splitlines()
    # Normalize for frequency counting (collapse whitespace, strip).
    norm = [_WS_RE.sub(" ", ln).strip() for ln in lines]
    cnt = Counter(n for n in norm if n)
    # If a non-trivial short line repeats often, treat as boilerplate.
    n_pages_est = max(1, text.count("\f") + 1)
    threshold = max(5, int(0.2 * n_pages_est))
    boiler = {n for n, c in cnt.items()
              if c >= threshold and 0 < len(n) <= 80}
    return "\n".join(ln for ln, n in zip(lines, norm) if n not in boiler)


def split_paragraphs(text: str) -> list[str]:
    """Split into paragraphs on blank lines OR on indentation changes
    (`-layout` mode often separates paragraphs only by indent, not blank lines).
    A line with >=2 leading spaces starts a new paragraph; lines starting at
    column 0 continue the previous paragraph (line-wrap)."""
    paras: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            paras.append(" ".join(s.strip() for s in buf))
            buf.clear()

    for ln in text.splitlines():
        if not ln.strip():
            flush()
            continue
        leading = len(ln) - len(ln.lstrip(" "))
        if leading >= 2 and buf:
            # Indented line -> new paragraph.
            flush()
        buf.append(ln)
    flush()
    return paras


def clean_paragraph(p: str) -> str:
    p = _WS_RE.sub(" ", p)
    return p.strip()


def keep_paragraph(p: str) -> bool:
    """Filter: drop too short, table-of-contents dot-leaders, page-number-only,
    and non-Croatian paragraphs."""
    if len(p) < 30:
        return False
    if _PAGE_NUM_RE.match(p):
        return False
    if _DOT_LEADER_RE.search(p):
        return False
    # Many digits / few letters -> likely TOC / metadata.
    letters = sum(c.isalpha() for c in p)
    if letters < 0.5 * len(p):
        return False
    # Language ID. fastText needs single line.
    labels, probs = _LID.predict(p.replace("\n", " "), k=3)
    top = labels[0].replace("__label__", "")
    top_p = float(probs[0])
    if top == "hr" and top_p > 0.5:
        return True
    # Accept bs/sr/sh as Croatian-like only if Croatian-specific diacritics present.
    if top in ("bs", "sr", "sh") and top_p > 0.6:
        if any(ch in p for ch in "ćčžšđ"):
            return True
    # Sometimes hr is #2 behind sr/bs — accept if hr is in top-3 and reasonable.
    for lab, pr in zip(labels, probs):
        if lab.replace("__label__", "") == "hr" and float(pr) > 0.25:
            return True
    return False


def clean_text(raw: str) -> str:
    raw = raw.replace("\f", "\n\n")
    raw = strip_repeated_lines(raw)
    paras = split_paragraphs(raw)
    kept = []
    for p in paras:
        p = clean_paragraph(p)
        if keep_paragraph(p):
            kept.append(p)
    return "\n\n".join(kept) + ("\n" if kept else "")


def process_one(args: tuple[str, str, bool]) -> tuple[str, int, int]:
    src_path_str, dst_path_str, force = args
    src = Path(src_path_str)
    dst = Path(dst_path_str)
    if dst.exists() and not force and dst.stat().st_size > 0:
        return (src_path_str, dst.stat().st_size, 0)
    ext = src.suffix.lower()
    if ext == ".pdf":
        raw = extract_pdf(src)
    elif ext == ".doc":
        raw = extract_doc(src)
    else:
        return (src_path_str, 0, -1)
    if not raw:
        return (src_path_str, 0, -2)
    cleaned = clean_text(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")
    return (src_path_str, len(cleaned), len(raw))


def collect_jobs(force: bool, limit: int | None) -> list[tuple[str, str, bool]]:
    jobs = []
    for src in SRC.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in (".pdf", ".doc"):
            continue
        rel = src.relative_to(SRC).with_suffix(".txt")
        dst = DST / rel
        jobs.append((str(src), str(dst), force))
    jobs.sort()
    if limit:
        jobs = jobs[:limit]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = collect_jobs(args.force, args.limit)
    print(f"Jobs: {len(jobs)}  workers: {args.workers}", flush=True)

    done = 0
    failed = []
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for src_path, n_clean, n_raw in pool.imap_unordered(process_one, jobs, chunksize=1):
            done += 1
            if n_raw == -2 or (n_raw > 0 and n_clean == 0):
                failed.append(src_path)
            if done % 25 == 0 or done == len(jobs):
                pct = 100 * done / len(jobs)
                last = Path(src_path).name
                print(f"[{done:4d}/{len(jobs)}  {pct:5.1f}%] clean={n_clean:>9}  raw={n_raw:>9}  {last}",
                      flush=True)
    print(f"Done. Failed: {len(failed)}")
    for f in failed[:20]:
        print("  FAIL:", f)
    if len(failed) > 20:
        print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
