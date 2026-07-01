"""Build the monograph PDF from the chapter Markdown via pandoc + xelatex.

Concatenates chapters 00–08 (README is navigation, excluded) into a single report-class
PDF with a title page and table of contents, and writes it to monograph/SR-Core-Monograph.pdf.

Requirements:
  - pandoc (either on PATH, or install the `pypandoc_binary` pip package which bundles it)
  - a LaTeX distribution with xelatex (e.g. MiKTeX or TeX Live)
  - the DejaVu Serif / DejaVu Sans Mono fonts (ship with most LaTeX distributions)

Run from the repository root:
    python monograph/scripts/build_pdf.py
"""
from __future__ import annotations
import glob, os, shutil, subprocess, sys

MONO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # monograph/
CHAPTERS = os.path.join(MONO, "docs", "chapters")
HEADER = os.path.join(MONO, "scripts", "latex_header.tex")
OUT = os.path.join(MONO, "SR-Core-Monograph.pdf")

TITLE_BLOCK = """---
title: "Recursive Block-Sparse Language Models"
subtitle: "SR-Core: A Hard Working-Set Guarantee for Cache-Efficient Parameter Streaming — A Research Monograph"
author: "Viktor Jedich"
date: "2026"
documentclass: report
toc: true
toc-depth: 2
numbersections: false
geometry: "margin=1in"
colorlinks: true
linkcolor: RoyalBlue
urlcolor: RoyalBlue
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
monofontoptions: "Scale=0.82"
---

"""


def find_pandoc() -> str:
    """Prefer pandoc on PATH; fall back to the binary bundled with pypandoc_binary."""
    p = shutil.which("pandoc")
    if p:
        return p
    try:
        import pypandoc
        return pypandoc.get_pandoc_path()
    except Exception:
        sys.exit("pandoc not found. Install it, or: pip install pypandoc_binary")


def main() -> None:
    chapters = sorted(glob.glob(os.path.join(CHAPTERS, "0*.md")))
    if not chapters:
        sys.exit(f"No chapters found in {CHAPTERS}")
    print("Chapters:")
    for c in chapters:
        print(f"  {os.path.basename(c)}")

    # Front-matter YAML must be its own input file so pandoc parses it as metadata.
    front = os.path.join(CHAPTERS, "_title_block.md")
    with open(front, "w", encoding="utf-8") as f:
        f.write(TITLE_BLOCK)

    pandoc = find_pandoc()
    cmd = [
        pandoc, front, *chapters,
        "--from", "gfm+yaml_metadata_block+tex_math_dollars",
        "--pdf-engine=xelatex",
        "-H", HEADER,
        "--top-level-division=chapter",
        "-o", OUT,
    ]
    print(f"\npandoc: {pandoc}\nBuilding {OUT} ...")
    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(front):
            os.remove(front)

    size = os.path.getsize(OUT) / 1024
    print(f"\nWrote {OUT}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
