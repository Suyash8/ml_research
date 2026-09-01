#!/usr/bin/env python3
"""
Stage 0 of paper verification: turn a compiled PDF into things that can be
inspected.

Produces, under <outdir>:

    pdf_text.txt          plain text, reading order, page markers
    pdf_text_layout.txt   plain text preserving column layout
    pages/page-NN.png     one raster per page, for visual inspection
    embedded_images.tsv   every raster the PDF actually embeds, with true ppi
    pdfinfo.txt           page count, page size, producer, fonts
    fonts.txt             font inventory, to catch missing embeds
    manifest.json         everything above, machine readable

Why this exists: reading the .tex tells you what was requested. Reading the PDF
tells you what a reviewer will actually see. Those differ, and the differences
are where the embarrassing problems live: a figure upscaled past legibility, a
table running off the page, a caption orphaned from its float.

Usage:
    python3 extract_artifacts.py <paper.pdf> <outdir> [--dpi 110]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        sys.exit(
            f"required tool not found: {tool}\n"
            "install poppler-utils (pdftotext, pdftoppm, pdfimages, pdfinfo)"
        )
    return path


def run(cmd: list[str], out: Path | None = None) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stdout
    if out is not None:
        out.write_text(text + (("\n[stderr]\n" + proc.stderr) if proc.stderr else ""))
    return text


def parse_pdfimages_list(raw: str) -> list[dict]:
    """
    pdfimages -list emits a fixed-width table. Columns of interest:
    page, num, type, width, height, color, comp, bpc, enc, interp, object,
    ID, x-ppi, y-ppi, size, ratio
    """
    rows: list[dict] = []
    lines = [l for l in raw.splitlines() if l.strip()]
    if len(lines) < 2:
        return rows
    header = lines[0].split()
    for line in lines[2:] if lines[1].startswith("-") else lines[1:]:
        parts = line.split()
        if len(parts) < len(header):
            continue
        rec = dict(zip(header, parts))
        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("outdir")
    ap.add_argument("--dpi", type=int, default=110,
                    help="raster resolution for page images (default 110)")
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    if not pdf.exists():
        sys.exit(f"no such pdf: {pdf}")

    out = Path(args.outdir).resolve()
    pages = out / "pages"
    pages.mkdir(parents=True, exist_ok=True)

    need("pdftotext"); need("pdftoppm"); need("pdfimages"); need("pdfinfo")

    manifest: dict = {"pdf": str(pdf), "outdir": str(out), "dpi": args.dpi}

    # --- metadata ---------------------------------------------------------
    info_raw = run(["pdfinfo", str(pdf)], out / "pdfinfo.txt")
    info: dict[str, str] = {}
    for line in info_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    manifest["pdfinfo"] = info
    n_pages = int(info.get("Pages", "0") or 0)
    manifest["n_pages"] = n_pages

    # --- fonts ------------------------------------------------------------
    if shutil.which("pdffonts"):
        fonts_raw = run(["pdffonts", str(pdf)], out / "fonts.txt")
        not_embedded = []
        for line in fonts_raw.splitlines()[2:]:
            parts = line.split()
            # name type encoding emb sub uni object ID
            if len(parts) >= 6 and parts[-5] == "no":
                not_embedded.append(parts[0])
        manifest["fonts_not_embedded"] = not_embedded

    # --- text -------------------------------------------------------------
    run(["pdftotext", "-q", str(pdf), "-"], out / "pdf_text.txt")
    run(["pdftotext", "-q", "-layout", str(pdf), "-"], out / "pdf_text_layout.txt")
    txt = (out / "pdf_text.txt").read_text(errors="replace")
    manifest["pdf_text_chars"] = len(txt)
    manifest["pdf_text_words"] = len(txt.split())

    # --- page rasters -----------------------------------------------------
    for f in pages.glob("page-*.png"):
        f.unlink()
    subprocess.run(
        ["pdftoppm", "-q", "-png", "-r", str(args.dpi), str(pdf), str(pages / "page")],
        capture_output=True, text=True,
    )
    page_files = sorted(pages.glob("page-*.png"))
    manifest["page_images"] = [str(p.relative_to(out)) for p in page_files]

    # Report page image sizes so the caller knows the read cost up front.
    sizes = []
    for p in page_files:
        sizes.append({"file": p.name, "bytes": p.stat().st_size})
    manifest["page_image_sizes"] = sizes
    manifest["page_image_total_bytes"] = sum(s["bytes"] for s in sizes)

    # --- embedded rasters and their true resolution -----------------------
    img_raw = run(["pdfimages", "-list", str(pdf)])
    rows = parse_pdfimages_list(img_raw)
    tsv = out / "embedded_images.tsv"
    if rows:
        cols = list(rows[0].keys())
        with tsv.open("w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(r.get(c, "") for c in cols) + "\n")
    else:
        tsv.write_text("no embedded rasters found\n")
    manifest["embedded_images"] = rows
    manifest["n_embedded_images"] = len(rows)

    # --- quick summary to stdout -----------------------------------------
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"pdf              : {pdf}")
    print(f"pages            : {n_pages}")
    print(f"page size        : {info.get('Page size', '?')}")
    print(f"producer         : {info.get('Producer', '?')}")
    print(f"text words       : {manifest['pdf_text_words']}")
    print(f"page rasters     : {len(page_files)} at {args.dpi} dpi "
          f"({manifest['page_image_total_bytes'] / 1e6:.1f} MB total)")
    print(f"embedded rasters : {len(rows)}")
    if manifest.get("fonts_not_embedded"):
        print(f"FONTS NOT EMBEDDED: {manifest['fonts_not_embedded']}")
    print(f"\nwrote {out}")
    print("next: read every file under pages/ as an image, one at a time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
