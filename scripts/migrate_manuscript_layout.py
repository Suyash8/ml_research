#!/usr/bin/env python3
"""
One-shot migration: flat repo root -> versioned manuscript/ tree.

Layout produced:

    manuscript/
      README.md
      shared/
        figures/            all figure PNGs, single copy
        references.bib      master verified bibliography
      v1-cox-enet-pipeline/ archived first submission (was paper.tex)
      v2-multiomic-tcga/    archived second submission (was paper_new.tex)
      v3-simplified/        active rewrite

Archived .tex files are moved byte-for-byte. A relative symlink named
"images" is placed inside each version folder so their original
\\includegraphics{images/...} paths keep resolving against shared/figures.

Idempotent: re-running after a partial run finishes the job.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / "manuscript"
SHARED = MS / "shared"
FIGURES = SHARED / "figures"

V1 = MS / "v1-cox-enet-pipeline"
V2 = MS / "v2-multiomic-tcga"
V3 = MS / "v3-simplified"

AUX_EXTS = ("aux", "bbl", "blg", "log", "out", "toc", "synctex.gz", "fls", "fdb_latexmk")

log: list[str] = []


def note(msg: str) -> None:
    log.append(msg)
    print(msg)


def git_mv(src: Path, dst: Path) -> None:
    """Move src -> dst, preferring `git mv` so history follows the file."""
    if not src.exists():
        return
    if dst.exists():
        note(f"skip (dst exists): {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "mv", str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        note(f"git mv  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    except (subprocess.CalledProcessError, ValueError):
        shutil.move(str(src), str(dst))
        note(f"mv      {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def relink(link: Path, target: str) -> None:
    """Create or replace a relative symlink."""
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            # A real directory sitting where the symlink belongs. Leave it alone.
            note(f"skip symlink, real dir present: {link.relative_to(ROOT)}")
            return
        else:
            link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    note(f"symlink {link.relative_to(ROOT)} -> {target}")


def main() -> None:
    for d in (FIGURES, V1 / "build", V2 / "build", V3):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. figures into one shared place -------------------------------
    old_images = ROOT / "images"
    if old_images.is_dir() and not old_images.is_symlink():
        for png in sorted(old_images.glob("*.png")):
            git_mv(png, FIGURES / png.name)
        leftovers = list(old_images.iterdir())
        if not leftovers:
            old_images.rmdir()
            note("rmdir   images/")
        else:
            note(f"images/ not empty, left in place: {[p.name for p in leftovers]}")

    # ---- 2. v1: paper.tex family ----------------------------------------
    git_mv(ROOT / "paper.tex", V1 / "paper.tex")
    for name in ("paper.pdf", "paper.docx", "paper.md"):
        git_mv(ROOT / name, V1 / name)
    for ext in AUX_EXTS:
        git_mv(ROOT / f"paper.{ext}", V1 / "build" / f"paper.{ext}")

    # ---- 3. v2: paper_new.tex family ------------------------------------
    git_mv(ROOT / "paper_new.tex", V2 / "paper.tex")
    git_mv(ROOT / "paper_new.pdf", V2 / "paper.pdf")
    for ext in AUX_EXTS:
        git_mv(ROOT / f"paper_new.{ext}", V2 / "build" / f"paper_new.{ext}")

    # ---- 4. bibliography ------------------------------------------------
    root_bib = ROOT / "references.bib"
    if root_bib.exists():
        # Freeze the as-submitted bibliography alongside each archived version
        # so those PDFs stay reproducible even after the master is corrected.
        for dest in (V1 / "references.bib", V2 / "references.bib"):
            if not dest.exists():
                shutil.copy2(root_bib, dest)
                note(f"copy    references.bib -> {dest.relative_to(ROOT)}")
        git_mv(root_bib, SHARED / "references.bib")

    # ---- 5. symlinks ----------------------------------------------------
    for v in (V1, V2, V3):
        relink(v / "images", "../shared/figures")
    relink(V3 / "references.bib", "../shared/references.bib")
    # Keep a root-level images/ alias: several scripts/ still write there.
    relink(ROOT / "images", "manuscript/shared/figures")

    report = MS / "MIGRATION_LOG.txt"
    report.write_text("\n".join(log) + "\n")
    note(f"\nwrote {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
