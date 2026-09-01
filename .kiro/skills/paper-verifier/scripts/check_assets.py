#!/usr/bin/env python3
"""
Figure and image verification, on both the source files and the compiled PDF.

    python3 check_assets.py <paper.tex> --extracted <extract_dir> [--repo <root>] [--json out.json]

The source files say what was requested. The PDF says what the reviewer will
actually see, at the resolution they will see it. Both are checked, because a
2460x1575 source image placed across a full text width is fine while the same
image inside a two-column subfigure may not be.

The reproducibility check matters more than it first appears. In a research
paper every figure should be produced by code in the repository. A figure with
no generating script is either hand-drawn, pasted from elsewhere, or generated
by a model, and all three need a human to look at it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import Report, emit, read_tex, strip_comments  # noqa: E402

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

# Print thresholds. 300 ppi is the usual publisher floor for raster art;
# 600 ppi is asked for line art by some journals.
PPI_BLOCKER = 150
PPI_MAJOR = 300
PPI_MINOR = 600

VECTOR_EXT = {".pdf", ".eps", ".ps", ".svg"}
RASTER_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def parse_includegraphics(raw: str) -> list[dict]:
    """Every \\includegraphics with its options and path."""
    out = []
    for m in re.finditer(
        r"\\includegraphics\s*(?:\[([^\]]*)\])?\s*\{([^}]*)\}", strip_comments(raw)
    ):
        opts_raw = (m.group(1) or "").strip()
        opts = {}
        for part in re.split(r",(?![^{}]*\})", opts_raw):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                opts[k.strip()] = v.strip()
            else:
                opts[part] = True
        out.append({"path": m.group(2).strip(), "options": opts, "offset": m.start()})
    return out


def resolve(tex_dir: Path, ref: str) -> Path | None:
    cand = [tex_dir / ref]
    if not Path(ref).suffix:
        for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps"):
            cand.append(tex_dir / (ref + ext))
    for c in cand:
        if c.exists():
            return c.resolve()
    return None


def find_generating_scripts(repo: Path, filename: str) -> list[str]:
    """Any tracked source file that writes this figure name."""
    stem = Path(filename).stem
    hits = []
    for pat in ("*.py", "*.ipynb", "*.R", "*.sh", "*.jl", "*.m"):
        for f in repo.rglob(pat):
            parts = set(f.parts)
            if parts & {".venv", "venv", "node_modules", ".git", "__pycache__", "build"}:
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if stem in txt:
                hits.append(str(f.relative_to(repo)))
    return sorted(set(hits))


def check_source_files(rep: Report, tex: Path, uses: list[dict], repo: Path) -> dict:
    tex_dir = tex.parent
    info: dict[str, dict] = {}
    missing = []
    raster_diagrams = []
    no_script = []
    distorted = []

    for u in uses:
        ref = u["path"]
        p = resolve(tex_dir, ref)
        if p is None:
            missing.append(ref)
            continue
        rec: dict = {"ref": ref, "file": str(p), "bytes": p.stat().st_size,
                     "ext": p.suffix.lower()}
        if HAVE_PIL and p.suffix.lower() in RASTER_EXT:
            try:
                with Image.open(p) as im:
                    rec["pixels"] = list(im.size)
                    rec["mode"] = im.mode
                    rec["dpi_metadata"] = im.info.get("dpi")
                    rec["aspect"] = round(im.size[0] / im.size[1], 4)
            except Exception as e:  # noqa: BLE001
                rec["pil_error"] = str(e)
        rec["options"] = {k: (v if v is not True else "") for k, v in u["options"].items()}

        # Both width and height fixed, with no keepaspectratio, distorts.
        o = u["options"]
        if "width" in o and "height" in o and "keepaspectratio" not in o:
            distorted.append(ref)

        # Raster used for what is almost certainly line art.
        if p.suffix.lower() in RASTER_EXT and re.search(
                r"architect|diagram|flow|block|schema|pipeline|overview", p.name, re.I):
            raster_diagrams.append(ref)

        scripts = find_generating_scripts(repo, p.name)
        rec["generating_scripts"] = scripts
        if not scripts:
            no_script.append(ref)

        rec["sha256_12"] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        info[ref] = rec

    rep.stats["includegraphics uses"] = len(uses)
    rep.stats["distinct source files found"] = len(info)

    if missing:
        rep.add("assets", "BLOCKER",
                f"{len(missing)} referenced graphic(s) could not be resolved on disk",
                "\n".join(missing))
    if distorted:
        rep.add("assets", "MAJOR",
                f"{len(distorted)} graphic(s) set both width and height without "
                "keepaspectratio, which stretches the image",
                "\n".join(distorted))
    if raster_diagrams:
        rep.add("assets", "MINOR",
                f"{len(raster_diagrams)} diagram(s) supplied as raster rather than vector",
                "\n".join(raster_diagrams) +
                "\nLine art stays sharp at any zoom as PDF or EPS. Most journals prefer it.")
    if no_script:
        rep.add("assets", "MAJOR",
                f"{len(no_script)} figure(s) have no generating script in the repository",
                "\n".join(no_script) +
                "\nEvery figure in a paper should be reproducible from code. Confirm the "
                "provenance of each, and inspect it visually for fabricated content.")

    # Identical bytes reused under two names.
    by_hash = defaultdict(list)
    for ref, rec in info.items():
        by_hash[rec["sha256_12"]].append(ref)
    dupes = {h: v for h, v in by_hash.items() if len(v) > 1}
    if dupes:
        rep.add("assets", "MAJOR",
                f"{len(dupes)} image(s) reused under more than one name",
                "\n".join(", ".join(v) for v in dupes.values()))

    return info


def check_orphan_figures(rep: Report, tex: Path, info: dict) -> None:
    """Figure files present on disk that the paper never uses."""
    tex_dir = tex.parent
    used = {Path(rec["file"]).resolve() for rec in info.values()}
    searched = []
    for d in (tex_dir / "images", tex_dir / "figures", tex_dir):
        if d.exists():
            searched.append(d)
    on_disk = set()
    for d in searched:
        for ext in RASTER_EXT | VECTOR_EXT:
            for f in d.glob(f"*{ext}"):
                if f.name.startswith("paper"):
                    continue  # the compiled output itself
                on_disk.add(f.resolve())
    orphans = sorted(p.name for p in on_disk - used)
    rep.stats["figure files on disk"] = len(on_disk)
    rep.stats["figure files unused"] = len(orphans)
    if orphans:
        rep.add("assets", "INFO",
                f"{len(orphans)} figure file(s) on disk are never used",
                "\n".join(orphans) + "\nRemove them or confirm they are intentional.")


def check_embedded(rep: Report, extract_dir: Path) -> None:
    manifest_path = extract_dir / "manifest.json"
    if not manifest_path.exists():
        rep.add("assets", "INFO",
                "no extract manifest found; run extract_artifacts.py first",
                str(manifest_path))
        return
    man = json.loads(manifest_path.read_text())

    rows = [r for r in man.get("embedded_images", []) if r.get("type") == "image"]
    rep.stats["rasters embedded in PDF"] = len(rows)

    low: list[tuple[str, float]] = []
    worst = None
    for r in rows:
        try:
            ppi = min(float(r.get("x-ppi", 0)), float(r.get("y-ppi", 0)))
        except ValueError:
            continue
        label = f"page {r.get('page')} obj {r.get('object')} " \
                f"{r.get('width')}x{r.get('height')}"
        low.append((label, ppi))
        if worst is None or ppi < worst[1]:
            worst = (label, ppi)

    if worst:
        rep.stats["lowest effective ppi in PDF"] = f"{worst[1]:.0f} ({worst[0]})"

    for label, ppi in low:
        if ppi < PPI_BLOCKER:
            rep.add("assets", "BLOCKER",
                    f"image at {ppi:.0f} ppi will look blurred in print", label)
        elif ppi < PPI_MAJOR:
            rep.add("assets", "MAJOR",
                    f"image at {ppi:.0f} ppi is below the usual 300 ppi print floor", label)
        elif ppi < PPI_MINOR:
            rep.add("assets", "INFO",
                    f"image at {ppi:.0f} ppi; fine for raster art, below the "
                    "600 ppi some journals ask for line art", label)

    # Aspect distortion visible in the PDF: compare embedded pixel aspect with
    # the aspect of the area it occupies. pdfimages gives separate x and y ppi,
    # and unequal values mean the image was stretched.
    for r in rows:
        try:
            xp, yp = float(r.get("x-ppi", 0)), float(r.get("y-ppi", 0))
        except ValueError:
            continue
        if xp and yp and abs(xp - yp) / max(xp, yp) > 0.02:
            rep.add("assets", "MAJOR",
                    f"image stretched: {xp:.0f} ppi horizontally vs {yp:.0f} vertically",
                    f"page {r.get('page')} obj {r.get('object')}")

    # Fonts must be embedded or the PDF will not render identically elsewhere.
    if man.get("fonts_not_embedded"):
        rep.add("assets", "BLOCKER",
                f"{len(man['fonts_not_embedded'])} font(s) not embedded in the PDF",
                ", ".join(man["fonts_not_embedded"]))

    # Page geometry.
    size = man.get("pdfinfo", {}).get("Page size", "")
    rep.stats["page size"] = size or "unknown"
    if size and "A4" not in size and "letter" not in size.lower():
        rep.add("assets", "MINOR",
                f"unusual page size: {size}", "Confirm it matches the venue template.")

    rep.stats["pages"] = man.get("n_pages", "?")
    total_mb = man.get("page_image_total_bytes", 0) / 1e6
    rep.stats["page rasters for inspection"] = \
        f"{len(man.get('page_images', []))} ({total_mb:.1f} MB)"


def check_visual_todo(rep: Report, extract_dir: Path, info: dict) -> None:
    """
    Emit the explicit list of images the agent has to open and look at. A script
    cannot tell whether an axis label is unreadable, whether a diagram contains
    garbled text, or whether a panel was fabricated.
    """
    pages = sorted((extract_dir / "pages").glob("page-*.png"))
    if pages:
        rep.add("visual-review", "INFO",
                f"{len(pages)} page raster(s) must be opened and inspected",
                "\n".join(str(p) for p in pages) +
                "\n\nPer page, look for: text running into the margin, tables "
                "overflowing, figures too small to read, orphaned captions, large "
                "white gaps, inconsistent fonts, and any figure whose content does "
                "not match its caption.")
    if info:
        rep.add("visual-review", "INFO",
                f"{len(info)} source figure(s) must be opened and inspected",
                "\n".join(rec["file"] for rec in info.values()) +
                "\n\nPer figure, look for: garbled or nonsensical text, axis labels "
                "without units, unreadable legends, mismatched fonts between panels, "
                "decorative gradients or 3D effects, invented data points, and any "
                "sign the image was drawn by a model rather than plotted from data.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--extracted", required=True,
                    help="output directory from extract_artifacts.py")
    ap.add_argument("--repo", default=".", help="repository root for the script search")
    ap.add_argument("--json")
    args = ap.parse_args()

    tex = Path(args.tex).resolve()
    repo = Path(args.repo).resolve()
    extract_dir = Path(args.extracted).resolve()

    raw = read_tex(tex)
    uses = parse_includegraphics(raw)

    rep = Report(f"ASSET CHECK  {tex.name}")
    if not HAVE_PIL:
        rep.add("assets", "INFO", "Pillow not installed; source pixel checks skipped")

    info = check_source_files(rep, tex, uses, repo)
    check_orphan_figures(rep, tex, info)
    check_embedded(rep, extract_dir)
    check_visual_todo(rep, extract_dir, info)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"report": json.loads(rep.to_json()), "figures": info}, indent=2))
        return emit(rep, None)
    return emit(rep, None)


if __name__ == "__main__":
    sys.exit(main())
