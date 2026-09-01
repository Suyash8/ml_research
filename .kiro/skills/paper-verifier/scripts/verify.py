#!/usr/bin/env python3
"""
Run the whole mechanical verification pass and write one consolidated report.

    python3 verify.py <paper.tex> [--outdir DIR] [--artifacts D1 D2] [--repo R] [--dpi 110]

Stages:
    0  extract_artifacts.py   PDF -> text, page rasters, embedded image inventory
    1  check_text.py          prose, structure, overstatement, AI tells
    2  check_assets.py        figures, resolution, provenance
    3  check_numbers.py       traceability against result files
    4  check_bib.py           bibliography structure, verification queue

What this does NOT do, and cannot: confirm a reference exists, read the page
rasters, judge whether a figure matches its caption, or decide whether a claim
is honest. Those need the agent. The report ends with that queue spelled out,
so a run that stops here is visibly incomplete rather than silently so.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_stage(name: str, cmd: list[str], log_dir: Path) -> dict:
    print(f"\n>>> {name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else "")
    (log_dir / f"{name}.txt").write_text(out)
    print(out)
    return {"name": name, "returncode": proc.returncode,
            "cmd": " ".join(cmd), "log": str(log_dir / f"{name}.txt")}


def load_counts(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if "report" in data:
        data = data["report"]
    return data.get("counts", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--outdir", help="default: <tex dir>/verification")
    ap.add_argument("--artifacts", nargs="*", default=["results", "data"],
                    help="directories holding result files")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--pdf", help="default: same stem as the tex file")
    args = ap.parse_args()

    tex = Path(args.tex).resolve()
    if not tex.exists():
        sys.exit(f"no such tex file: {tex}")
    pdf = Path(args.pdf).resolve() if args.pdf else tex.with_suffix(".pdf")
    out = Path(args.outdir).resolve() if args.outdir else tex.parent / "verification"
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    stages = []

    if pdf.exists():
        stages.append(run_stage("0-extract", [
            sys.executable, str(HERE / "extract_artifacts.py"),
            str(pdf), str(out), "--dpi", str(args.dpi)], logs))
    else:
        print(f"\n!!! no PDF at {pdf}")
        print("!!! build the paper first; the visual and layout checks cannot run "
              "without it, and half of what a reviewer sees lives only in the PDF.")

    stages.append(run_stage("1-text", [
        sys.executable, str(HERE / "check_text.py"), str(tex),
        "--json", str(out / "text.json")], logs))

    if pdf.exists():
        stages.append(run_stage("2-assets", [
            sys.executable, str(HERE / "check_assets.py"), str(tex),
            "--extracted", str(out), "--repo", str(Path(args.repo).resolve()),
            "--json", str(out / "assets.json")], logs))

    stages.append(run_stage("3-numbers", [
        sys.executable, str(HERE / "check_numbers.py"), str(tex),
        "--artifacts", *args.artifacts,
        "--json", str(out / "numbers.json")], logs))

    stages.append(run_stage("4-bib", [
        sys.executable, str(HERE / "check_bib.py"), str(tex),
        "--json", str(out / "bib.json"),
        "--queue", str(out / "reference_queue.txt")], logs))

    # ---- consolidate ----------------------------------------------------
    totals = {"BLOCKER": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}
    per_stage = {}
    for jf, label in (("text.json", "text"), ("assets.json", "assets"),
                      ("numbers.json", "numbers"), ("bib.json", "bib")):
        c = load_counts(out / jf)
        per_stage[label] = c
        for k in totals:
            totals[k] += c.get(k, 0)

    n_pages = 0
    manifest = out / "manifest.json"
    if manifest.exists():
        n_pages = json.loads(manifest.read_text()).get("n_pages", 0)
    n_refs = 0
    if (out / "bib.json").exists():
        try:
            n_refs = len(json.loads((out / "bib.json").read_text()).get("entries", []))
        except json.JSONDecodeError:
            pass

    summary = {
        "paper": str(tex),
        "pdf": str(pdf) if pdf.exists() else None,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "totals": totals,
        "per_stage": per_stage,
        "stages": stages,
        "outdir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    bar = "=" * 78
    lines = [
        "", bar,
        "  MECHANICAL PASS COMPLETE",
        bar,
        f"  paper   : {tex}",
        f"  pdf     : {pdf if pdf.exists() else 'MISSING - visual checks skipped'}",
        f"  outputs : {out}",
        "",
        f"  BLOCKER {totals['BLOCKER']:>4}   fix before sending",
        f"  MAJOR   {totals['MAJOR']:>4}   a reviewer will raise it",
        f"  MINOR   {totals['MINOR']:>4}   polish",
        f"  INFO    {totals['INFO']:>4}   context to judge",
        "",
        "  per stage: " + ", ".join(
            f"{k}={v.get('BLOCKER', 0)}B/{v.get('MAJOR', 0)}M"
            for k, v in per_stage.items()),
        "",
        bar,
        "  NOT DONE YET. THREE THINGS ONLY THE AGENT CAN DO:",
        bar,
        f"  1. Open and inspect all {n_pages} page rasters under {out}/pages/",
        "     A script cannot see a table running off the page, a figure too small",
        "     to read, an orphaned caption, or a figure that contradicts its caption.",
        "",
        f"  2. Web-verify all {n_refs} references using {out}/reference_queue.txt",
        "     Check the author list name by name. A fabricated reference passes every",
        "     structural check, because the title and journal look right.",
        "",
        "  3. Read the untraced numbers and the overstatement findings, and judge",
        "     each one. Confirm every claim against what the results actually support.",
        bar,
        "",
    ]
    text = "\n".join(lines)
    print(text)
    (out / "SUMMARY.txt").write_text(text)

    return 1 if (totals["BLOCKER"] or totals["MAJOR"]) else 0


if __name__ == "__main__":
    sys.exit(main())
