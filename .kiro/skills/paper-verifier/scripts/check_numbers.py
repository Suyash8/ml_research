#!/usr/bin/env python3
"""
Number traceability. Every figure quoted in the paper must exist in a result
artifact, and the abstract must not quote a number the results do not support.

    python3 check_numbers.py <paper.tex> --artifacts results data/analysis_reports [--json out.json]

How matching works: artifact values are indexed at every rounding from 0 to 5
decimal places, and also at x100 and /100 so that 0.882 in a file matches
"88.2%" in the text. A paper number counts as traced when some artifact value
rounds to it at the precision the paper used.

An unmatched number is not automatically wrong. "four cohorts" and "0.5 means
random ordering" are unmatched and correct. The point is to produce a short
list a human can actually check, instead of a vague instruction to verify the
numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    Report, emit, read_tex, strip_comments, body_only, to_prose,
    section_text, numbers_in, canon_number, split_sentences,
)

ARTIFACT_SUFFIXES = {".json", ".csv", ".tsv", ".md", ".txt", ".yaml", ".yml"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "pages"}

# Numbers that are structural rather than empirical, so absence from the
# artifacts means nothing.
BENIGN = {
    "0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0",
    "10.0", "0.5", "100.0", "1000.0",
}

NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w])")


def index_artifacts(roots: list[Path]) -> tuple[dict[str, set[str]], int, int]:
    """value-string -> set of files containing it, at several roundings."""
    index: dict[str, set[str]] = defaultdict(set)
    n_files = 0
    n_values = 0
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in ARTIFACT_SUFFIXES:
                continue
            if set(f.parts) & SKIP_DIRS:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            n_files += 1
            name = str(f)
            for m in NUM_RE.finditer(text):
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                n_values += 1
                for scale in (1.0, 100.0, 0.01):
                    v = val * scale
                    if abs(v) > 1e12:
                        continue
                    for nd in range(0, 6):
                        index[f"{round(v, nd):.{nd}f}".rstrip()].add(name)
                    index[repr(v)].add(name)
    return index, n_files, n_values


def lookup(index: dict[str, set[str]], token: str) -> set[str]:
    """Files containing a value that rounds to this token as written."""
    t = token.replace("{,}", "").replace(",", "")
    m = re.match(r"^([\d.]+)\s*\\times\s*10\^\{(-?\d+)\}$", t)
    if m:
        try:
            t = repr(float(m.group(1)) * (10 ** int(m.group(2))))
        except ValueError:
            return set()
    try:
        val = float(t)
    except ValueError:
        return set()
    nd = len(t.split(".")[1]) if "." in t and "e" not in t.lower() else 0
    keys = {f"{round(val, nd):.{nd}f}".rstrip(), repr(val)}
    hits: set[str] = set()
    for k in keys:
        hits |= index.get(k, set())
    return hits


def check_traceability(rep: Report, raw: str, index: dict[str, set[str]],
                       out: dict) -> None:
    body = to_prose(raw)
    found = numbers_in(body)

    traced, untraced = [], []
    for tok, ctx in found:
        canon = canon_number(tok)
        if canon in BENIGN:
            continue
        hits = lookup(index, tok)
        if hits:
            traced.append({"value": tok, "context": ctx,
                           "sources": sorted(Path(h).name for h in hits)[:4]})
        else:
            untraced.append({"value": tok, "context": ctx})

    rep.stats["numeric literals in prose"] = len(found)
    rep.stats["traced to an artifact"] = len(traced)
    rep.stats["not traced"] = len(untraced)
    out["traced"] = traced
    out["untraced"] = untraced

    if untraced:
        rep.add("numbers", "MAJOR",
                f"{len(untraced)} numeric literal(s) not found in any artifact",
                "\n".join(f"{u['value']:>12}  {u['context'][:130]}"
                          for u in untraced[:40]) +
                "\n\nMany will be structural (counts of families, thresholds explained "
                "in words). Confirm each one individually. Delete anything that cannot "
                "be traced rather than rounding or hedging it.")


def check_abstract_drift(rep: Report, raw: str) -> None:
    """
    Numbers in the abstract and conclusion must appear in the body. Rounding a
    metric up on the way into the abstract is a classic reviewer catch.
    """
    text = strip_comments(raw)
    doc = body_only(text)

    m = re.search(r"\\section\{", doc)
    abstract = doc[: m.start()] if m else ""
    concl = section_text(text, r"conclusion")
    # The comparison body is Method, Results and Discussion. Method matters:
    # a design constant such as the number of simulation draws is introduced
    # there and legitimately restated in the abstract.
    body_ref = to_prose("\n".join([
        section_text(text, r"method|approach|materials|proposed"),
        section_text(text, r"results|experiments"),
        section_text(text, r"discussion"),
    ]))

    body_vals = {canon_number(t) for t, _ in numbers_in(body_ref)}
    body_vals.discard(None)

    for label, chunk in (("abstract", to_prose(abstract)), ("conclusion", to_prose(concl))):
        if not chunk:
            continue
        missing = []
        for tok, ctx in numbers_in(chunk):
            c = canon_number(tok)
            if c is None or c in BENIGN:
                continue
            if c in body_vals:
                continue
            # Allow a value that the body reports at higher precision.
            near = any(
                b is not None and abs(float(b) - float(c)) < 5 * 10 ** -(
                    len(tok.split(".")[1]) if "." in tok else 0) / 2
                for b in body_vals
            )
            if not near:
                missing.append(f"{tok}  {ctx[:120]}")
        rep.stats[f"{label} numbers absent from the body"] = len(missing)
        if missing:
            rep.add("numbers", "BLOCKER",
                    f"{len(missing)} number(s) in the {label} do not appear in "
                    "Method, Results or Discussion",
                    "\n".join(missing[:20]) +
                    f"\n\nEither the {label} overstates the result or the body omits it.")


def check_internal_arithmetic(rep: Report, raw: str) -> None:
    """Sums and percentages the paper states about itself must add up."""
    prose = to_prose(raw)

    # Split sizes should sum to the cohort size.
    tot = re.search(r"(?:N|cohort|patients?)[^.]{0,40}?(\d[\d,{}]{2,})\s*patients?", prose)
    splits = re.findall(r"(?:[Tt]raining|[Cc]alibration|[Tt]est(?:ing)?)[^.]{0,30}?"
                        r"(\d[\d,{}]{2,})", prose)
    if tot and len(splits) >= 3:
        try:
            total = int(re.sub(r"[^\d]", "", tot.group(1)))
            parts = [int(re.sub(r"[^\d]", "", s)) for s in splits[:3]]
            if sum(parts) != total:
                rep.add("numbers", "BLOCKER",
                        f"split sizes {parts} sum to {sum(parts)}, "
                        f"but the cohort is stated as {total}")
            else:
                rep.stats["split sizes sum to cohort"] = f"{parts} = {total}"
        except ValueError:
            pass

    # "dropped X of Y (Z%)" must be arithmetically true.
    for m in re.finditer(
        r"(\d[\d,{}]*)\s+of\s+(\d[\d,{}]*)[^.]{0,60}?(\d+(?:\.\d+)?)\s*\\?%", prose
    ):
        try:
            a = int(re.sub(r"[^\d]", "", m.group(1)))
            b = int(re.sub(r"[^\d]", "", m.group(2)))
            pct = float(m.group(3))
        except ValueError:
            continue
        if b == 0:
            continue
        actual = 100.0 * a / b
        if abs(actual - pct) > 0.15:
            rep.add("numbers", "BLOCKER",
                    f"{a} of {b} is {actual:.1f}%, but {pct}% is stated",
                    m.group(0)[:150])

    # A metric described as a range must match the values quoted elsewhere.
    for m in re.finditer(
        r"(?:from|between)\s+(\d+\.\d+)\s+(?:to|and)\s+(\d+\.\d+)", prose
    ):
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            rep.add("numbers", "MAJOR",
                    f"range stated backwards: {lo} to {hi}", m.group(0))


def check_number_hygiene(rep: Report, raw: str) -> None:
    prose = to_prose(raw)

    # Bare p = 0.000 or p = 0 is meaningless.
    for m in re.finditer(r"\bp\s*[=<]\s*0(?:\.0+)?\b", prose):
        rep.add("numbers", "MAJOR",
                "p reported as zero; report a bound such as p < 0.001 instead",
                m.group(0))

    # A correlation or probability outside its valid range.
    for label, pat, lo, hi in [
        ("correlation", r"(?:rho|r)\s*=\s*(-?\d+\.\d+)", -1.0, 1.0),
        ("probability", r"(?:probability|AUC|area under the curve|"
                       r"concordance index|Brier)[^.]{0,40}?(\d+\.\d+)", 0.0, 1.0),
    ]:
        for m in re.finditer(pat, prose, re.I):
            v = float(m.group(1))
            if not (lo <= v <= hi):
                rep.add("numbers", "BLOCKER",
                        f"{label} value {v} is outside [{lo}, {hi}]", m.group(0))

    # A mean without a spread is hard to interpret.
    means = len(re.findall(r"\bmean\b", prose, re.I))
    spreads = len(re.findall(r"standard deviation|s\.d\.|interquartile|"
                             r"\\pm|confidence interval", prose, re.I))
    rep.stats["mentions of a mean"] = means
    rep.stats["mentions of a spread"] = spreads
    if means > 2 and spreads == 0:
        rep.add("numbers", "MINOR",
                f"{means} mean(s) reported and no measure of spread anywhere")

    # Sentences with three or more numbers are hard to read aloud.
    dense = [s for s in split_sentences(prose)
             if len(re.findall(r"\d+\.\d+", s)) >= 4]
    rep.stats["sentences with 4+ decimal numbers"] = len(dense)
    if dense:
        rep.add("numbers", "MINOR",
                f"{len(dense)} sentence(s) carry four or more decimal numbers; "
                "move them into a table",
                "\n".join(s[:170] for s in dense[:8]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--artifacts", nargs="+", required=True,
                    help="directories holding result files to match against")
    ap.add_argument("--json")
    args = ap.parse_args()

    tex = Path(args.tex).resolve()
    raw = read_tex(tex)
    roots = [Path(a).resolve() for a in args.artifacts]

    index, n_files, n_values = index_artifacts(roots)

    rep = Report(f"NUMBER CHECK  {tex.name}")
    rep.stats["artifact files scanned"] = n_files
    rep.stats["artifact values indexed"] = n_values

    out: dict = {}
    check_traceability(rep, raw, index, out)
    check_abstract_drift(rep, raw)
    check_internal_arithmetic(rep, raw)
    check_number_hygiene(rep, raw)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"report": json.loads(rep.to_json()), **out}, indent=2))
    return emit(rep, None)


if __name__ == "__main__":
    sys.exit(main())
