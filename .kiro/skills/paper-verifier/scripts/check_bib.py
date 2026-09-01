#!/usr/bin/env python3
"""
Bibliography verification, structural pass.

    python3 check_bib.py <paper.tex> [--bib references.bib] [--json out.json]

This finds what a script can find: missing required fields, absent identifiers,
duplicate entries, author lists that look padded, years that cannot be right,
and keys that disagree with the entry they name.

It cannot confirm that a paper exists or that its author list is correct. That
requires a web lookup per entry, and the script therefore ends by printing a
ready-made verification queue with a suggested search string for each entry.
Fabricated references almost always pass a structural check: the title and the
journal look right while two of the five authors do not exist. Skipping the web
pass defeats the purpose of running this at all.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import Report, emit, read_tex, strip_comments  # noqa: E402

REQUIRED = {
    "article": ["author", "title", "journal", "year"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "incollection": ["author", "title", "booktitle", "year"],
    "book": ["title", "year", "publisher"],
    "inbook": ["title", "year", "publisher"],
    "phdthesis": ["author", "title", "school", "year"],
    "mastersthesis": ["author", "title", "school", "year"],
    "techreport": ["author", "title", "institution", "year"],
    "misc": ["title"],
    "unpublished": ["author", "title", "note"],
}

# Phrases in an author field that mean the list was not actually checked.
PADDED_AUTHOR = [
    r"\band others\b", r"\bet al\.?", r"\bothers\b", r"\bunknown\b",
    r"\banonymous\b", r"\bvarious\b", r"\bTBD\b", r"\bXXX\b",
]


def parse_bib(text: str) -> list[dict]:
    """Minimal BibTeX parser: enough for auditing, not for rendering."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype = m.group(1).lower()
        key = m.group(2)
        # Walk braces to the end of the entry.
        i = text.index("{", m.start())
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[i + 1: j]
        fields: dict[str, str] = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            name = fm.group(1).lower()
            if name == key.lower():
                continue
            rest = body[fm.end():].lstrip()
            if rest.startswith("{"):
                d, k = 0, 0
                while k < len(rest):
                    if rest[k] == "{":
                        d += 1
                    elif rest[k] == "}":
                        d -= 1
                        if d == 0:
                            break
                    k += 1
                val = rest[1:k]
            elif rest.startswith('"'):
                k = rest.index('"', 1)
                val = rest[1:k]
            else:
                val = re.split(r"[,\n}]", rest, 1)[0]
            fields[name] = re.sub(r"\s+", " ", val).strip()
        entries.append({"type": etype, "key": key, "fields": fields,
                        "raw_offset": m.start()})
    return entries


def author_list(field: str) -> list[str]:
    return [a.strip() for a in re.split(r"\s+and\s+", field) if a.strip()]


def check_structure(rep: Report, entries: list[dict]) -> None:
    this_year = datetime.date.today().year
    no_id, padded, thin, bad_year, single_name = [], [], [], [], []

    for e in entries:
        f, key, etype = e["fields"], e["key"], e["type"]

        for req in REQUIRED.get(etype, ["author", "title", "year"]):
            if req not in f or not f[req]:
                rep.add("bib-structure", "MAJOR",
                        f"{key}: missing required field '{req}' for @{etype}")

        if not (f.get("doi") or f.get("url") or f.get("isbn") or
                f.get("eprint") or f.get("archiveprefix")):
            no_id.append(key)

        auth = f.get("author", "") or f.get("editor", "")
        if any(re.search(p, auth, re.I) for p in PADDED_AUTHOR):
            padded.append(f"{key}: {auth[:110]}")
        for a in author_list(auth):
            if a and "," not in a and " " not in a and not a.startswith("{"):
                single_name.append(f"{key}: {a!r}")

        if etype == "article":
            if not f.get("volume") and not f.get("doi"):
                thin.append(f"{key}: no volume and no DOI")
            if not f.get("pages") and not f.get("number") and not f.get("doi"):
                thin.append(f"{key}: no pages, number or DOI")

        y = f.get("year", "")
        if y:
            ym = re.search(r"\d{4}", y)
            if not ym:
                bad_year.append(f"{key}: year {y!r} is not a year")
            else:
                yi = int(ym.group(0))
                if yi > this_year + 1 or yi < 1800:
                    bad_year.append(f"{key}: year {yi} is not plausible")

        # A key whose embedded year disagrees with the year field usually means
        # the entry was edited but the key was not, which then renders the wrong
        # author-year in styles that show one.
        km = re.search(r"(?:19|20)\d{2}", key)
        ym2 = re.search(r"\d{4}", y) if y else None
        if km and ym2 and km.group(0) != ym2.group(0):
            rep.add("bib-structure", "MAJOR",
                    f"{key}: key says {km.group(0)} but the year field says {ym2.group(0)}",
                    "Rename the key, or correct the year. One of the two is wrong.")

        # A key whose embedded surname disagrees with the first author. Corporate
        # authors are wrapped in braces and have no surname, so they are skipped.
        authors = author_list(auth)
        first = authors[0] if authors else ""
        if first.startswith("{"):
            continue
        surname = (first.split(",")[0] if "," in first
                   else (first.split()[-1] if first.split() else ""))
        # Compare with spaces, hyphens and apostrophes removed, so that the key
        # "vanhouwelingen" matches the surname "van Houwelingen".
        norm = lambda s: re.sub(r"[^a-z]", "", s.lower())  # noqa: E731
        key_word = re.match(r"([a-zA-Z]+)", key)
        if surname and key_word and len(key_word.group(1)) > 3:
            kw, sn = norm(key_word.group(1)), norm(surname)
            if sn and kw not in sn and sn not in kw:
                rep.add("bib-structure", "MINOR",
                        f"{key}: key starts {key_word.group(1)!r} but the first "
                        f"author is {first!r}",
                        "Rename the key, or check that the entry body is the right paper.")

    rep.stats["entries"] = len(entries)
    rep.stats["entries with no DOI, URL or ISBN"] = len(no_id)

    if no_id:
        rep.add("bib-structure", "MAJOR",
                f"{len(no_id)} entry/entries carry no resolvable identifier",
                ", ".join(no_id) +
                "\nWithout an identifier the entry cannot be checked mechanically, "
                "and an invented reference will pass unnoticed.")
    if padded:
        rep.add("bib-structure", "BLOCKER",
                f"{len(padded)} author list(s) contain 'and others' or 'et al.'",
                "\n".join(padded) +
                "\nA truncated author list means the list was never verified. "
                "Fabricated co-authors hide here.")
    if single_name:
        rep.add("bib-structure", "MAJOR",
                f"{len(single_name)} author name(s) have no given name or comma",
                "\n".join(single_name[:20]))
    if thin:
        rep.add("bib-structure", "MINOR",
                f"{len(thin)} article entry/entries are missing locator fields",
                "\n".join(thin[:20]))
    if bad_year:
        rep.add("bib-structure", "BLOCKER",
                f"{len(bad_year)} implausible year value(s)", "\n".join(bad_year))


def check_duplicates(rep: Report, entries: list[dict]) -> None:
    by_title = defaultdict(list)
    by_doi = defaultdict(list)
    by_key = defaultdict(list)
    for e in entries:
        t = re.sub(r"[^a-z0-9]", "", e["fields"].get("title", "").lower())
        if t:
            by_title[t].append(e["key"])
        d = e["fields"].get("doi", "").lower().strip()
        if d:
            by_doi[d].append(e["key"])
        by_key[e["key"].lower()].append(e["key"])

    for label, mapping, sev in (("title", by_title, "MAJOR"),
                                ("DOI", by_doi, "MAJOR"),
                                ("key", by_key, "BLOCKER")):
        dupes = {k: v for k, v in mapping.items() if len(v) > 1}
        if dupes:
            rep.add("bib-duplicates", sev,
                    f"{len(dupes)} duplicated {label}(s)",
                    "\n".join(f"{', '.join(v)}" for v in dupes.values()))


def check_against_tex(rep: Report, raw: str, entries: list[dict]) -> None:
    body = strip_comments(raw)
    defined = {e["key"] for e in entries}
    used: set[str] = set()
    for group in re.findall(r"\\cite[a-z]*\{([^}]*)\}", body):
        used.update(k.strip() for k in group.split(",") if k.strip())

    missing = sorted(used - defined)
    unused = sorted(defined - used)
    rep.stats["keys cited in the tex"] = len(used)
    rep.stats["keys cited but undefined"] = len(missing)
    rep.stats["entries never cited"] = len(unused)

    if missing:
        rep.add("bib-usage", "BLOCKER",
                f"{len(missing)} key(s) cited with no matching entry",
                ", ".join(missing))
    if unused:
        rep.add("bib-usage", "MAJOR",
                f"{len(unused)} entry/entries are never cited",
                ", ".join(unused) +
                "\nDelete them, or cite them where they belong. An uncited entry is "
                "often a leftover from a claim that was removed.")


def check_balance(rep: Report, entries: list[dict]) -> None:
    years = []
    for e in entries:
        m = re.search(r"\d{4}", e["fields"].get("year", ""))
        if m:
            years.append(int(m.group(0)))
    if not years:
        return
    this_year = datetime.date.today().year
    recent = sum(1 for y in years if y >= this_year - 5)
    rep.stats["median reference year"] = sorted(years)[len(years) // 2]
    rep.stats[f"references from the last 5 years"] = f"{recent}/{len(years)}"
    if recent / len(years) < 0.2:
        rep.add("bib-balance", "MINOR",
                f"only {recent} of {len(years)} references are from the last five years",
                "A reviewer will read this as unfamiliarity with current work. "
                "Foundational citations are expected; the recent literature still "
                "needs covering.")

    types = defaultdict(int)
    for e in entries:
        types[e["type"]] += 1
    rep.stats["entry types"] = dict(types)
    preprints = sum(1 for e in entries
                    if re.search(r"arxiv|preprint|biorxiv|medrxiv",
                                 json.dumps(e["fields"]), re.I))
    rep.stats["preprints"] = preprints
    if preprints and preprints / len(entries) > 0.25:
        rep.add("bib-balance", "MINOR",
                f"{preprints} of {len(entries)} references are preprints",
                "Prefer the peer-reviewed version where one exists.")


def emit_verification_queue(entries: list[dict], out_path: Path | None) -> str:
    lines = []
    for e in entries:
        f = e["fields"]
        auth = author_list(f.get("author", "") or f.get("editor", ""))
        first = auth[0].split(",")[0] if auth else "?"
        query = " ".join(x for x in [
            first,
            f'"{f.get("title", "")[:90]}"',
            f.get("journal", "") or f.get("booktitle", "") or f.get("publisher", ""),
            f.get("year", ""),
        ] if x).strip()
        lines.append(
            f"[ ] {e['key']}\n"
            f"      type      : @{e['type']}\n"
            f"      authors   : {len(auth)} listed -> {'; '.join(auth[:12])}\n"
            f"      title     : {f.get('title', '(none)')}\n"
            f"      venue     : {f.get('journal') or f.get('booktitle') or f.get('publisher') or '(none)'}\n"
            f"      year/vol/pp: {f.get('year', '?')} / {f.get('volume', '-')} / {f.get('pages', '-')}\n"
            f"      id        : {f.get('doi') or f.get('url') or f.get('isbn') or 'NONE'}\n"
            f"      search    : {query}\n"
        )
    text = ("REFERENCE VERIFICATION QUEUE\n"
            "Confirm each field against the publisher record or PubMed. Check the\n"
            "author list name by name; that is where fabrication survives.\n"
            "Delete anything that cannot be confirmed, then remove or rewrite the\n"
            "sentence that depended on it.\n\n" + "\n".join(lines))
    if out_path:
        out_path.write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--bib")
    ap.add_argument("--json")
    ap.add_argument("--queue", help="write the verification queue to this path")
    args = ap.parse_args()

    tex = Path(args.tex).resolve()
    raw = read_tex(tex)

    if args.bib:
        bib = Path(args.bib).resolve()
    else:
        m = re.search(r"\\bibliography\{([^}]*)\}", raw)
        name = (m.group(1).split(",")[0].strip() if m else "references")
        bib = (tex.parent / name)
        if not bib.suffix:
            bib = bib.with_suffix(".bib")
        bib = bib.resolve()

    if not bib.exists():
        print(f"error: bibliography not found: {bib}")
        return 2

    entries = parse_bib(bib.read_text(encoding="utf-8", errors="replace"))

    rep = Report(f"BIBLIOGRAPHY CHECK  {bib.name}")
    check_structure(rep, entries)
    check_duplicates(rep, entries)
    check_against_tex(rep, raw, entries)
    check_balance(rep, entries)

    rep.add("bib-verify", "INFO",
            f"{len(entries)} entry/entries still need a web lookup",
            "A structural pass cannot detect an invented reference. Work through "
            "the queue below, one search per entry, and check the author list "
            "name by name.")

    queue_path = Path(args.queue) if args.queue else None
    queue = emit_verification_queue(entries, queue_path)

    status = emit(rep, None)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"report": json.loads(rep.to_json()),
             "entries": entries}, indent=2))
    if queue_path:
        print(f"\n  verification queue written to {queue_path}")
    else:
        print("\n" + queue)
    return status


if __name__ == "__main__":
    sys.exit(main())
