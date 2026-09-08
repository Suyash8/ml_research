#!/usr/bin/env python3
"""Shared helpers for the paper-verifier checks."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Severity meanings, used consistently across every check:
#   BLOCKER  factually wrong, unverifiable, or fails a hard rule. Fix before sending.
#   MAJOR    a reviewer will very likely raise it. Fix before sending.
#   MINOR    polish. Fix if time allows.
#   INFO     context for the agent to judge. Not a defect by itself.
SEVERITIES = ("BLOCKER", "MAJOR", "MINOR", "INFO")


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    detail: str = ""
    location: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"bad severity {self.severity!r}")


@dataclass
class Report:
    name: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, check: str, severity: str, message: str,
            detail: str = "", location: str = "") -> None:
        self.findings.append(Finding(check, severity, message, detail, location))

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "stats": self.stats,
                "counts": {s: self.count(s) for s in SEVERITIES},
                "findings": [asdict(f) for f in self.findings],
            },
            indent=2,
        )

    def print_human(self) -> None:
        print(f"\n{'=' * 78}")
        print(f"  {self.name}")
        print(f"{'=' * 78}")
        if self.stats:
            for k, v in self.stats.items():
                print(f"  {k:.<44} {v}")
            print()
        if not self.findings:
            print("  no findings")
            return
        for sev in SEVERITIES:
            group = [f for f in self.findings if f.severity == sev]
            if not group:
                continue
            print(f"  --- {sev} ({len(group)}) ---")
            for f in group:
                loc = f" [{f.location}]" if f.location else ""
                print(f"  * {f.check}: {f.message}{loc}")
                if f.detail:
                    for line in f.detail.rstrip().splitlines():
                        print(f"      {line}")
            print()


# ---------------------------------------------------------------------------
# LaTeX handling
# ---------------------------------------------------------------------------

FLOAT_ENVS = ("table", "figure", "tabular", "subfigure", "wrapfigure", "longtable")
MATH_ENVS = ("equation", "align", "gather", "displaymath", "multline", "eqnarray")


def read_tex(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def body_only(text: str) -> str:
    """Everything after \\begin{document}, minus the title/author block."""
    if r"\begin{document}" in text:
        text = text.split(r"\begin{document}", 1)[1]
    m = re.search(r"\{\s*\\bfseries\s+ABSTRACT|\\begin\{abstract\}|\\maketitle", text)
    if m:
        text = text[m.end():]
    return text


def trim_backmatter(text: str) -> str:
    """
    Cut the declarations and bibliography tail. Acknowledgments, funding and
    data-availability blocks are boilerplate, often supplied by the publisher,
    and measuring their prose alongside the argument distorts every statistic.
    """
    m = re.search(
        r"\\section\*\{\s*(?:Acknowledg|Disclosure|Funding|Declaration|"
        r"Data availability|Conflict|Competing|Author contribution)",
        text, re.I,
    )
    if m:
        text = text[: m.start()]
    m = re.search(r"\\bibliography\{|\\begin\{thebibliography\}", text)
    if m:
        text = text[: m.start()]
    return text


def to_prose(text: str) -> str:
    """
    Reduce LaTeX to running prose so that word and sentence counts reflect what
    a reader reads. Floats are removed; captions are checked separately.
    """
    text = strip_comments(text)
    text = body_only(text)
    text = trim_backmatter(text)
    # Excise the keywords block. It is a semicolon-separated list with no full
    # stop, so it would otherwise merge into the Introduction's first sentence
    # and appear as one enormous sentence.
    text = re.sub(r"\{\s*\\bfseries\s+KEYWORDS.*?(?=\\section)", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{keywords\}.*?\\end\{keywords\}", " ", text, flags=re.S)
    for env in MATH_ENVS:
        text = re.sub(rf"\\begin{{{env}\*?}}.*?\\end{{{env}\*?}}", " MATH. ", text, flags=re.S)
    for env in FLOAT_ENVS:
        text = re.sub(rf"\\begin{{{env}\*?}}.*?\\end{{{env}\*?}}", " ", text, flags=re.S)
    text = re.sub(r"\\cite[a-z]*\{[^}]*\}", "CITE", text)
    text = re.sub(r"\\(?:ref|label|eqref|pageref)\{[^}]*\}", "REF", text)
    text = re.sub(r"\$[^$]*\$", "MATH", text)
    # Drop section titles entirely. Keeping them would make "Method." count as
    # a one-word sentence and skew every sentence-length statistic.
    text = re.sub(r"\\(?:sub)*section\*?\{[^}]*\}", " ", text)
    # Spacing and sizing commands must be dropped whole. Keeping the argument
    # would leave "1em" in the text, which blocks the sentence splitter and
    # silently fuses the sentences either side of a \vspace into one.
    text = re.sub(
        r"\\(?:v|h)space\*?\{[^}]*\}|\\(?:setlength|addvspace|vskip|hskip|"
        r"rule|includegraphics|label|graphicspath)\s*(?:\[[^\]]*\])?\{[^}]*\}",
        " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("~", " ").replace("\\", " ")
    text = re.sub(r"[{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(prose: str) -> list[str]:
    """
    Sentence splitter tuned for academic prose: protects common abbreviations
    and decimal numbers so they do not create false boundaries.
    """
    dot = "\x00"  # stand-in for a full stop that must not split a sentence
    protected = prose
    for abbr in ("e.g.", "i.e.", "et al.", "cf.", "vs.", "Fig.", "Eq.", "Tab.",
                 "Dr.", "Prof.", "approx.", "s.d.", "No."):
        protected = protected.replace(abbr, abbr.replace(".", dot))
    # A replacement template cannot carry a NUL escape, so substitute via callable.
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + dot + m.group(2), protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", protected)
    return [p.replace(dot, ".").strip() for p in parts if p.strip()]


def sections(text: str) -> list[tuple[str, int, str]]:
    """
    Return (level_name, char_offset, title) for each sectioning command, in
    document order. level_name is one of section / subsection / subsubsection.
    """
    out = []
    for m in re.finditer(r"\\(sub)*section(\*?)\{([^}]*)\}", strip_comments(text)):
        depth = (m.group(0).count("sub"))
        name = ("section", "subsection", "subsubsection")[min(depth, 2)]
        out.append((name, m.start(), m.group(3)))
    return out


def section_text(text: str, title_pattern: str) -> str:
    """
    Extract the body of the first \\section whose title matches title_pattern.
    Stops at the next \\section of the same or higher level.
    """
    text = strip_comments(text)
    secs = [s for s in sections(text) if s[0] == "section"]
    for i, (_, off, title) in enumerate(secs):
        if re.search(title_pattern, title, re.I):
            end = secs[i + 1][1] if i + 1 < len(secs) else len(text)
            return text[off:end]
    return ""


def numbers_in(text: str) -> list[tuple[str, str]]:
    """
    Every numeric literal with a snippet of surrounding context.
    Scientific notation, percentages, decimals and comma-grouped integers.
    """
    out = []
    pat = re.compile(
        r"(?<![\w.])("
        r"\d+\.\d+\s*\\times\s*10\^\{-?\d+\}"   # LaTeX scientific
        r"|\d+\.?\d*[eE][-+]?\d+"                # plain scientific
        r"|\d{1,3}(?:[,{]?\d{3})+(?:\.\d+)?"     # 1,620 or 1{,}620
        r"|\d+\.\d+"                             # decimal
        r"|\d+"                                  # integer
        r")(?![\w])"
    )
    for m in pat.finditer(text):
        lo = max(0, m.start() - 70)
        hi = min(len(text), m.end() + 70)
        ctx = re.sub(r"\s+", " ", text[lo:hi]).strip()
        out.append((m.group(1), ctx))
    return out


def canon_number(tok: str) -> str | None:
    """Normalise a numeric token to a plain float string, or None."""
    t = tok.replace("{,}", "").replace(",", "").strip()
    m = re.match(r"^([\d.]+)\s*\\times\s*10\^\{(-?\d+)\}$", t)
    if m:
        try:
            return repr(float(m.group(1)) * (10 ** int(m.group(2))))
        except ValueError:
            return None
    try:
        return repr(float(t))
    except ValueError:
        return None


def load_wordlist(path: Path) -> set[str]:
    """One entry per line, '#' comments, case-folded."""
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line.lower())
    return out


def emit(report: Report, json_path: str | None) -> int:
    report.print_human()
    if json_path:
        Path(json_path).write_text(report.to_json())
    blockers = report.count("BLOCKER")
    majors = report.count("MAJOR")
    print(f"  totals: {report.count('BLOCKER')} blocker, {majors} major, "
          f"{report.count('MINOR')} minor, {report.count('INFO')} info")
    return 1 if (blockers or majors) else 0


def die(msg: str) -> None:
    sys.exit(f"error: {msg}")
