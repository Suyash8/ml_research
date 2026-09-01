#!/usr/bin/env python3
"""
Text-level verification: prose rules, structure, overstatement, AI tells.

    python3 check_text.py <paper.tex> [--json out.json] [--allow allowlist.txt]

Every check reports a count. A check that cannot produce a count reports INFO
and hands the judgement to the agent, which is the honest split: scripts count,
the agent decides. Nothing here silently passes on a "looks fine".
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    Report, emit, read_tex, strip_comments, body_only, to_prose,
    split_sentences, sections, section_text, load_wordlist,
)

MAX_SENTENCE_WORDS = 25
TARGET_MEAN_WORDS = 16

FIRST_PERSON = [
    r"\bwe\b", r"\bour\b", r"\bours\b", r"\bus\b", r"\bourselves\b",
    r"\bI\b", r"\bmy\b", r"\bmine\b", r"\bme\b", r"\blet\s?'?s\b",
]

BANNED_VOCAB = [
    # core AI vocabulary
    "delve", "leverage", "leverages", "leveraging", "utilize", "utilise",
    "utilized", "utilised", "utilizing", "robust", "robustly", "comprehensive",
    "comprehensively", "streamline", "streamlined", "foster", "facilitate",
    "pivotal", "nuanced", "multifaceted", "intricate", "intricacies",
    "tapestry", "testament", "showcase", "showcases", "interplay",
    "elucidate", "aforementioned", "ameliorate", "underscores", "highlights",
    # hedges and filler
    "it is important to note", "it is worth noting", "it is worth mentioning",
    "notably", "generally speaking", "in many cases", "it can be argued",
    "needless to say", "it goes without saying",
    # transitions
    "furthermore", "moreover", "it is clear that", "as previously mentioned",
    "additionally,",
    # formula openers and closers
    "in today's", "in conclusion", "in summary", "to summarize",
    "at the end of the day", "at its core", "under the hood",
    # quantifier inflation
    "a myriad of", "a plethora of", "in the realm of", "the landscape of",
    # signposting
    "let's dive in", "let us explore", "here's what", "without further ado",
]

# Claims that need either a citation or a statistical test behind them.
OVERSTATEMENT = {
    "state-of-the-art": "BLOCKER",
    "outperforms": "MAJOR",
    "outperformed": "MAJOR",
    "out-performs": "MAJOR",
    "superior to": "MAJOR",
    "best-in-class": "BLOCKER",
    "unprecedented": "BLOCKER",
    "groundbreaking": "BLOCKER",
    "revolutionary": "BLOCKER",
    "we prove": "BLOCKER",
    "proves that": "BLOCKER",
    "proven": "MAJOR",
    "guarantees": "MAJOR",
    "guaranteed": "MAJOR",
    "ensures that": "MINOR",
    "optimal": "MINOR",
    "the best": "MAJOR",
    "highly accurate": "MAJOR",
    "excellent performance": "MAJOR",
    "strong performance": "MINOR",
    "clinically validated": "BLOCKER",
    "ready for clinical": "BLOCKER",
    "novel": "MINOR",
    "significantly better": "MAJOR",
    "significantly improves": "MAJOR",
    "dramatically": "MAJOR",
    "substantially outperform": "MAJOR",
    "perfectly": "MINOR",
    "eliminates the need": "MINOR",
    "solves the problem": "MAJOR",
    "fully interpretable": "MINOR",
    "cutting-edge": "BLOCKER",
}

# Novelty claims. Bare "the first" is almost always an ordinal ("the first
# family", "the first was glioblastoma"), so only flag it where a priority
# claim is actually being made.
NOVELTY_PATTERNS = {
    r"\bthe first (?:to|study|work|paper|model|method|framework|pipeline|approach|"
    r"system|time\b)": "MAJOR",
    r"\bfor the first time\b": "MAJOR",
    r"\bwe are the first\b": "BLOCKER",
    r"\bno (?:prior|previous) work has\b": "MAJOR",
    r"\bhas never been (?:done|attempted|reported)\b": "MAJOR",
}

# "significant" is only safe next to a reported test.
STAT_WORDS = ("p =", "p<", "p >", "p-value", "q =", "confidence interval",
              "ci", "test", "corrected")

AI_TELL_PATTERNS = {
    "negation pivot 'not just'": r"\bnot just\b",
    "negation pivot 'not X, it is Y'": r"\bit(?:'s| is) not\b[^.]{0,60}\bit(?:'s| is)\b",
    "negation pivot 'rather than merely'": r"\brather than merely\b",
    "'turns out' pivot": r"\bturns out\b",
    # Only the rhetorical framing form. A plain quantitative comparative
    # ("needs more patients than one cohort provides") is normal academic prose.
    "comparative framing": r"\bmore of an? \w+ than\b|\bless about \w+ than\b|"
                           r"\bmore \w+ than it is \w+\b|\brather than merely\b|"
                           r"\bnot so much \w+ as\b",
    "landing phrase 'the real question'": r"\bthe real question\b",
    "landing phrase 'what really matters'": r"\bwhat really matters\b",
    "reframe pivot 'seen this way'": r"\bseen this way\b|\blaid out that way\b",
    "copula avoidance 'serves as'": r"\bserves as\b|\bstands as\b",
    "significance inflation": r"\bmarks a pivotal\b|\bstands as a testament\b|"
                              r"\bevolving landscape\b|\bsetting the stage for\b",
    "vague attribution": r"\b(?:industry observers|experts argue|critics have suggested|"
                         r"researchers agree|it is widely believed)\b",
    "pattern announcement": r"\bthe pattern is\b|\bthe key insight\b",
}

BRITISH_AMERICAN = [
    ("tumour", "tumor"), ("analyse", "analyze"), ("analysed", "analyzed"),
    ("normalise", "normalize"), ("normalised", "normalized"),
    ("standardise", "standardize"), ("standardised", "standardized"),
    ("regularise", "regularize"), ("regularised", "regularized"),
    ("summarise", "summarize"), ("summarised", "summarized"),
    ("behaviour", "behavior"), ("colour", "color"), ("centre", "center"),
    ("modelling", "modeling"), ("modelled", "modeled"),
    ("labelled", "labeled"), ("optimise", "optimize"), ("optimised", "optimized"),
    ("generalise", "generalize"), ("generalisation", "generalization"),
    ("recognise", "recognize"), ("licence", "license"), ("fibre", "fiber"),
]

# Technical terms that a first-year reader will not know. Each should carry a
# short plain-language gloss somewhere in the paper.
JARGON_NEEDING_GLOSS = {
    "hazard": r"hazard[^.]{0,120}?\b(?:risk|chance|rate) of\b|"
              r"\b(?:risk|chance)[^.]{0,60}hazard",
    "right censoring": r"censor\w*[^.]{0,160}?\b(?:still alive|end of the study|"
                       r"last (?:seen|contact)|survived at least)",
    "partial likelihood": r"partial log-?likelihood|partial likelihood",
    "concordance index": r"concordance index[^.]{0,200}?\b(?:order|ranks?|ranking)",
    "Brier score": r"[Bb]rier score[^.]{0,200}?\b(?:squared|error|difference)",
    "one-hot": r"one-hot[^.]{0,140}?\b(?:categor|binary column|own column)",
    "loading matrix": r"loading",
    "elastic net": r"elastic-?net[^.]{0,200}?\b(?:penalt|shrink)",
    "isotonic regression": r"isotonic regression[^.]{0,240}?\b(?:never decreas|"
                           r"monotonic|preserv\w+ the order|step function)",
}

EXPECTED_SECTION_ORDER = [
    ("introduction", r"introduction"),
    ("related work", r"related work|literature|background|prior work"),
    ("method", r"method|approach|materials|proposed"),
    ("results", r"results|experiments|evaluation"),
    ("discussion", r"discussion"),
    ("conclusion", r"conclusion|concluding"),
]

# Tokens that look like acronyms but are not, so should not demand an expansion.
DEFAULT_NON_ACRONYMS = {
    "a", "i", "ii", "iii", "iv", "v", "vi", "ok", "id", "no", "pm",
    "abstract", "keywords", "introduction", "and", "the", "of", "in", "or",
    "cite", "ref", "math", "true", "false",
}


def find_all(pattern: str, text: str, flags=re.I):
    return list(re.finditer(pattern, text, flags))


def ctx(text: str, m: re.Match, pad: int = 60) -> str:
    lo, hi = max(0, m.start() - pad), min(len(text), m.end() + pad)
    return "..." + re.sub(r"\s+", " ", text[lo:hi]).strip() + "..."


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_first_person(rep: Report, prose: str) -> None:
    hits = []
    for pat in FIRST_PERSON:
        for m in re.finditer(pat, prose):
            hits.append(ctx(prose, m))
    if hits:
        rep.add("first-person", "BLOCKER",
                f"{len(hits)} first-person pronoun(s); the paper must be impersonal",
                "\n".join(hits[:20]))
    rep.stats["first-person pronouns"] = len(hits)


def check_sentences(rep: Report, prose: str) -> None:
    sents = split_sentences(prose)
    counts = [len(s.split()) for s in sents]
    if not counts:
        return
    long_ones = [(n, s) for n, s in zip(counts, sents) if n > MAX_SENTENCE_WORDS]
    mean = sum(counts) / len(counts)

    rep.stats["sentences"] = len(counts)
    rep.stats["mean sentence words"] = f"{mean:.1f}"
    rep.stats["longest sentence words"] = max(counts)
    rep.stats[f"sentences over {MAX_SENTENCE_WORDS} words"] = len(long_ones)

    if long_ones:
        rep.add("sentence-length", "MAJOR",
                f"{len(long_ones)} sentence(s) over {MAX_SENTENCE_WORDS} words",
                "\n".join(f"[{n}w] {s[:200]}" for n, s in sorted(long_ones, reverse=True)[:20]))
    if mean > TARGET_MEAN_WORDS:
        rep.add("sentence-length", "MINOR",
                f"mean sentence length {mean:.1f} words, target under {TARGET_MEAN_WORDS}")

    # Uniformity: a flat rhythm reads as machine-written even when every
    # sentence is legal on its own.
    band = sum(1 for n in counts if 10 <= n <= 20)
    frac = band / len(counts)
    rep.stats["fraction of sentences 10-20 words"] = f"{frac:.0%}"
    # A hard 25-word ceiling and a wide length spread pull against each other.
    # Academic prose cannot reach the 20-word range that general writing advice
    # asks for, so this is reported as context unless it is extreme.
    if frac > 0.75 and len(counts) > 40:
        rep.add("sentence-rhythm", "MINOR",
                f"{frac:.0%} of sentences sit in the 10-20 word band, which reads flat",
                f"range {min(counts)}-{max(counts)} words. Merge a few adjacent "
                "short sentences and split a few mid-length ones.")
    elif frac > 0.55 and len(counts) > 40:
        rep.add("sentence-rhythm", "INFO",
                f"{frac:.0%} of sentences sit in the 10-20 word band",
                f"range {min(counts)}-{max(counts)} words. Expected given the "
                f"{MAX_SENTENCE_WORDS}-word ceiling. Worth one pass for variety, "
                "not a defect.")
    if max(counts) - min(counts) < 15 and len(counts) > 40:
        rep.add("sentence-rhythm", "MINOR",
                f"sentence length range is only {max(counts) - min(counts)} words")


def check_banned_vocab(rep: Report, prose: str) -> None:
    hits = []
    low = prose.lower()
    for word in BANNED_VOCAB:
        for m in re.finditer(rf"(?<![a-z]){re.escape(word)}(?![a-z])", low):
            hits.append(f"{word!r}: {ctx(prose, m, 45)}")
    if hits:
        rep.add("banned-vocabulary", "MAJOR",
                f"{len(hits)} banned word(s) or phrase(s)", "\n".join(hits[:25]))
    rep.stats["banned vocabulary hits"] = len(hits)


def check_overstatement(rep: Report, prose: str) -> None:
    by_sev: dict[str, list[str]] = {}
    for phrase, sev in OVERSTATEMENT.items():
        for m in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", prose, re.I):
            by_sev.setdefault(sev, []).append(f"{phrase!r}: {ctx(prose, m, 70)}")
    for pat, sev in NOVELTY_PATTERNS.items():
        for m in re.finditer(pat, prose, re.I):
            by_sev.setdefault(sev, []).append(f"novelty claim: {ctx(prose, m, 70)}")

    total = sum(len(v) for v in by_sev.values())
    for sev, items in by_sev.items():
        rep.add("overstatement", sev,
                f"{len(items)} claim(s) at {sev} strength that need evidence or softening",
                "\n".join(items[:20]))
    rep.stats["overstatement hits"] = total

    # "significant" without a test nearby is a word-choice error in a paper.
    for m in re.finditer(r"\bsignifican\w*", prose, re.I):
        window = prose[max(0, m.start() - 240): m.end() + 240].lower()
        if not any(w in window for w in STAT_WORDS):
            rep.add("overstatement", "MAJOR",
                    "'significant' used without a nearby statistical test",
                    ctx(prose, m, 90))


def check_ai_tells(rep: Report, raw: str, prose: str) -> None:
    words = len(prose.split())

    em = prose.count("\u2014")
    allowed_em = max(0, words // 300)
    rep.stats["em dashes"] = f"{em} (budget {allowed_em})"
    if em > allowed_em:
        rep.add("ai-tell", "MINOR", f"{em} em dash(es), budget is {allowed_em} at this length")

    semis = prose.count(";")
    rep.stats["semicolons"] = semis
    if semis > words // 400:
        rep.add("ai-tell", "MINOR",
                f"{semis} semicolon(s); prefer a full stop unless listing comma-bearing items")

    curly = sum(prose.count(c) for c in "\u201c\u201d\u2018\u2019")
    rep.stats["curly quotes"] = curly
    if curly:
        rep.add("ai-tell", "MINOR", f"{curly} curly quote/apostrophe character(s)",
                "Use straight ' and \" or the LaTeX `` '' forms.")

    tell_hits = 0
    for label, pat in AI_TELL_PATTERNS.items():
        ms = find_all(pat, prose)
        if ms:
            tell_hits += len(ms)
            rep.add("ai-tell", "MINOR", f"{len(ms)}x {label}",
                    "\n".join(ctx(prose, m, 70) for m in ms[:6]))
    rep.stats["rhetorical AI tells"] = tell_hits

    # Anaphora: consecutive sentences opening on the same word.
    sents = split_sentences(prose)
    runs = 0
    for a, b in zip(sents, sents[1:]):
        wa, wb = a.split(), b.split()
        if wa and wb and wa[0].lower() == wb[0].lower() and len(wa[0]) > 3:
            runs += 1
    rep.stats["consecutive same-opening sentences"] = runs
    if runs > max(3, len(sents) // 40):
        rep.add("ai-tell", "MINOR",
                f"{runs} sentence pair(s) opening on the same word; vary the openers")

    # Near-duplicate sentences: recycled phrasing across sections.
    norm = [re.sub(r"[^a-z ]", "", s.lower()) for s in sents]
    dupes = [s for s, n in Counter(norm).items() if n > 1 and len(s.split()) > 6]
    if dupes:
        rep.add("repetition", "MAJOR",
                f"{len(dupes)} sentence(s) repeated verbatim",
                "\n".join(d[:150] for d in dupes[:10]))
    rep.stats["duplicated sentences"] = len(dupes)


def check_acronyms(rep: Report, prose: str, allow: set[str]) -> None:
    """
    Detect acronyms rather than checking a fixed list, so a newly introduced
    one cannot slip through. An acronym is a token of 2+ characters that is
    mostly uppercase. Gene symbols look identical, so anything on the
    allowlist is skipped and everything else is reported for the agent to
    confirm.
    """
    # Take the maximal token starting at a capital, then decide whether it is
    # acronym-shaped by the density of capitals. Matching whole tokens matters:
    # splitting RNA-Seq into "RNA" or ACSM2A into "ACSM" both breaks the
    # allowlist and points at the wrong first use.
    pat = re.compile(r"(?<![A-Za-z0-9-])([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)")
    seen: dict[str, list[int]] = {}
    for m in pat.finditer(prose):
        tok = m.group(1)
        letters = [c for c in tok if c.isalpha()]
        uppers = [c for c in letters if c.isupper()]
        if len(uppers) < 2 or not letters:
            continue
        if len(uppers) / len(letters) < 0.5:
            continue
        if tok.lower() in DEFAULT_NON_ACRONYMS or tok.lower() in allow:
            continue
        seen.setdefault(tok, []).append(m.start())

    unexpanded = []
    for tok, positions in sorted(seen.items()):
        first = positions[0]
        after = prose[first + len(tok):]
        before = prose[:first]
        # An expansion looks like "Long Form (ACR)" or "ACR (Long Form)".
        wrapped = bool(re.search(r"\(\s*$", before)) and bool(re.match(r"\s*\)", after))
        followed = bool(re.match(r"\s*\(\s*[A-Za-z]", after))
        if wrapped or followed:
            continue
        unexpanded.append(
            f"{tok} (used {len(positions)}x, first at char {first}): "
            f"...{prose[max(0, first - 85):first + 45].strip()}..."
        )

    rep.stats["distinct acronym-like tokens"] = len(seen)
    rep.stats["without a parenthetical expansion"] = len(unexpanded)
    if unexpanded:
        rep.add("acronym", "MAJOR",
                f"{len(unexpanded)} acronym-like token(s) with no expansion at first use",
                "\n".join(unexpanded[:30]) +
                "\n\nGene symbols and dataset codes will appear here. Confirm each one, "
                "then add real non-acronyms to the allowlist file.")


def check_jargon_gloss(rep: Report, prose: str) -> None:
    missing = []
    for term, gloss_pat in JARGON_NEEDING_GLOSS.items():
        if not re.search(re.escape(term.split()[0]), prose, re.I):
            continue  # term not used, nothing to gloss
        if not re.search(gloss_pat, prose, re.I | re.S):
            missing.append(term)
    rep.stats["jargon terms lacking a plain gloss"] = len(missing)
    if missing:
        rep.add("jargon", "MAJOR",
                f"{len(missing)} technical term(s) used without a plain-language gloss",
                ", ".join(missing) +
                "\nAdd one short sentence saying what each means before it does any work.")


def check_spelling_consistency(rep: Report, prose: str) -> None:
    mixed = []
    low = prose.lower()
    brit = amer = 0
    for b, a in BRITISH_AMERICAN:
        nb = len(re.findall(rf"\b{b}\b", low))
        na = len(re.findall(rf"\b{a}\b", low))
        brit += nb
        amer += na
        if nb and na:
            mixed.append(f"{b} ({nb}x) and {a} ({na}x)")
    rep.stats["British spellings"] = brit
    rep.stats["American spellings"] = amer
    if mixed:
        rep.add("consistency", "MAJOR",
                f"{len(mixed)} word(s) spelled both ways", "\n".join(mixed))
    elif brit and amer:
        rep.add("consistency", "MINOR",
                f"both spelling conventions present ({brit} British, {amer} American)",
                "Pick one and apply it throughout, or match the venue's house style.")


def check_structure(rep: Report, raw: str) -> None:
    secs = sections(raw)
    top = [(t, o) for lvl, o, t in secs if lvl == "section"]
    rep.stats["top-level sections"] = len(top)

    titles = [t.lower() for t, _ in top]

    # Required sections present?
    for label, pat in EXPECTED_SECTION_ORDER:
        if not any(re.search(pat, t) for t in titles):
            rep.add("structure", "MAJOR", f"no section matching '{label}'")

    # Order correct?
    found_order = []
    for t in titles:
        for i, (label, pat) in enumerate(EXPECTED_SECTION_ORDER):
            if re.search(pat, t):
                found_order.append((i, label, t))
                break
    idxs = [i for i, _, _ in found_order]
    if idxs != sorted(idxs):
        rep.add("structure", "MAJOR", "sections are out of conventional order",
                " -> ".join(f"{lbl}" for _, lbl, _ in found_order))

    # Introduction must be continuous prose.
    intro = section_text(raw, r"introduction")
    if intro:
        subs = re.findall(r"\\subsection\*?\{([^}]*)\}", intro)
        rep.stats["subsections inside Introduction"] = len(subs)
        if subs:
            rep.add("structure", "BLOCKER",
                    f"Introduction contains {len(subs)} subsection heading(s); "
                    "it must be continuous prose",
                    "; ".join(subs))
        paras = [p for p in re.split(r"\n\s*\n", intro) if len(p.split()) > 25]
        rep.stats["Introduction paragraphs"] = len(paras)
        if len(paras) < 4:
            rep.add("structure", "MINOR",
                    f"Introduction has {len(paras)} substantial paragraph(s); "
                    "motivation, background, existing work, gap and contributions "
                    "normally need at least four")
    else:
        rep.add("structure", "MAJOR", "could not locate an Introduction section")

    # Title length.
    m = re.search(r"\\(?:LARGE|Large|huge)\\bfseries\s*(.*?)\\par", raw, re.S)
    if not m:
        m = re.search(r"\\title\{(.*?)\}", raw, re.S)
    if m:
        title = re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+", "", m.group(1))).strip()
        n = len(title.split())
        rep.stats["title words"] = n
        rep.stats["title"] = title[:120]
        if n > 14:
            rep.add("title", "MAJOR",
                    f"title is {n} words; aim for 12 or fewer", title)
        stacked = re.findall(
            r"\b(calibrated|explainable|interpretable|novel|robust|unified|efficient|"
            r"comprehensive|hybrid|advanced|multi-omic|end-to-end)\b", title, re.I)
        if len(stacked) > 2:
            rep.add("title", "MAJOR",
                    f"title stacks {len(stacked)} qualifier adjectives",
                    ", ".join(stacked) + "\nKeep the one or two the paper is about.")

    # Declarations expected by most journals.
    for label, pat in [("funding", r"funding"),
                       ("competing interests", r"competing interest|conflict of interest|"
                                               r"disclosure of interest"),
                       ("data availability", r"data availability")]:
        if not re.search(rf"\\section\*?\{{[^}}]*{pat}", raw, re.I):
            rep.add("structure", "MINOR", f"no '{label}' declaration section")

    # Unfilled placeholders.
    ph = find_all(r"\[(?:insert|todo|tbd|xxx|placeholder)[^\]]*\]|\bTODO\b|\bFIXME\b|"
                  r"\bxyz\b", raw)
    rep.stats["placeholders remaining"] = len(ph)
    if ph:
        rep.add("structure", "MAJOR",
                f"{len(ph)} unfilled placeholder(s) in the source",
                "\n".join(ctx(raw, m, 45) for m in ph[:15]))


def check_floats(rep: Report, raw: str) -> None:
    raw = strip_comments(raw)
    labels = set(re.findall(r"\\label\{([^}]*)\}", raw))
    refs = set()
    for group in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", raw):
        refs.update(k.strip() for k in group.split(","))

    floats = {l for l in labels if l.split(":")[0] in ("fig", "tab", "table", "figure")}
    orphan = sorted(floats - refs)
    dangling = sorted(refs - labels)

    rep.stats["labelled floats"] = len(floats)
    rep.stats["floats never referenced"] = len(orphan)

    if orphan:
        rep.add("floats", "MAJOR",
                f"{len(orphan)} float(s) never referenced from the text",
                "\n".join(orphan))
    if dangling:
        rep.add("floats", "BLOCKER",
                f"{len(dangling)} \\ref to a label that does not exist",
                "\n".join(dangling))

    # Every float needs a caption, and the caption should say something. A
    # figure caption has to stand on its own, so it needs more words than a
    # table caption. Subcaptions like "Before filtering" are labels, not
    # captions, and are excluded.
    min_words = {"figure": 8, "table": 4}
    for env in ("figure", "table"):
        blocks = re.findall(rf"\\begin{{{env}\*?}}(.*?)\\end{{{env}\*?}}", raw, re.S)
        for i, b in enumerate(blocks, 1):
            outer = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", " ", b, flags=re.S)
            m = re.search(r"\\caption\{", outer)
            if not m:
                rep.add("floats", "MAJOR", f"{env} #{i} has no top-level caption")
                continue
            depth, j = 0, m.end() - 1
            while j < len(outer):
                if outer[j] == "{":
                    depth += 1
                elif outer[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            cap = outer[m.end():j]
            words = len(re.sub(r"\\[a-zA-Z]+|[{}]", " ", cap).split())
            if words < min_words[env]:
                rep.add("floats", "MINOR",
                        f"{env} #{i} caption is only {words} words; "
                        f"expected at least {min_words[env]}",
                        cap.strip()[:150])

    # Float placement specifiers: [H] forces exact placement and often causes
    # large gaps. Worth knowing, not a defect.
    h_forced = len(re.findall(r"\\begin\{(?:figure|table)\}\[H\]", raw))
    if h_forced:
        rep.add("floats", "INFO",
                f"{h_forced} float(s) use [H] exact placement",
                "Check the page images for large white gaps. [htbp] usually flows better.")


def check_citation_hygiene(rep: Report, raw: str) -> None:
    raw_nc = strip_comments(raw)
    body = body_only(raw_nc)

    keys: list[str] = []
    for group in re.findall(r"\\cite[a-z]*\{([^}]*)\}", body):
        keys.extend(k.strip() for k in group.split(",") if k.strip())
    rep.stats["citation commands"] = len(re.findall(r"\\cite[a-z]*\{", body))
    rep.stats["distinct keys cited"] = len(set(keys))

    # Citation density per section, judged against the enclosing top-level
    # section. Method and Results describe this work, so they legitimately run
    # without citations. Introduction and Related Work do not.
    secs = sections(raw_nc)
    NEEDS_CITES = r"introduction|related work|literature|background|prior work"
    enclosing = ""
    for i, (lvl, off, title) in enumerate(secs):
        if lvl == "section":
            enclosing = title
        end = secs[i + 1][1] if i + 1 < len(secs) else len(raw_nc)
        chunk = raw_nc[off:end]
        words = len(to_prose(chunk).split())
        n_cites = len(re.findall(r"\\cite[a-z]*\{", chunk))
        if words < 60 or n_cites:
            continue
        if re.search(NEEDS_CITES, enclosing, re.I) or re.search(
                r"limitation|research gap|related", title, re.I):
            rep.add("citations", "MAJOR",
                    f"'{title}' has {words} words and no citation",
                    f"inside section '{enclosing}', which must be sourced")

    # Claim verbs that almost always need a source.
    prose = to_prose(raw)
    unsourced = []
    for m in re.finditer(
            r"[^.]*\b(?:is known to|has been shown|studies (?:show|have shown)|"
            r"is widely used|is the standard|it is established|research (?:shows|indicates)|"
            r"prior work|previous studies|are associated with|is associated with)\b[^.]*\.",
            prose, re.I):
        sent = m.group(0)
        if "CITE" not in sent:
            unsourced.append(sent.strip()[:190])
    rep.stats["claim sentences without a citation"] = len(unsourced)
    if unsourced:
        rep.add("citations", "BLOCKER",
                f"{len(unsourced)} sentence(s) assert prior knowledge with no citation",
                "\n".join(unsourced[:15]))


def check_empty_sentences(rep: Report, prose: str) -> None:
    """
    A sentence earns its place by carrying a number, a citation, a named
    entity, or a mechanism. Sentences carrying none of those, and matching a
    filler shape, are candidates for deletion.
    """
    # Strong shapes are near-certain filler in any register.
    strong_filler = [
        r"\bplays? an? (?:important|key|vital|major|crucial) role\b",
        r"\bhas (?:attracted|received) (?:much|significant|considerable|widespread) attention\b",
        r"\bis an? (?:important|active|growing|emerging) (?:area|field|topic)\b",
        r"\bhas become increasingly\b",
        r"\bin recent years\b",
        r"\bwith the (?:rapid )?(?:development|advancement|growth) of\b",
        r"\bmany (?:studies|approaches|methods) have been proposed\b",
        r"\bis of great (?:importance|interest)\b",
        r"\bremains an open (?:problem|question|challenge)\b",
    ]
    # Weak shape: a bare demonstrative sentence. Sometimes a real clarification,
    # so it is reported at lower severity for the agent to judge.
    weak_filler = (
        r"^(?:This|These|It|That|There) (?:is|are|was|were|means|shows|demonstrates|"
        r"suggests|indicates)\b[^,]{0,45}\.$"
    )

    strong, weak = [], []
    for s in split_sentences(prose):
        if re.search(r"\d", s) or "CITE" in s:
            continue
        if any(re.search(p, s, re.I) for p in strong_filler):
            strong.append(s[:170])
        elif re.search(weak_filler, s):
            weak.append(s[:170])

    rep.stats["strong filler sentences"] = len(strong)
    rep.stats["weak filler sentences"] = len(weak)
    if strong:
        rep.add("empty-sentences", "MAJOR",
                f"{len(strong)} sentence(s) match a strong filler shape",
                "\n".join(strong[:20]))
    if weak:
        rep.add("empty-sentences", "MINOR",
                f"{len(weak)} bare demonstrative sentence(s); confirm each adds information",
                "\n".join(weak[:20]) +
                "\nCover each one and ask whether the paragraph lost anything.")

    # Short sentences are wanted for rhythm, so they are reported as context
    # rather than as defects. They still have to carry meaning.
    shorts = [s for s in split_sentences(prose)
              if len(s.split()) <= 5 and not s.endswith(":")]
    rep.stats["sentences of 5 words or fewer"] = len(shorts)
    if shorts:
        rep.add("empty-sentences", "INFO",
                f"{len(shorts)} very short sentence(s); confirm each earns its place",
                "\n".join(shorts[:15]))


def check_tense(rep: Report, raw: str) -> None:
    results = section_text(raw, r"results|experiments")
    if not results:
        return
    prose = to_prose(results)
    # "Table 3 reports..." and "Figure 5 shows..." are correct in the present
    # tense, because the table still does that. Only an experiment described in
    # the present tense is the error, so exclude float subjects.
    present = []
    for m in re.finditer(r"\b(achieves|reaches|shows|obtains|performs|yields|gives|"
                         r"produces|reports|attains|demonstrates)\b", prose, re.I):
        before = prose[max(0, m.start() - 90):m.start()]
        if re.search(r"\b(?:Table|Figure|Fig\.|REF|MATH|panel|curve|plot|row|column)\s*\S*\s*$",
                     before, re.I):
            continue
        present.append(m)
    if present:
        rep.add("tense", "MINOR",
                f"{len(present)} present-tense reporting verb(s) in Results; "
                "a completed experiment takes the past tense",
                "\n".join(ctx(prose, m, 70) for m in present[:12]))
    rep.stats["present-tense verbs in Results"] = len(present)


def check_formatting_consistency(rep: Report, raw: str) -> None:
    prose = to_prose(raw)

    # Thousands separators used inconsistently.
    grouped = len(re.findall(r"\d\{,\}\d{3}\b|\d,\d{3}\b", raw))
    ungrouped = [m.group(0) for m in re.finditer(r"(?<![\d.,{])\d{4,}(?![\d,}])", prose)]
    ungrouped = [u for u in ungrouped if not (1900 <= int(u) <= 2100)]
    if grouped and ungrouped:
        rep.add("consistency", "MINOR",
                f"{grouped} number(s) use a thousands separator but "
                f"{len(ungrouped)} four-plus digit number(s) do not",
                ", ".join(ungrouped[:12]))

    # Same metric quoted at different precision. A value quoted from another
    # paper keeps that paper's precision, so a nearby citation exempts it.
    for name, pat in [("concordance index", r"concordance index[^.]{0,60}?(0\.\d+)"),
                      ("Brier score", r"[Bb]rier[^.]{0,60}?(0\.\d+)"),
                      ("area under the curve", r"area under the curve[^.]{0,60}?(0\.\d+)")]:
        own = []
        for m in re.finditer(pat, prose):
            window = prose[max(0, m.start() - 130): m.end() + 130]
            if "CITE" in window:
                continue  # quoted from a cited source
            own.append(m.group(1))
        precisions = {len(v.split(".")[1]) for v in own}
        if len(precisions) > 1:
            rep.add("consistency", "MINOR",
                    f"{name} reported at mixed precision: {sorted(precisions)} decimals",
                    ", ".join(sorted(set(own))[:12]))

    # Percent sign spacing and style.
    if re.search(r"\d\s+%", raw) and re.search(r"\d\\?%", raw):
        rep.add("consistency", "MINOR", "percent sign spacing is inconsistent")

    # Hyphen vs en dash in numeric ranges.
    hyph = len(re.findall(r"\d\s?-\s?\d", prose))
    endash = len(re.findall(r"\d\s?[\u2013-]{2}\s?\d", raw))
    if hyph and endash:
        rep.add("consistency", "MINOR",
                f"numeric ranges use both a hyphen ({hyph}x) and an en dash ({endash}x)")

    # Unescaped underscores outside math or verbatim would have failed the
    # build, but stray \texttt of field names is worth flagging for style.
    rep.stats["\\texttt spans"] = len(re.findall(r"\\texttt\{", raw))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--json")
    ap.add_argument("--allow", help="allowlist of tokens that are not acronyms")
    args = ap.parse_args()

    tex = Path(args.tex).resolve()
    raw = read_tex(tex)
    prose = to_prose(raw)

    allow = load_wordlist(Path(args.allow)) if args.allow else set()
    default_allow = Path(__file__).parent.parent / "references" / "not-acronyms.txt"
    allow |= load_wordlist(default_allow)

    rep = Report(f"TEXT CHECK  {tex.name}")
    rep.stats["body prose words"] = len(prose.split())

    check_structure(rep, raw)
    check_first_person(rep, prose)
    check_sentences(rep, prose)
    check_acronyms(rep, prose, allow)
    check_jargon_gloss(rep, prose)
    check_banned_vocab(rep, prose)
    check_overstatement(rep, prose)
    check_ai_tells(rep, raw, prose)
    check_empty_sentences(rep, prose)
    check_spelling_consistency(rep, prose)
    check_formatting_consistency(rep, raw)
    check_tense(rep, raw)
    check_floats(rep, raw)
    check_citation_hygiene(rep, raw)

    return emit(rep, args.json)


if __name__ == "__main__":
    sys.exit(main())
