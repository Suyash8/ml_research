# Manuscript

Paper versions live in numbered folders here. Git tracks edits inside a version;
the folders track the versions themselves. A new folder is created whenever the
paper changes enough that the previous PDF needs to stay readable on its own,
which is normally after a supervisor or reviewer returns comments.

## Layout

```
manuscript/
  shared/
    figures/              the 9 figure PNGs, one copy, used by every version
    references.bib        master bibliography, verified entry by entry
    REFERENCE_AUDIT.md    what was checked, corrected and deleted
  v1-cox-enet-pipeline/   first draft
  v2-multiomic-tcga/      second draft, the one returned with comments
  v3-simplified/          active version
```

Every version folder holds `paper.tex`, `paper.pdf`, a `NOTES.md` describing
what that version was, and a `build/` directory for LaTeX intermediates.

Two symlinks inside each version folder keep the archived `.tex` files working
without editing them:

- `images` -> `../shared/figures`, so `\includegraphics{images/fig1...}` resolves
- `references.bib` -> `../shared/references.bib` (v3 only)

`v1` and `v2` carry a frozen copy of the bibliography as it stood when they were
written, not a symlink. Their PDFs stay reproducible even though the master
bibliography has since been corrected.

The repository root also has an `images` symlink pointing at
`manuscript/shared/figures`, because several files under `scripts/` still write
figures to that path.

## Versions

| Version | Title | Status |
|---|---|---|
| v1 | A Calibrated, Explainable Cox Elastic-Net Pipeline for Pan-Cancer Survival Prediction from Multi-Omic TCGA Data | superseded |
| v2 | same title, expanded literature survey | returned with comments |
| v4 | same title, prose restyled to match published work in the field | superseded |
| v5 | same title, feedback-aligned structure and contributions | active |

v3 and v4 carry identical content: same sections, numbers, claims and citations.
They differ only in prose. v3 uses short single-clause sentences (mean 13.7 words)
and reads easily but mechanically; v4 matches the measured sentence rhythm of ten
published papers (mean 23.7 words). Pick on readability preference, not on risk.

## Building

```bash
bash scripts/build_paper.sh manuscript/v3-simplified
```

The script runs pdflatex, bibtex, then pdflatex twice more. It copies the PDF up
to the version folder and prints a log summary: undefined citations, undefined
references, overfull boxes, errors, bibtex warnings. All five should read 0.

## Checking

Build first, then run the verifier. A stale PDF makes the visual pass worthless
and the mismatch is silent.

```bash
bash scripts/build_paper.sh manuscript/v3-simplified

python3 .kiro/skills/paper-verifier/scripts/verify.py \
    manuscript/v3-simplified/paper.tex \
    --artifacts results data/analysis_reports \
    --repo .
```

Output lands in `manuscript/v3-simplified/verification/`. Start with
`SUMMARY.txt`. Individual stages can also be run alone; see
`.kiro/skills/paper-verifier/SKILL.md`.

The scripts cover what can be counted: prose rules, structure, figure
resolution and provenance, number traceability against the result files, and
bibliography structure. They do not cover three things that matter just as
much, and the summary ends by listing them:

1. every page raster under `verification/pages/` has to be looked at
2. every reference in `verification/reference_queue.txt` has to be checked online
3. every claim has to be judged against what the results support

A run that stops after the scripts is not a verification.

## Starting a new version

1. `cp -r manuscript/v3-simplified manuscript/v4-<short-name>`
2. Delete the copied `paper.pdf` and `build/`, keep the symlinks.
3. Write a `NOTES.md` saying what prompted the new version.
4. Add a row to the table above.
5. Leave older folders alone. They are the record of what was submitted.

## Numbers in the paper

Every figure quoted in v3 was traced back to a file under `results/` or
`data/analysis_reports/` before it was written down. The mapping is recorded in
`v3-simplified/NOTES.md`. Anything that could not be traced was deleted rather
than rounded or hedged.
