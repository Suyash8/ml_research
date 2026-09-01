---
name: paper-verifier
description: >
  Use whenever a research paper, manuscript, thesis chapter, abstract or conference
  submission needs to be checked, reviewed, audited, proofread or validated before
  submission. Trigger it when the user says "verify the paper", "check the paper",
  "review my manuscript", "find problems in the paper", "is this ready to submit",
  "audit the references", "check if the numbers are right", "does this look
  AI-generated", "check the figures", or when a supervisor or reviewer has returned
  comments and the draft needs a full pass. Also use it after any substantial edit to
  a paper, because a targeted edit routinely breaks something elsewhere: a number in
  the abstract stops matching the results, a deleted claim orphans a reference, a
  reworded sentence reintroduces first person. Run it on the compiled PDF as well as
  the source, never on the source alone.
---

# Paper Verifier

Finds every problem in a paper and lists all of them. Not a sample, not the
interesting ones, all of them, each with a severity and a location.

The central design decision: **scripts count, the agent judges.** Neither alone
works. A script can tell you a sentence is 34 words long, that a reference has no
identifier, that a number appears nowhere in the results. A script cannot tell you
that a figure contradicts its caption, that a reference is fabricated, or that a
claim is dishonest. An agent can judge all three but cannot reliably count.

Running only the scripts produces a clean-looking report on a broken paper. That
failure has a specific shape: the mechanical pass reports zero blockers, everyone
relaxes, and the paper goes out with an invented citation in it. **A verification
that stops after the scripts is not a verification.** The scripts end by printing
the queue of work they could not do, so that stopping early is visible.

## Why the PDF, not just the source

Reading the `.tex` tells you what was requested. Reading the PDF tells you what a
reviewer sees. These differ, and the differences are the embarrassing ones.

A real example from this repository. The source and the caption both described a
Pearson correlation filter, correctly. Every text check passed. The rendered figure
carried colorbar labels reading "Cosine Similarity |CosSim|", left over from an
earlier version of the method. The figure contradicted its own caption on the same
page. No amount of reading the source would have found it. One look at the page
raster found it immediately.

So the pipeline always converts the PDF to text and to one image per page, and the
agent always looks at every page.

## Workflow

### Step 1: build, so the PDF matches the source

A stale PDF makes the whole visual pass worthless, and the mismatch is silent.
Rebuild first, and confirm the log is clean.

```bash
bash scripts/build_paper.sh manuscript/v3-simplified
```

Undefined citations, undefined references and errors must all read 0. If they do
not, stop and fix the build. Verifying a paper that does not compile wastes the
rest of the pass.

If the project has no build script, run pdflatex, bibtex, pdflatex, pdflatex and
read the log.

### Step 2: run the mechanical pass

```bash
python3 .kiro/skills/paper-verifier/scripts/verify.py \
    manuscript/v3-simplified/paper.tex \
    --artifacts results data/analysis_reports \
    --repo .
```

This runs five stages and writes everything to `<paper dir>/verification/`:

| Output | What it holds |
|---|---|
| `SUMMARY.txt` | severity totals and the remaining agent queue |
| `summary.json` | the same, machine readable |
| `pages/page-NN.png` | one raster per page, for Step 3 |
| `pdf_text.txt` | PDF text in reading order |
| `pdf_text_layout.txt` | PDF text preserving column layout |
| `embedded_images.tsv` | every raster in the PDF with its true resolution |
| `reference_queue.txt` | one checklist entry per reference, for Step 4 |
| `text.json` `assets.json` `numbers.json` `bib.json` | per-stage findings |
| `logs/` | full output of each stage |

`--artifacts` should point at wherever the results actually live. Without it the
number-traceability stage has nothing to match against and will report every
figure as untraced.

### Step 3: look at every page

Not a sample. Every page. Read each `pages/page-NN.png` as an image, one at a time.

Per page, check:

- Text crossing into the margin, or a line sticking out past the block
- A table wider than the text block, or one split awkwardly across pages
- A figure too small for its axis labels to be readable at print size
- A caption separated from its float, or on a different page from it
- Large white gaps, usually from `[H]` placement
- Fonts that change mid-page, or a figure whose font differs from the body
- **Figure content against its caption.** Does the figure show what the caption
  claims? Do the axis labels, legend and colorbar agree with the body text?
- Numbers in a figure against the same numbers in the text
- Headers, page numbers, orphaned headings at a page foot

Write down what you find with its page number. This is the slowest step and the
one that finds what nothing else finds.

### Step 4: verify every reference on the web

Work through `reference_queue.txt`. One lookup per entry. For each, confirm the
title, the **full author list**, the year, the venue, the volume and the pages.

Check the author list name by name. This is not optional and it is not paranoia.
Fabricated references are overwhelmingly plausible-looking: the title is real, the
journal is real, the year is right, and two of the five authors do not exist. Every
structural check passes. Only a lookup catches it.

Three outcomes:

- **Confirmed.** Move on.
- **Wrong in a field.** Correct the field. Note what was wrong.
- **Cannot be found.** Delete the entry, then delete or rewrite the sentence that
  depended on it. Do not soften the sentence and keep the citation. Report every
  deletion to the user with the reason.

Then check that each citation supports the specific sentence it hangs on. A methods
paper cannot support a claim about clinical practice. This drift is common and
reviewers catch it.

### Step 5: judge the claims

Read the overstatement and number findings, and decide on each one.

- Every number in the abstract and conclusion must match the body exactly. Rounding
  a metric up on the way into the abstract is a classic reviewer catch.
- Every comparison against prior work needs a citation and a statement of what
  differs. "Comparable to published benchmarks" without naming one is not a
  comparison.
- Hyperparameters selected at the edge of their search grid mean the grid was too
  small. That belongs in the limitations, not hidden.
- A univariate screen does not show independent effect. "Genome-wide significance"
  means the genome-wide threshold, not a small p-value.
- Anything the results cannot support gets deleted, not hedged.

Read `references/ai-tells.md` before this step and apply the image checklist to
every figure.

### Step 6: read the source figures

Open each figure file directly, not just as rendered on the page. Check for garbled
text, axis labels without units, unreadable legends, mismatched fonts between
panels, decorative gradients or 3D effects, and invented data points.

Pay particular attention to any figure the asset stage reported as having **no
generating script**. In a research paper every figure should be reproducible from
code in the repository. A figure with no script behind it was hand-drawn, pasted
from elsewhere, or generated by a model. All three need a human decision, and the
third needs the figure regenerated from data.

### Step 7: write the report

Use the format in `references/report-template.md`. Group by severity, not by check,
because the reader wants to know what to fix first. Every finding needs a location
specific enough to act on: a section name, a page number, a line, or a quoted
phrase.

State plainly what was verified and what was not. If 12 of 47 references were
checked, say 12 of 47. Do not imply a complete pass.

## Severity

| Level | Meaning |
|---|---|
| BLOCKER | factually wrong, unverifiable, or breaks a hard rule. Cannot be sent. |
| MAJOR | a reviewer will very likely raise it. Fix before sending. |
| MINOR | polish. Fix if time allows. |
| INFO | context for a judgement call. Not a defect by itself. |

A number that cannot be traced to an artifact is a BLOCKER, not a MINOR. It is the
only category of error that cannot be recovered after publication.

## What each stage checks

**Text** (`check_text.py`): first-person pronouns; sentences over 25 words; mean
sentence length and rhythm; acronyms detected automatically and checked for an
expansion at first use; technical terms used without a plain-language gloss; banned
vocabulary; overstatement and novelty claims; "significant" used with no test
nearby; AI rhetorical tells; em dash, semicolon and curly quote budgets; repeated
sentences; filler sentence shapes; British against American spelling; mixed metric
precision; thousands separators; tense in Results; section presence and order;
subsections inside the Introduction; title length and stacked adjectives; unfilled
placeholders; missing declarations; float captions and cross-references; citation
density per section; claim verbs with no citation.

**Assets** (`check_assets.py`): every graphic resolves on disk; effective print
resolution of every raster in the PDF; stretched images; both width and height set
without `keepaspectratio`; diagrams supplied as raster rather than vector; images
reused under two names; figures never used; fonts not embedded; page size; and
whether each figure has a generating script.

**Numbers** (`check_numbers.py`): every numeric literal matched against the result
files at every sensible rounding; numbers in the abstract or conclusion that do not
appear in the body; split sizes that do not sum to the cohort; "X of Y (Z%)" that
does not divide; ranges stated backwards; p reported as zero; correlations or
probabilities outside their valid range; means with no spread; sentences carrying
four or more decimals.

**Bibliography** (`check_bib.py`): required fields per entry type; entries with no
DOI, URL or ISBN; "and others" or "et al." in an author field; author names with no
given name; articles missing locator fields; implausible years; a key whose year or
surname disagrees with the entry; duplicate keys, titles and DOIs; keys cited but
undefined; entries never cited; recency balance; preprint share.

## Configuration

`references/not-acronyms.txt` lists tokens that look like acronyms but are not,
mostly gene symbols and dataset codes. Add project-specific symbols there.

Never add a real acronym to that file to silence the check. If the check fires on
PCA, AUC or RMST, the paper is missing an expansion and the check is correct. The
file says so at the point where the temptation arises.

## Reference files

- `references/rules.md` — the full rule catalogue with the reasoning behind each
- `references/ai-tells.md` — text and image tells, including the image checklist
- `references/report-template.md` — the output format

## Honest limits

- Reference verification needs network access. Without it, say so and mark every
  entry unverified rather than implying a pass.
- The empty-sentence and filler checks have false positives by design. They are
  tuned to surface candidates, not to convict.
- Sentence-rhythm advice from general writing guidance conflicts with a hard
  25-word ceiling. Academic prose cannot reach the length spread that blog prose
  can. The check reports this as context rather than as a defect.
- No detector reliably identifies AI-generated text or images. The checklists raise
  suspicion and direct attention; they do not prove authorship.
- A figure can be perfectly rendered, correctly captioned, and still show the wrong
  data. Only checking it against the source artifact catches that.
