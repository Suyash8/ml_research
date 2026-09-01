# v2 — second draft, returned with comments

Archived. Do not edit.

Was `paper_new.tex` at the repository root. This is the draft that was submitted
to the faculty reviewer and returned as not acceptable.

Title: A Calibrated, Explainable Cox Elastic-Net Pipeline for Pan-Cancer Survival
Prediction from Multi-Omic TCGA Data

Changes over v1: an expanded literature survey with a comparison table, a
research gap subsection, and reworked discussion.

## Contents

- `paper.tex` — source, moved unchanged from `paper_new.tex`
- `paper.pdf` — the compiled version that was reviewed
- `references.bib` — frozen copy of the bibliography as it stood for this draft
- `build/` — LaTeX intermediates, filenames still carry the `paper_new` stem
- `images` — symlink to `../shared/figures`

## Reviewer comments

Recorded here because they drove the v3 rewrite. Annotations were made on the
PDF pages covering the title through Section 2.2.

- Title: "Calibrated" circled out, "from Multi-Omic TCGA Data" marked extra and
  redundant.
- Introduction: subsections should be replaced by flowing paragraphs covering
  motivation, background, existing works and contributions.
- "TNM classification" marked as needing its full form.
- Several sentences marked too long, with the note that a sentence cannot run for
  five lines.
- The paragraph on the Cox model marked "what is this?? Rephrase".
- "Why the description here?" against the P versus N discussion.
- The Cancer Genome Atlas and tumour mutational burden passages marked as needing
  references.
- "Four concrete gaps motivate this work" marked as a long sentence to break up.
- Section 1.2 heading: "Where is architecture overview?? No need of subheadings."
- Section 2.1: "Full form is wrong" against LIHC, annotated HCC. "Check the
  abbreviated form."

## Problems found later

Two errors in this draft were found while rebuilding v3, and were not in the
reviewer comments:

1. The tumour mutational burden significance result cannot be traced to any
   result file. See `../v3-simplified/NOTES.md`.
2. The collinearity filter is described as an incremental Gram-Schmidt filter
   using cosine similarity. The implementation is a greedy absolute-Pearson
   correlation screen. None of the three named properties are present in the
   code.

Superseded by v3.
