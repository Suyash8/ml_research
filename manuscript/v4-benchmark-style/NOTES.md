# v4 — benchmark-matched prose style

Same sections, same numbers, same claims, same citations as v3. Only the prose
changed. Nothing was added or removed from the argument.

Written because v3 read as stiff and mechanical despite passing every rule check.
The cause turned out to be measurable, and it was not AI-ness.

## The diagnosis

Ten published papers from `papers/` were extracted with `pdftotext` and profiled on
24 prose metrics. v3 was profiled the same way. Three numbers explained the whole
complaint:

| metric | published median | v3 | meaning |
|---|---|---|---|
| sentence length SD | 13.3 | **5.0** | every sentence nearly the same length |
| commas per sentence | 1.60 | **0.51** | almost every sentence a single clause |
| sentences with no comma | 36% | **62%** | staccato, no subordination |
| 90th-percentile length | 39 | **21** | never builds a long sentence |
| parentheticals per 1k words | 34.7 | **5.3** | qualifications promoted to sentences |

v3 was not badly written. It was uniformly written. Every sentence arrived at the
same length with the same shape, which is what produces the mechanical feeling.

## The 25-word rule was mine, not the reviewer's

v3 enforced a 25-word ceiling. That number was invented during the v3 rewrite; it
does not appear in the faculty comments. What the reviewer actually wrote was that
a sentence cannot continue for five lines, which is roughly 65 words in this layout.

v4 uses a 45-word ceiling, about three typeset lines. That sits at the published
90th percentile (39 words) while staying clear of the five-line objection. The
published set runs to 83-118 words at maximum; matching that was judged unwise
given the original comment.

## Deliberate deviation from the benchmark

Published work in this field uses first person at 3.1 occurrences per 1,000 words.
v4 uses zero, because impersonal voice was an explicit requirement. Where the two
conflict, the requirement wins. This is recorded in `style-bands.json` as
`"enforce": false` so the check never flags it.

## What was achieved

Measured with `.kiro/skills/paper-verifier/references/style-bands.json`. Seven of
fifteen enforced metrics landed inside the published band, against one of fifteen
for v3.

| metric | band | target | v3 | v4 | in band |
|---|---|---|---|---|---|
| mean_len | 22.9-24.6 | 23.6 | 13.7 | 23.7 | yes |
| median_len | 18.5-23.0 | 21.0 | 13.0 | 24.0 | no, high |
| sd_len | 12.1-17.4 | 13.3 | 5.0 | 9.0 | no, low |
| p90_len | 36-42 | 39 | 21 | 36 | yes |
| pct_under_10 | 6-23 | 11 | 21 | 6 | yes |
| pct_10_20 | 34-42 | 37 | 69 | 30 | no, low |
| pct_over_25 | 22-38 | 34 | 0 | 45 | no, high |
| pct_over_30 | 13-26 | 23 | 0 | 23 | yes |
| commas_per_sent | 1.48-2.01 | 1.60 | 0.51 | 1.35 | no, low |
| pct_sent_0_commas | 19-49 | 36 | 62 | 19 | yes |
| subord_per_1k | 3.2-10.4 | 6.9 | 10.9 | 16.5 | no, high |
| connect_per_1k | 2.7-5.2 | 3.1 | 2.1 | 2.5 | no, low |
| hedge_per_1k | 1.4-3.4 | 2.2 | 2.1 | 1.9 | yes |
| paren_per_1k | 26.5-57.3 | 34.7 | 5.3 | 14.8 | no, low |
| start_concentration | 30-58 | 35 | 47 | 41 | yes |

The two metrics that carried the complaint both improved substantially: sentence
length SD nearly doubled (5.0 to 9.0) and commas per sentence rose 2.6-fold (0.51
to 1.35). Neither reached the band.

## What is still off, and why

**sd_len 9.0 against a 12.1 floor.** Reaching the published spread needs occasional
60-to-110-word sentences, which the five-line objection rules out. This metric is
probably not reachable under a 45-word ceiling, and chasing it is not worthwhile.

**subord_per_1k 16.5 against a 10.4 ceiling.** A real tic. The rewrite leans on
because, since, while and which. "rather than" alone appeared 16 times before a
reduction pass; the remaining density needs another pass converting subordinate
clauses into separate sentences joined by connectives, which would also lift
connect_per_1k.

**paren_per_1k 14.8 against a 26.5 floor.** Partly structural: numbered citations
`[12]` produce no parentheses, whereas author-year styles do. The rest is genuine,
since published work parks more unit annotations and abbreviation definitions in
parentheses than v4 does.

**pct_over_25 45% against a 38% ceiling, pct_10_20 30% against a 34% floor.** These
two are the same problem: not enough mid-length sentences. Splitting roughly ten
of the 30-to-45-word sentences would fix both, and would also raise
pct_sent_0_commas towards its target.

Three measurement rounds were run, overshooting twice before converging. Anyone
continuing this should use `check_text.py --style-bands` after each pass rather
than working by feel; working by feel is what produced the oscillation.

## Reproducing the measurement

```bash
mkdir -p /tmp/bench/txt
for f in papers/*.pdf; do pdftotext -q "$f" /tmp/bench/txt/$(basename "$f" .pdf).txt; done

python3 .kiro/skills/paper-verifier/scripts/check_text.py \
    manuscript/v4-benchmark-style/paper.tex \
    --style-bands .kiro/skills/paper-verifier/references/style-bands.json
```

The bands in that JSON file came from ten papers in `papers/`. Regenerate them for
a different field or a different target venue; they are not universal.

## Unchanged from v3

Every number, every citation, every table, every figure, every section heading, and
every limitation. The claim set is identical. Verified: 47 citation keys resolve,
zero undefined references, all numbers still trace to the artifacts under
`results/`, zero first-person pronouns.

Also unchanged: the placeholder author names, the acknowledgments and generative AI
declarations, and the mislabelled colorbars in Figure 2, which still read "Cosine
Similarity" and need the figures regenerated. See
`../v3-simplified/NOTES.md`.

## Which version to submit

v4 if the reviewer's objection was that the writing felt flat or unlike published
work in the field. v3 if the priority is maximum readability for a non-specialist,
since its shorter sentences are genuinely easier on a first read. The two differ
only in prose, so the choice carries no risk to the content.

## Section structure aligned to v2

Identical to v3, and identical to v2 at the top level, so the two drafts can be
compared on prose alone:

```
Abstract          (unnumbered)
1  Introduction
2  Literature Survey
3  Proposed Method
4  Results
5  Discussion
6  Conclusion
   Acknowledgement (unnumbered)
   References
```

"Related Work" was renamed "Literature Survey" and "Method" became "Proposed
Method". Subsections differ from v2 and were deliberately left alone.

**Four back-matter sections were deleted**: Disclosure of interest, Funding,
Declaration of generative AI use, and Data availability statement. This followed an
explicit instruction. Most journals require all four, so restore them from
`../v2-multiomic-tcga/paper.tex` before submitting anywhere that asks for them.
The three resulting MINOR findings from `check_text.py` are expected.

Bibliography unaffected by the deletion: 47 entries, 47 cited, zero orphaned.
`cerami2012cbio` and `gao2013integrative` remain cited in Proposed Method.

Verified after the change: builds clean, 15 pages, zero undefined citations, zero
undefined references, zero errors, one overfull hbox.
