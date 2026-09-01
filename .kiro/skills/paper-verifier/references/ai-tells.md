# AI tells in text and figures

No detector reliably identifies machine-written text or machine-generated images.
What follows raises suspicion and directs attention. It does not prove authorship,
and it should never be reported as if it did.

The useful framing for a research paper is not "was this written by a model" but
"does this sentence carry information a reader needs". Most AI-written academic
prose fails on that second question, which is why the checks below overlap heavily
with plain bad-writing checks.

---

## Text

### Caught automatically by check_text.py

Punctuation budgets, banned vocabulary, negation pivots, "turns out" framing,
repeated sentence openers, verbatim repeats, filler sentence shapes, sentence
length uniformity. Read the findings rather than re-deriving them.

### Needs a human read

**Uniform paragraph architecture.** Every paragraph opening with a thesis, followed
by two or three supporting sentences, closing on a restatement. Human academic
paragraphs are lumpier: one runs long because the argument needs it, another is two
sentences because that was all there was to say.

**Local coherence that is too smooth.** Every sentence connecting perfectly to the
next, with no thought that shifts direction and no aside. Reads frictionless and
slightly lifeless. The fix is not to insert artificial roughness but to let one
sentence per section do two jobs, or leave one implication unstated.

**Symmetric treatment of asymmetric things.** Three limitations each given one
sentence of equal weight, when one of them is far more serious. Real limitations are
lopsided. If the untested proportional hazards assumption matters more than the
spelling of a cohort code, the paragraph should show that.

**Contentless comparison.** "Comparable to published benchmarks", "in line with
prior work", "competitive performance", with no paper named and no number given.
This is the single most common tell in a machine-written results section, and it is
also a straightforward reviewer objection.

**Suspicious completeness.** Every subsection the same length, every table with the
same number of rows, every limitation neatly resolved by a proposed future work
item. Real research is uneven.

**Confident wrong terminology.** A method named for something it does not do,
described fluently. Cosine similarity described where Pearson correlation was
computed. Gram-Schmidt orthogonalisation described where a greedy correlation
screen was implemented. This one is worth checking against the source code, not
just against the text, because fluent prose hides it well.

**Numbers with no provenance.** A p-value, an effect size, or a metric that appears
in no result file. This is the most serious tell because it is also a fabrication.
The number-traceability stage catches it.

---

## Figures

### Caught automatically by check_assets.py

Resolution, stretching, vector against raster, duplicates, orphans, embedded fonts,
and whether a generating script exists.

### Checklist for every figure, applied by eye

Open the figure file itself, and also look at it as rendered on the page. Some
problems only show at print size.

**Text inside the image**
- [ ] Every word is a real word. Garbled or invented labels are the strongest
      single signal of a generated image.
- [ ] Axis labels present, with units where units apply.
- [ ] Tick labels legible at the size the figure appears on the page, not at the
      size of the source file.
- [ ] Legend entries match the plotted series in count and in order.
- [ ] Gene names, variable names and dataset codes spelled as in the text.
- [ ] Colorbar label describes the quantity actually plotted. A colorbar reading
      "Cosine Similarity" on a Pearson correlation matrix is a real defect found
      in this repository.

**Data integrity**
- [ ] Values in the figure match the same values in the text and tables.
- [ ] Number of points, bars or cells matches the stated sample size.
- [ ] Axis ranges do not truncate in a way that exaggerates a difference.
- [ ] A y-axis not starting at zero is either justified or fixed.
- [ ] Error bars, where present, are defined in the caption.
- [ ] No point sits outside the range the data allows.

**Style consistency**
- [ ] Same font family across all figures, ideally matching the body text.
- [ ] Same colour palette across figures for the same variable.
- [ ] Panel labels in one style, (a) (b) or A B, not both.
- [ ] No decorative gradient, drop shadow, bevel or 3D effect on a 2D quantity.
      These are common in generated images and rare in plotting library defaults.

**Provenance**
- [ ] A script in the repository produces this file.
- [ ] Running that script reproduces the figure.
- [ ] The script reads from a result artifact, not from values typed inline.

A figure that fails the provenance check should be regenerated from data before
submission, whatever it looks like.

---

## Reporting

Say what was observed, not what it implies about authorship.

Write: "The colorbar in Figure 2 reads Cosine Similarity while the caption and
Section 3.2 describe a Pearson correlation. One of the two is wrong."

Not: "Figure 2 appears AI-generated."

The first is actionable and correct. The second is an accusation that cannot be
supported and that tells the author nothing about what to fix.
