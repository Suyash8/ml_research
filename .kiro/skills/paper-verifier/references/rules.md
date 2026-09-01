# Rule catalogue

Every rule the verifier enforces, with the reason. The reason matters: a rule
applied without understanding produces mechanical edits that satisfy the letter and
miss the point.

Rules marked **[auto]** are counted by a script. Rules marked **[judge]** need the
agent. Rules marked **[both]** are counted approximately and confirmed by hand.

---

## 1. Reader accessibility

The target reader is a first-year undergraduate with no background in the field.
That person is unfamiliar, not unintelligent. Writing for unfamiliarity produces a
better paper for the specialist too, because the specialist is also reading at
speed and also skips anything dense.

**1.1 No sentence over 25 words. [auto]**
A four-line sentence cannot be held in working memory. If a sentence covers two
ideas it is two sentences.

**1.2 Mean sentence length under 16 words. [auto]**
A paper where every sentence is legal at 24 words is still exhausting.

**1.3 Every acronym expanded at first use, without exception. [auto]**
Including RNA, DNA, PCA, AUC, CT, MRI, GPU, API. The author knows which acronyms
are common in their field; the reader from an adjacent field does not. The abstract
counts as its own scope: expand there, and again at the first body occurrence.
Verify the expansion is correct. A wrong expansion is worse than none.

**1.4 Every technical term glossed in plain words before it does any work. [both]**
One sentence, not a tutorial. Hazard, right censoring, partial likelihood,
collinearity, one-hot encoding, loading matrix. The script checks for a gloss near
the term; whether the gloss is any good needs a human.

**1.5 Shortest precise word wins. [auto]**
Utilise becomes use. Demonstrate becomes show. Prior to becomes before. Dense
academic vocabulary is usually habit rather than precision.

**1.6 No empty sentences. [both]**
Every sentence carries a fact, a number, a mechanism or a consequence. Test: cover
it and ask whether the paragraph lost information. The script flags known filler
shapes; borderline cases need judgement, and a short sentence is often doing
deliberate rhythmic work.

---

## 2. Voice and register

**2.1 No first person. [auto]**
No we, our, us, I, my. Passive voice or an impersonal subject. This leaks back into
the abstract and conclusion most often, because those get rewritten last.

**2.2 Past tense for completed work. [auto]**
The experiment is finished. "The model reached 0.736", not "reaches". A table is an
exception: "Table 3 reports" is correct, because the table still does that.

**2.3 One spelling convention. [auto]**
British or American, applied throughout, or matching the venue. Mixing tumour and
tumor in one paper is the kind of thing a reviewer notices and remembers.

**2.4 Consistent precision per metric. [auto]**
A concordance index quoted as 0.736 in one place and 0.74 in another looks careless.
A value quoted from another paper keeps that paper's precision and is exempt.

---

## 3. Structure

**3.1 No subsections in the Introduction. [auto]**
Motivation, background, existing work, gap, contributions. Five moves, separate
paragraphs, joined by argument rather than by headings. If the paragraphs could be
shuffled without loss, the argument is not built yet.

**3.2 Conventional section order. [auto]**
Introduction, Related Work, Method, Results, Discussion, Conclusion. A reviewer
navigates by expectation.

**3.3 Short title. [auto]**
Twelve words or fewer, at most two qualifier adjectives. A title listing every
property of the method reads as insecurity. Pick what the paper is about.

**3.4 No heading that promises content it does not deliver. [judge]**
A section called "Architecture Overview" must contain an architecture overview.

**3.5 Declarations present. [auto]**
Funding, competing interests, data availability. Most venues require all three.

**3.6 No placeholders. [auto]**
No `[Insert X here]`, no `xyz`, no TODO. Obvious, and it still reaches reviewers.

**3.7 Every float referenced from the text, and every reference resolving. [auto]**
An unreferenced figure is either unnecessary or the text forgot to use it.

**3.8 Figure captions readable standalone. [auto]**
Readers look at figures before reading the body. Eight words minimum for a figure,
four for a table. Subcaptions like "Before filtering" are labels and are exempt.

---

## 4. Citations

**4.1 Every non-result claim carries a citation. [both]**
Statements about biology, prior methods, clinical practice, what is standard, what
is difficult. Only descriptions of this work and numbers measured in this work may
go uncited.

**4.2 Every reference verified against the publisher record. [judge]**
Title, full author list, year, venue, volume, pages. Check the author list name by
name. Fabricated references are plausible: real title, real journal, right year,
and two authors who do not exist. Every structural check passes.

**4.3 The citation supports the specific sentence. [judge]**
A survival-analysis methods paper cannot support a claim about tumour staging
capturing anatomy. This drift is common and reviewers catch it.

**4.4 Primary sources for methods. [judge]**
The Cox model gets Cox 1972. Kaplan-Meier gets Kaplan and Meier. A textbook goes
alongside, not instead.

**4.5 Delete what cannot be verified. [judge]**
Remove the entry, then remove or rewrite the dependent sentence. An uncited weaker
claim beats a cited fabrication. Report every deletion with its reason.

**4.6 Every entry carries a DOI, URL or ISBN. [auto]**
Without an identifier the entry cannot be checked mechanically, which is exactly
where an invented reference survives.

**4.7 No truncated author lists. [auto]**
"and others" or "et al." in a bib field means the list was never checked.

**4.8 No orphan entries. [auto]**
An uncited entry is usually a leftover from a claim that was removed, and it is
worth asking whether the claim should have gone.

---

## 5. Numbers

**5.1 Every number traces to a result artifact. [auto]**
Confirm before writing it down. Delete what cannot be traced. Do not round or hedge
it. This is the only error class that cannot be fixed after publication.

**5.2 Abstract and conclusion numbers match the body exactly. [auto]**
Rounding a metric up on the way into the abstract is a classic reviewer catch.

**5.3 Internal arithmetic holds. [auto]**
Split sizes sum to the cohort. "441 of 500 (88.2%)" divides correctly. Ranges run
low to high.

**5.4 Statistical language matches the test performed. [judge]**
A univariate screen does not show independent effect; that needs a multivariable
model. "Genome-wide significance" means the genome-wide threshold, not a small
p-value. A corrected q-value is not a p-value. Getting this wrong invites a desk
rejection.

**5.5 No p reported as zero, no probability outside [0,1]. [auto]**
Report a bound: p < 0.001.

**5.6 Means carry a spread. [auto]**
A mean with no standard deviation or interval cannot be interpreted.

---

## 6. Honest claims

**6.1 No unsupported superlatives. [auto]**
State-of-the-art, outperforms, best, unprecedented, groundbreaking, proven,
guaranteed. Each needs either a cited comparison or deletion.

**6.2 "Significant" only next to a test. [auto]**
In a paper the word has a technical meaning. Using it to mean "large" is a
word-choice error that a statistical reviewer will read as a claim.

**6.3 Comparisons name the baseline. [judge]**
"Comparable to published benchmarks" without naming one is not a comparison. State
the paper, the number, and what differs between the settings.

**6.4 Novelty claims are checkable. [auto]**
"The first to" invites a reviewer to find a counterexample, and they usually can.

**6.5 Limitations are specific and complete. [judge]**
"May not generalise" is a hedge, not a limitation. Name the untested assumption,
the boundary condition, the hyperparameter that sat at the edge of its grid. A
limitations section that omits a known weakness is worse than none, because the
reviewer will find it and now doubts everything else.

---

## 7. Figures

**7.1 Minimum 300 ppi effective resolution. [auto]**
Measured in the PDF, not in the source file. A 3000-pixel image can still be under
300 ppi if it spans a full page.

**7.2 No stretching. [auto]**
Both width and height set without keepaspectratio distorts. Unequal horizontal and
vertical ppi in the PDF proves it happened.

**7.3 Vector for line art. [auto]**
Diagrams and plots as PDF or EPS stay sharp at any zoom.

**7.4 Every figure reproducible from a script in the repository. [auto]**
A figure with no generating script was hand-drawn, pasted, or generated. All three
need a human decision.

**7.5 Figure content agrees with its caption and the body text. [judge]**
Axis labels, legends and colorbars must describe what is actually plotted. This is
found only by looking at the rendered page.

**7.6 Text in figures legible at print size. [judge]**
Check the page raster, not the source file.

**7.7 Fonts embedded. [auto]**
An unembedded font renders differently on the reviewer's machine.

---

## 8. Rule conflicts

Two conflicts come up repeatedly. Resolve them the same way each time.

**Sentence variety against the 25-word ceiling.** General writing advice asks for a
20-word spread between the longest and shortest sentence. A 25-word ceiling makes
that arithmetically hard. The ceiling wins, because comprehension matters more than
rhythm. The verifier reports rhythm as context rather than as a defect.

**Plain language against precision.** Precision wins. They rarely actually
conflict; when they seem to, the usual cause is a technical term with no gloss
rather than a term that needed to be complicated.
