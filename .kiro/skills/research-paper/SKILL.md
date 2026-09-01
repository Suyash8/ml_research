---
name: research-paper
description: >
  Use whenever the user is writing, rewriting, restructuring, or reviewing an academic
  research paper, manuscript, thesis chapter, abstract, or conference submission. Also use
  when a supervisor, reviewer, or faculty member has returned comments on a draft, when the
  user says a paper is "too complex", "too dense", "unreadable", "rejected", or "needs
  references", and when the user asks to simplify academic prose, expand acronyms, enforce
  passive voice, verify citations, or check that every claim in a draft is backed by a real
  source. Trigger it for LaTeX manuscripts even when the request sounds like a formatting
  task, because structure, language, and citation discipline in a paper are one problem.
---

# Research Paper Writing

A paper fails review for three reasons far more often than for weak results: a reader cannot
follow the sentences, a claim has no source behind it, or a number in the text does not match
the number in the output files. This skill exists to make those three failures impossible.

The audience assumption that drives every rule below: **a first-year undergraduate with no
background in the field must be able to read the paper and understand what was built, what was
measured, and why it matters.** That person is not stupid. They are unfamiliar. Write for
unfamiliarity, not for low intelligence, and the domain expert gets a better paper too.

---

## Hard rules

These are the ones that get caught in review. Check them last, against the actual draft text,
by counting. Not from memory.

1. **No first person.** No "we", "our", "us", "I", "my". Passive voice or an impersonal
   subject instead. "We applied X" becomes "X was applied". "Our model" becomes "the model"
   or "the proposed model". This includes the abstract and the conclusion, where it leaks back
   in most often.
2. **Sentence length ceiling.** No sentence longer than 25 words. Aim for a 15-word average.
   If a sentence covers two ideas, it is two sentences. Verify by counting words in the
   longest-looking sentences, not by reading for flow.
3. **Every acronym expanded at first use, without exception.** Write the full form, then the
   acronym in parentheses, then use the acronym. This applies to acronyms you think everyone
   knows: RNA, DNA, PCA, AUC, CT, MRI, GPU, API, TCGA. Expand once per paper, in the main
   text, at the earliest occurrence. The abstract counts as a separate scope: if an acronym
   appears in the abstract, expand it there and again at its first occurrence in the body.
   Verify the full form is correct. A wrong expansion is worse than none.
4. **Every claim carries a citation.** If a sentence asserts something that is not a result of
   this work, it needs a reference. Statements about biology, prior methods, clinical practice,
   what is standard, what is difficult, what other papers found: all of it. The only sentences
   that may go uncited are descriptions of what was done in this study and numbers measured in
   this study.
5. **Every number traces to an artifact.** Before writing any figure into the text, confirm it
   in the result files. Quote the file path in your working notes. A number that cannot be
   traced gets deleted, not rounded or hedged.
6. **No empty sentences.** Every sentence must add a fact, a number, a mechanism, or a
   consequence. Delete anything that only restates the previous sentence, announces what comes
   next, or gestures at importance without content. Test: cover the sentence and ask whether
   the paragraph lost information. If not, it was filler.
7. **No subsections inside the Introduction.** The introduction is continuous prose. Motivation,
   background, existing work, gap, and contributions each get their own paragraph or two, joined
   by argument, not by headings.

---

## Language rules

**Word choice.** Use the shortest word that is still precise. Precision wins over simplicity
when they conflict, but they rarely conflict; dense academic vocabulary is usually habit, not
precision.

| Instead of | Write |
|---|---|
| utilize, leverage, employ | use |
| demonstrate, exhibit | show |
| facilitate, enable | allow, help |
| prior to, subsequent to | before, after |
| in order to | to |
| due to the fact that | because |
| a plethora of, a myriad of | many |
| methodology (when you mean method) | method |
| paradigm, framework (when vague) | name the actual thing |
| elucidate | explain |
| aforementioned | this, that, the |
| heterogeneity | variation, differences |
| ameliorate | improve |

**Banned entirely**: delve, robust, comprehensive, streamline, novel (as self-praise), pivotal,
crucial, nuanced, multifaceted, intricate, landscape (figurative), tapestry, testament,
showcase, furthermore, moreover, "it is important to note", "it is worth noting", "in today's
world", "state-of-the-art" as a bare adjective.

**Jargon gets defined on first use.** Not just acronyms. If the paper says "right censoring",
"partial likelihood", "collinearity", or "hazard", one short sentence must say what that means
in plain words before the term does any work. One sentence. Not a tutorial.

**Punctuation.** Semicolons are acceptable in academic register but rarely needed; prefer a
period. Em dashes: at most one per 300 words, and only for a genuine interruption. Straight
quotes and apostrophes only.

**Sentence rhythm.** Vary length. A run of five 20-word sentences is exhausting even when each
one is legal. Follow a long explanatory sentence with a short one that lands the point.

---

## Structure

Use this section order unless the venue dictates otherwise:

```
Title
Abstract
Keywords
1. Introduction          (continuous prose, no subsections)
2. Related Work          (subsections allowed, grouped by approach not by paper)
3. Method                (subsections expected, one per pipeline stage)
4. Results               (subsections expected, one per claim family)
5. Discussion            (subsections allowed: interpretation, then limitations)
6. Conclusion            (one paragraph, no subsections)
Declarations
References
```

### Title

Short. Descriptive. No stacked adjectives. Every word must be doing work.

A title that lists every property of the method reads as insecurity. Pick the one or two
things the paper is actually about and name them. If a reader can guess the domain and the
approach from the title, it is long enough.

**Example 1:**
Before: A Calibrated, Explainable Cox Elastic-Net Pipeline for Pan-Cancer Survival Prediction from Multi-Omic TCGA Data
After: Interpretable Survival Prediction from Gene Expression and Clinical Data

**Example 2:**
Before: A Novel Attention-Augmented Multi-Scale Convolutional Architecture for Robust Real-Time Semantic Segmentation of Urban Street Scenes
After: Real-Time Semantic Segmentation of Urban Street Scenes

### Abstract

Roughly 200 to 250 words, no citations, no acronym left unexpanded. One sentence per job:
the problem, what was built, the data, the headline numbers, what the numbers mean. Lead with
the problem, not with a claim of novelty. Put the strongest verified number in the abstract.

### Introduction

Five moves, in order, as separate paragraphs with no headings:

1. **Motivation.** The real-world problem. Concrete, cited, no throat-clearing.
2. **Background.** The standard approach and why it exists. Define the core terms here.
3. **Existing work.** What has been tried, grouped by idea. What each family gets right.
4. **The gap.** What none of them do. State it plainly, one gap per sentence.
5. **Contributions.** What this paper adds. A short list is acceptable here, and only here.

The paragraphs must connect. Each one should end on something the next one picks up. If the
paragraphs could be shuffled without loss, the argument is not built yet.

Do not write an "architecture overview" paragraph in the introduction unless a figure is
actually being introduced. Do not promise a roadmap of the paper's sections.

### Method

One subsection per stage, in the order the data moves through them. Describe what the code
actually does, verified by reading the code, not what the method is conventionally called.
Give the concrete parameter values and the reason for each choice. Name the failure mode that
a choice avoids, when there is one.

### Results

Report the number, then say what it means. No interpretation in the Results beyond what the
number directly supports; save the argument for the Discussion. Every table and figure gets
referenced in the text, and the text says what the reader should notice.

### Discussion and limitations

Limitations get their own paragraph and are specific. "May not generalize" is not a
limitation, it is a hedge. "The proportional hazards assumption was not tested with Schoenfeld
residuals, and is likely violated across the four cancer types given their different
progression timelines" is a limitation.

---

## Citation discipline

**Verify before citing.** For each reference, confirm the title, the full author list, the
year, the venue, and the identifier resolve to one real publication. Search for it. Do not
reconstruct a citation from memory of the field: author lists are exactly where fabrication
hides, because the title and journal look right while three of the five names do not exist.

**Match the claim to the source.** A citation must support the specific sentence it is attached
to. A survival-analysis methods paper cannot be cited for a claim about tumour staging capturing
anatomy. This kind of drift is common and reviewers catch it.

**Cite the primary source.** For a method, cite the paper that introduced it, not a textbook
that mentions it and not a later paper that uses it. The Cox model gets Cox. The Kaplan-Meier
estimator gets Kaplan and Meier. Add a textbook alongside only when a reader needs a tutorial.

**Delete what you cannot verify.** If a reference cannot be confirmed to exist, remove it and
remove or rewrite the sentence that depended on it. An uncited weaker claim beats a cited
fabrication. Report every deletion to the user with the reason.

**Statistical language must match the test that was run.** A univariate screen does not show
that a variable is "independently" prognostic; that needs a multivariable model. "Genome-wide
significance" means the genome-wide threshold, not "a small p-value". A corrected q-value is
not a p-value. Getting this wrong invites a desk rejection.

---

## Working protocol

1. **Read the whole current draft** and the reviewer or supervisor comments before changing
   anything. Map each comment to the text it targets.
2. **Read the result artifacts.** Build a table of every number the paper claims and the file
   it came from. Numbers that fail this step are the first thing to fix, because they are the
   only unrecoverable error.
3. **Read the source code for anything the method section names.** Verify the described
   algorithm is the implemented algorithm. Misnamed methods are a frequent and serious finding.
4. **Verify every reference** by search. Fix, replace, or delete.
5. **Write the draft.** Structure first, then prose. Apply the language rules while writing,
   not as a cleanup pass; a cleanup pass on complex prose produces simplified complex prose.
6. **Run the checklist below by counting**, against the finished text.
7. **Compile**, if the source is LaTeX. Confirm zero undefined references and zero undefined
   citations in the log. A clean compile is part of the deliverable.
8. **Report to the user**: what changed, which numbers were corrected, which references were
   removed and why, and what still needs a human decision.

---

## Pre-delivery checklist

Go through the actual text and write the count for each item. An unenumerated "looks fine" is
a check you did not perform.

- [ ] First-person pronouns: search "we", "our", "us", " I ", "my". Count must be 0.
- [ ] Sentences over 25 words: count them, quote them, split them.
- [ ] Acronyms: list every acronym in the paper, and for each, quote its first occurrence and
      confirm the expansion is present and correct.
- [ ] Uncited claims: read each sentence and mark whether it is (a) this work, (b) a measured
      result, or (c) a claim needing a citation. Every (c) has one.
- [ ] Numbers: every figure in the text matched to a result file. List the pairs.
- [ ] Banned vocabulary: scan the list above, quote each hit, replace it.
- [ ] Empty sentences: for each paragraph, name the sentence that carries the least
      information and justify keeping it or cut it.
- [ ] Introduction has no subsection headings.
- [ ] Every table and figure is referenced from the text.
- [ ] LaTeX log: zero undefined citations, zero undefined references.

Three or more failures in one category means the category was not really checked. Go again.

---

## Interaction with the humanize skill

When both skills are active, this one wins on register: academic prose keeps its formality,
its passive voice, and its citations. Take from `humanize` the parts that do not conflict:
sentence-length variance, banned vocabulary, punctuation normalization, cutting
assistant-voice filler and significance inflation. Ignore its guidance on first-person voice,
contractions, rhetorical questions, and casual register. Those belong to blog posts.
