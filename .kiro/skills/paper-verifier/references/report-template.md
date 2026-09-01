# Verification report format

Group by severity, not by check. The reader wants to know what to fix first, and
they will not reorder the list themselves.

Every finding needs a location precise enough to act on without searching: a
section name, a page number, or a quoted phrase. "Some sentences are too long" is
not a finding. "Section 3.4, the sentence beginning 'The Cox model assigns', 34
words" is.

---

## Template

```
# Verification report: <paper> (<version>)

Verified on <date>. Build: <clean / N undefined citations>.

Coverage
  pages inspected visually    15 of 15
  references verified online  47 of 47
  numbers traced to artifacts 130 of 130
  scripts run                 text, assets, numbers, bibliography

Totals: N blocker, N major, N minor

## BLOCKER

1. <what is wrong> — <location>
   <why it cannot be sent, in one or two sentences>
   Fix: <the specific change>

## MAJOR

...

## MINOR

...

## Judgement calls

Findings the scripts raised that were reviewed and dismissed, with the reason.
This section exists so nobody re-litigates them next round.

## Not verified

What could not be checked, and why. Be specific. "No network access, so 47
references remain unverified" is useful. "Some items could not be checked" is not.
```

---

## Rules

**Lead with the coverage table.** A report with no coverage table implies a complete
pass. If 12 of 47 references were checked, the reader must see 12 of 47 before they
see the findings.

**Report the untraceable number first.** Every other class of error can be fixed
after publication with an erratum. A fabricated number cannot.

**One finding per defect.** Do not bundle "several long sentences and some passive
voice issues" into one bullet. Each gets its own line with its own location.

**Quote enough to locate, not more.** Eight to fifteen words of the offending text.
Not the whole paragraph.

**Give the fix, not just the diagnosis.** "Sentence too long" is half a finding.
"Split after 'partial log-likelihood'" is a whole one.

**Record dismissals.** A finding reviewed and judged acceptable goes in Judgement
calls with the reason. Otherwise the next run raises it again and someone spends
the same twenty minutes reaching the same conclusion.

**Do not pad.** If the paper has three problems, report three. A report inflated to
look thorough trains the reader to skim, and then they skim past the blocker.

**Separate defects from preferences.** "This sentence is 34 words, the ceiling is
25" is a defect. "This paragraph would read better reordered" is a preference. Mark
which is which, and never inflate a preference to MAJOR.
