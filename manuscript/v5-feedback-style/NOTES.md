# v5 — supervisor and reviewer feedback incorporation

Built from v4 prose style while implementing specific feedback requirements:
- Introduction: Contributions formatted as itemized bullet points with bold descriptive titles. Section roadmap added at the end of Introduction.
- Section 2 (Literature Survey): Restructured into:
  - 2.1 Clinical and staging approaches
  - 2.2 Statistical and machine learning models
  - 2.3 Limitations of existing models
- Section 5 (Discussion): Dedicated paragraphs for each contribution linking proposed mechanisms directly to empirical results, followed by a consolidated limitations paragraph.
- Section 6 (Conclusion): Expanded across three comprehensive paragraphs covering summary of findings, clinical translation pathways, and future research directions.

## Style and Verifier Compliance

Strict adherence to `.kiro/skills/research-paper` and `.kiro/skills/humanize`:
- Zero em dashes (`—`): Replaced with commas, parentheses, or structured phrasing.
- Zero first-person pronouns (`we`, `our`, `us`): Impersonal academic tone maintained throughout.
- Zero banned vocabulary words (`robust`, `furthermore`, `moreover`, `notably`, `delve`, `leverage`, `highlights`, etc.).
- Zero overstatements (`proven`, `superior to`, etc.).
- Sentence length cap: All sentences strictly <= 45 words.
- Acronyms expanded on first mention (including `DNA`, `AUC`, `ECOG`, `TNM`).
- British English spelling preserved (`tumour`, `regularisation`, `optimised`).

Verified clean:
- `build_paper.sh`: 0 errors, 0 undefined citations, 0 undefined references.
- `check_text.py`: 0 blockers, 0 warnings.
