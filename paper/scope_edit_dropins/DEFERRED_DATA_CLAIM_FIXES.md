# Deferred Data and Claim Fixes for `scope_edit_v0`

These are not applied in the scope-edit replacement files. They are the later
data/claim edits that should be handled piecemeal against the Overleaf draft.

- Replace HousingQA "parity" language with the audited framing: raw-question
  RAG remains the stronger HousingQA answer baseline, while generated-query
  prompt conditioning can still improve selected retrieval exposure.
- Soften Gold Evidence language: Snap/SCOPE exceeds Gold Evidence on Gemma 4
  26B and approximately matches it on Llama 3.3 70B.
- Qualify exemplar claims as `N=500` probe evidence. Do not say it lifts
  retrieval above raw search at every size.
- Keep Legal-Link-EU out of the main exemplar narrative unless it is explicitly
  framed as a boundary case.
- Use the corrected Figure 3 caption: HousingQA generated-query curves average
  over the two complete generated-query models, not all three model sizes.
- Use the corrected Table 4 Snap/SCOPE token-efficiency aggregate:
  2062 input tokens/q, 338 output tokens/q, 258.8 correct per million
  answer-stage tokens over five logged cells.
- Keep Appendix top-k bolding aligned with the caption, including the HousingQA
  mean HyDE row where it beats Snap/SCOPE.
- Fix bibliography placeholders before final upload, especially the Zheng venue
  placeholder and any invalid Legal RAG Bench metadata.

For full lineage and damage accounting, see:

- `paper/after_report/number_lineage.md`
- `paper/after_report/damage_report.md`
- `paper/after_report/internal_discrepancies_and_recommendations.md`
