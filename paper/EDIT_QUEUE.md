# Damage-Control Edit Queue for Scope Draft

This queue is for later piecemeal edits to `paper/scope_edit_v0.zip`. The
current Overleaf replacement pack only adds Background and Related Work; it
does not repair the data/claim issues below.

## Highest Priority Claim Repairs

| Claim area | Problem | Use this framing |
|---|---|---|
| HousingQA answer accuracy | Scope draft still has parity-style language. Table 1 favors raw-question RAG on complete HousingQA answer rows. | "Raw-question RAG remains the stronger HousingQA answer baseline; generated-query prompt conditioning can still improve selected retrieval exposure." |
| Gold Evidence comparison | "matches or exceeds" is too strong for Llama 3.3 70B because the gap is 79.7 vs 79.2. | "Exceeds Gold Evidence on Gemma 4 26B and approximately matches it on Llama 3.3 70B." |
| Exemplar probe | The draft can imply a broad or every-size improvement. The evidence is an `N=500` Gemma 26B prompt diagnostic. | "A single sanitized corpus example improves `N=500` Gemma 26B retrieval probes, especially HousingQA, with paired answer accuracy statistically unchanged." |
| Legal-Link-EU exemplar | Older drafts mention Legal-Link-EU in the main exemplar story, but raw retrieval is much stronger there. | Omit from the main exemplar narrative unless explicitly framed as a boundary case. |
| Figure 3 caption | Older wording says HousingQA averages across three model sizes. | HousingQA generated-query curves average over the two complete generated-query models, Llama 3.1 8B and Llama 3.3 70B. |
| Table 4 efficiency | Older value says Snap-HyRE/SCOPE has 2001 input tokens/q, 376 output tokens/q, 268.3 correct/M over four cells. | Use 2062 input tokens/q, 338 output tokens/q, 258.8 correct/M over five logged cells. |
| Appendix top-k bolding | HousingQA mean HyDE row must be bold when it beats Snap/SCOPE. | Bold HyDE in the HousingQA mean row for Hit@3, Hit@5, Hit@10, and MRR@10. |
| Bibliography | Zheng venue placeholder and Legal RAG Bench metadata need checking before final upload. | Use verified venue/metadata only; do not leave placeholder text. |

## Likely Locations in `scope_edit_v0`

- `sections/0Abstract.tex`: Gold Evidence, HousingQA, exemplar scope.
- `sections/1Introduction.tex`: HousingQA parity framing and main contribution wording.
- `sections/5Analysis.tex`: HousingQA interpretation, Figure 3 caption, Table 4 prose, exemplar prose.
- `sections/6Conclusion.tex`: HousingQA parity wording after the Related Work insertion.
- `sections/Appendix.tex`: coverage wording, exemplar details, top-k table interpretation.
- `tables/current_usage_metrics.tex`: token-efficiency table if the draft table is stale.
- `tables/topk_retrieval_summary.tex`: appendix bolding.
- `references.bib`: bibliography placeholders.

## Sources of Truth

- `paper/after_report/number_lineage.md`: number-to-source map.
- `paper/after_report/repro_bundle/source_file_manifest.csv`: raw file checksums.
- `paper/after_report/repro_bundle/answer_log_summaries.csv`: compact answer summaries.
- `paper/after_report/repro_bundle/retrieval_cache_summaries.csv`: compact retrieval summaries.
- `paper/after_report/damage_report.md`: stale claim to corrected claim mapping.
