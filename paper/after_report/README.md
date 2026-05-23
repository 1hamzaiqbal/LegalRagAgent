# After-Report Index

This folder is a post-audit bundle for comparing the stale
`paper/icml_submission.pdf` against the corrected `paper/main.pdf`, tracing the
reported data to JSONL/cache files, and recording what should be trusted.

## Files

- [`number_lineage.md`](number_lineage.md): row-by-row mapping from paper
  numbers to answer-detail JSONL files and retrieval-cache JSONL files.
- [`damage_report.md`](damage_report.md): what was wrong in
  `icml_submission.pdf` and what the final paper changed it to.
- [`data_generation_and_reproducibility.md`](data_generation_and_reproducibility.md):
  exactly what generated the final data, which Python/script path is usable,
  and what cannot currently be replicated exactly.
- [`internal_discrepancies_and_recommendations.md`](internal_discrepancies_and_recommendations.md):
  remaining source-package risks, acceptable caveats, and next cleanup steps.
- [`scripts/regenerate_figure3_from_final_csv.py`](scripts/regenerate_figure3_from_final_csv.py):
  saved script for regenerating Figure 3 from the final package CSV.
- [`plots/`](plots/): regenerated Figure 3 and rendered page snapshots used for
  visual comparison.
- [`tables/`](tables/): copied final table `.tex` files and source CSVs used by
  the number-lineage report.

## Plot Assets

- [`plots/figure3_topk_retrieval_curves_regenerated.png`](plots/figure3_topk_retrieval_curves_regenerated.png)
- [`plots/icml_submission_page-01.png`](plots/icml_submission_page-01.png):
  stale PDF page 6, where Table 1/2/Figure 3 and stale exemplar text collide.
- [`plots/icml_submission_page-02.png`](plots/icml_submission_page-02.png):
  stale PDF page 7, with stale Table 4/5 and Legal-Link-EU in main text.
- [`plots/main_page-01.png`](plots/main_page-01.png):
  final PDF page 6, with corrected Figure 3 caption and no stale exemplar text.
- [`plots/main_page-02.png`](plots/main_page-02.png):
  final PDF page 7, with corrected Table 4/5 and final exemplar framing.
- [`plots/main_appendix_topk_page-11.png`](plots/main_appendix_topk_page-11.png):
  final PDF page 11, with corrected Appendix top-k table bolding.

## Trust Decision

Use `paper/main.pdf` and `paper/FINALFINALVERSION.zip` as the submission
artifacts. Treat `paper/icml_submission.pdf` as a stale comparison artifact for
audit comparison only.
