# Paper Fast-Edit Handoff

This is the index for continuing the Overleaf scope edit without coming back to
the repo for missing context.

## Upload/Edit Artifacts

- `paper/scope_edit_overleaf_replacements.zip`: upload this into Overleaf and
  overwrite existing files. It changes only `sections/2Preliminary.tex` and
  `sections/6Conclusion.tex`.
- `paper/scope_edit_overleaf_replacements/`: same two files unpacked for review.
- `paper/scope_edit_dropins/`: standalone versions and notes if manual editing
  becomes easier later.
- `paper/scope_edit_v0.zip`: the scope-edit source package these replacements
  were tested against.
- `paper/OVERLEAF_FAST_ITERATION_WORKFLOW.md`: workflow for importing newer
  Overleaf zips, making drop-in packs, and using the data source pathway.

## Data and Reproducibility Artifacts

- `paper/after_report/number_lineage.md`: every reported paper number mapped to
  the JSONL/cache source that supports it.
- `paper/after_report/repro_bundle/source_file_manifest.csv`: every raw source
  file path, byte count, line count, and SHA-256.
- `paper/after_report/repro_bundle/answer_log_summaries.csv`: compact accuracy,
  token, latency, and health summaries from the answer detail logs.
- `paper/after_report/repro_bundle/retrieval_cache_summaries.csv`: compact
  Hit@k/MRR@k summaries from retrieval caches.
- `paper/after_report/tables/`: final paper-facing `.tex` tables and CSVs.

The raw detail logs and retrieval caches referenced by the paper total about
2.35 GiB. They are not duplicated under `paper/`; use the source manifest for
exact paths and checksums.

## Plot Regeneration

Figure 3 can be regenerated from committed summarized data:

```bash
python3 paper/after_report/scripts/regenerate_figure3_from_final_csv.py
```

The input is `paper/after_report/tables/topk_retrieval_metrics.csv`. The output
is `paper/after_report/plots/figure3_topk_retrieval_curves_regenerated.png`.

To refresh the compact source manifests from the local raw logs/caches:

```bash
python3 paper/after_report/scripts/build_repro_bundle.py
```

## Damage-Control Notes

- `paper/after_report/damage_report.md`: what was wrong in the stale ICML PDF
  and what the corrected final PDF changed.
- `paper/after_report/internal_discrepancies_and_recommendations.md`: remaining
  easy-to-misread package issues and recommendations.
- `paper/scope_edit_dropins/DEFERRED_DATA_CLAIM_FIXES.md`: short checklist for
  the scope-edit draft.
- `paper/EDIT_QUEUE.md`: concrete edit queue by claim type and paper location.

## Current Trust Decision

Use `paper/main.pdf` and `paper/FINALFINALVERSION.zip` as the corrected final
paper artifacts. Treat `paper/archive/icml_submission.pdf` as a stale comparison target
only. Treat `paper/scope_edit_v0.zip` as the live scope-edit base that still
needs later data/claim fixes.
