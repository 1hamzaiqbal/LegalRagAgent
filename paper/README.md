# Paper Workspace

Top-level map of the paper directory after the 2026-05-22 cleanup. Use this
file as the front door; everything else links from here.

## Final Submission Artifacts

- [`main.pdf`](main.pdf): the upload PDF. Trust this.
- [`FINALFINALVERSION.zip`](FINALFINALVERSION.zip): final source package
  (contains `final_icml_submission/main.tex` and supporting files).

## Active Editing Surface

The scope-edit draft is the live Overleaf base. The damage-control queue
below tracks what still has to land in that draft.

- [`EDIT_QUEUE.md`](EDIT_QUEUE.md): claim-by-claim repair queue for the scope
  draft (HousingQA parity wording, Gold Evidence framing on Llama 70B,
  exemplar scope, Figure 3 caption, Table 4 efficiency, Table 13 bolding,
  bibliography placeholders).
- [`scope_edit_v0.zip`](scope_edit_v0.zip): current Overleaf source snapshot.
- [`scope_edit_overleaf_replacements/`](scope_edit_overleaf_replacements/)
  and [`scope_edit_overleaf_replacements.zip`](scope_edit_overleaf_replacements.zip):
  overwrite-only drop-in pack for Overleaf. Only changes
  `sections/2Preliminary.tex` and `sections/6Conclusion.tex`.
- [`scope_edit_dropins/`](scope_edit_dropins/): standalone fragments, the
  `DEFERRED_DATA_CLAIM_FIXES.md` checklist, and
  [`scope_edit_dropins/historical_drafts/`](scope_edit_dropins/historical_drafts/)
  with older draft snapshots (longer Background / Method / Analysis sections)
  to mine prose from when expanding the trimmed scope draft.

## Workflow Docs

- [`PAPER_FAST_EDIT_HANDOFF.md`](PAPER_FAST_EDIT_HANDOFF.md): top-level index of
  upload artifacts and data sources.
- [`OVERLEAF_FAST_ITERATION_WORKFLOW.md`](OVERLEAF_FAST_ITERATION_WORKFLOW.md):
  how to import a newer Overleaf zip, build drop-in packs, and stay aligned
  with the data sources.

## Audit and Reproducibility

- [`after_report/`](after_report/): post-audit bundle. Number-to-source map,
  damage report (stale vs corrected), reproducibility notes, regenerated
  Figure 3, and copied final tables. Start here for any "where did this
  number come from?" question.
- Figure 3 regenerate:
  `python3 paper/after_report/scripts/regenerate_figure3_from_final_csv.py`
- Refresh source manifest:
  `python3 paper/after_report/scripts/build_repro_bundle.py`

## Archive

[`archive/`](archive/) holds superseded artifacts kept for provenance:

- `icml_submission.pdf` — stale comparison target (see
  [`after_report/damage_report.md`](after_report/damage_report.md) for what
  was wrong).
- `snap_hyre_2025_05_18/` — older local LaTeX tree and `build_current_paper_assets.py`.
  Not the final-package generator.
- `FINAL_VERSION.zip`, `Toward_Adaptive_HyRE_*.zip` — earlier source bundles.
- `paper_fast_edit_complete_handoff.zip` — the 14 MB bundled handoff (its
  contents are already in this directory).
- `reported_data_lineage.md`, `icml_submission_damage_report.md` — root
  duplicates of files under `after_report/` (kept for provenance only).
- `archive_overleaf_uploads_2026-05-21/` — earlier nested archive.

Local-only Overleaf snapshots (~215 MB) live outside the repo at
`~/grad/paper_local_archive/`.

## Untracked Inputs at Root

These are kept on disk but not tracked in git; they are feedback inputs to the
ongoing scope edit:

- `main_langlin_feedback.tex` — advisor markup of the abstract/intro.
- `comments_w_nearby_snippets.md` — extracted reviewer comments with
  surrounding paper text.
