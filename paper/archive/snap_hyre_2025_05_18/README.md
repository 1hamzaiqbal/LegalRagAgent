# Snap-HyRE ICML/Workshop Draft

This is the ICML-format workshop paper draft. It evaluates one fixed Snap-HyRE
method on the main BarExamQA/HousingQA story.

## Scope

- Main matrix: BarExamQA and HousingQA.
- Models/providers: `groq-llama8b`, `or-gemma4-26b`, and `groq-llama70b`.
- Main modes: `llm_only`, `rag_simple`, `rag_hyde`, `snap_hyre`, and
  `golden_passage`.
- Supplemental controls: `rag_rewrite` and `golden_plus_neighbors`.
- HousingQA retrieval rows must use jurisdiction state filtering to appear in
  the main comparison.

## Files

- `main.tex` - ICML paper shell.
- `sections/` - paper prose sections.
- `build_current_paper_assets.py` - regenerates paper tables and figures from
  validated result sources.
- `current_audited_rows.csv` - generated answer/retrieval table used by the
  draft.
- `tables/current_*.tex` - generated LaTeX tables used by `main.tex`.
- `figures/20_snap_hyre_pipeline_art.png` - generated Figure 1 method diagram.
- Active included figures: `figures/20_snap_hyre_pipeline_art.png`,
  `figures/25_barexam_retrieval_deltas.png`, and
  `figures/35_topk_retrieval_curves.png`.
- Other numbered PNGs in `figures/` are inactive generated scratch/provenance
  assets unless referenced by `sections/*.tex`.
- `tables/exemplar_probe_q20.tex` and `figures/exemplar_probe_q20_metrics.csv`
  - probe-only real-passage exemplar summary; not part of the audited main
  matrix.
- `references.bib`, `icml2026.sty`, and `icml2026.bst` - bibliography and ICML
  style dependencies.

Older figures, tables, stale generated names, previous export packages, and
earlier root drafts are preserved under
`archive_pre_current_icml_2026-05-20/` or the root paper archive. They are not
part of the build.

## Evidence Sources

Use these gates for reported result claims, in order:

1. `docs/signoff_log.md`
2. `docs/compiled_results.md`
3. `logs/experiments.jsonl`

Do not promote numbers from status files, stale package CSVs, or older
narrative files unless the audited result gate supports the claim.

## Build

From the repository root:

```bash
python3 paper/snap_hyre_2025_05_18/build_current_paper_assets.py
cd paper/snap_hyre_2025_05_18
tectonic main.tex
```

If `tectonic` is unavailable, use `pdflatex`/`bibtex` in this directory.
