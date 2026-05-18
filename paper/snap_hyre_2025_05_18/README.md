# Snap-HyRE Paper Draft

This folder contains the NeurIPS-style paper draft for the fixed-method
Snap-HyRE comprehensive branch. It evaluates one Snap-HyRE method across
BarExamQA, LegalBench-SCALR, CaseHOLD, and HousingQA, reporting retrieval
exposure alongside downstream answer accuracy.

## Files

- `main.tex` - paper source.
- `main.pdf` - latest compiled PDF.
- `build_figures.py` - regenerates paper-local plots and LaTeX table
  fragments from current result package CSVs and signed qrel facts.
- `tables/` - generated LaTeX result tables included by `main.tex`.
- `figures/` - generated plots and `figure_metrics.csv`.
- `figures/archive_pre_fixed_snap_hyre_2026-05-18/` - unused plots from the
  older routing/framing draft, preserved for provenance.
- `references.bib` - BibTeX references used by the paper.
- `neurips_2024.sty` - local style file.

## Evidence Sources

Treat these as read-only inputs while eval/signoff jobs are running:

- `docs/signoff_log.md`
- `docs/compiled_results.md`
- `docs/generated/snap_hyre_package/`
- `docs/generated/retrieval_qrels_*.md`
- `logs/experiments.jsonl`

The paper directory is the writable surface for prose and paper-local figures.

## Build

From the repository root:

```bash
python3 paper/snap_hyre_2025_05_18/build_figures.py
cd paper/snap_hyre_2025_05_18
tectonic main.tex
```

If `tectonic` is unavailable, use `pdflatex`/`bibtex` in the paper directory.
