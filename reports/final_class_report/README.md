# Final Class Report Draft

This folder contains the NeurIPS-style class report draft centered on
snap-conditioned HyDE and bottleneck-typed legal RAG. The main report uses a
legal-only four-benchmark set: BarExamQA, HousingQA, LegalBench-SCALR, and
CaseHOLD. MuSiQue is treated as an internal non-legal mechanism check, not a
main-table benchmark.

## Files

- `main.tex` - report source.
- `references.bib` - BibTeX references used by the report.
- `evidence_snapshot.md` - local validation notes for the headline rows.
- `build_figures.py` - regenerates report-specific plots from detail logs.
- `figures/` - generated plots and `figure_metrics.csv`.

## Build

From the repository root:

```bash
tectonic reports/final_class_report/main.tex
```

The source uses the local `neurips_2024.sty` template and figures under
`docs/presentation/figures/` plus generated figures under this folder.

To rebuild the generated figures first:

```bash
python reports/final_class_report/build_figures.py
```

## Run Surface

Cluster follow-up scripts used for report validation live under `scripts/hpc/`.
The current targeted additions are HousingQA state filtering via
`scripts/hpc/slurm_housing_state_filter.sh` plus the chunked recovery script
`scripts/hpc/slurm_housing_state_filter_chunks.sh`, and SCALR snap/HyDE
ablations via `scripts/hpc/slurm_scalr_snap_ablation.sh`. Promote rows into
the report only after pulling the detail logs and re-running the local sanity
checks in `evidence_snapshot.md`.
