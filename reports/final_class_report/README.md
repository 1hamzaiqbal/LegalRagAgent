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

## Build

From the repository root:

```bash
tectonic reports/final_class_report/main.tex
```

The source uses the local `neurips_2024.sty` template and figures under
`docs/presentation/figures/`.

## Run Surface

The laptop Chroma checkout is currently useful for SCALR smoke tests. The
larger BarExamQA, HousingQA, CaseHOLD, and SCALR follow-up retrieval runs are
cluster-backed through SLURM, so local Chroma population is not the benchmark
scope boundary.
