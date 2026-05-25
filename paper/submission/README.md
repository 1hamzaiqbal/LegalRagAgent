# SCOPE — Active ICML Submission Source

This is the canonical, git-tracked source for the SCOPE paper submission.
Built from the `scope_edit_v7` Overleaf snapshot.

## Build

```bash
cd paper/submission
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Compiles with TeX Live 2026 `pdflatex` (matches Overleaf). Current: 13 pages.

## Editing workflow (single-writer)

To avoid cross-machine merge conflicts, the paper has **one writer**: edits land
here, on the laptop, then get committed and pushed. The experiment machine only
*produces results* (logs/caches) and never edits the paper.

When new experiment results arrive:
1. Pull the relevant `logs/experiments.jsonl` / `logs/eval_*_detail.jsonl` rows
   from the experiment machine.
2. Recompute the affected cells.
3. Edit the corresponding `tables/*.tex` here.
4. Recompile, verify, commit, push.

## Where numbers come from

Each reported number maps to a source JSONL. See
`paper/archive/reported_data_lineage.md` (and the copy under
`paper/after_report/number_lineage.md`) for the table-cell → JSONL mapping.

## Layout notes

- `main.tex` wires the sections; `\hyre` macro renders as "SCOPE".
- `tables/` holds every result table as a standalone `.tex` (individually
  diffable — this is where result updates go).
- `sections/` holds the prose. `2Preliminary.tex` and `old_preliminary.tex`
  are unused backups kept from earlier drafts.
- `figures/35_topk_retrieval_curves.png` (Figure 3) is regenerated via
  `paper/after_report/scripts/regenerate_figure3_from_final_csv.py`.
