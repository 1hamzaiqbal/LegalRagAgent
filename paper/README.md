# Paper bundle — Diagnosing Legal RAG: Bottleneck-Aware Routing of Snap-HyRE

ICML 2026 submission. 8-page main paper (anonymous double-blind review),
unlimited references and appendices. Build with `pdflatex` + `bibtex` +
`pdflatex` + `pdflatex` (Overleaf does this automatically).

## File map

```
paper/
├── main.tex                       Top-level paper shell. Compile this.
├── references.bib                 33-entry bibliography.
├── icml2026.sty, icml2026.bst     ICML 2026 style + bib style.
├── fancyhdr.sty, algorithm.sty, algorithmic.sty   Required by icml2026.sty.
├── icml2026_styles/               Original ICML zip extract (reference copy).
├── sections/
│   ├── 0Abstract.tex
│   ├── 1Introduction.tex
│   ├── 2Preliminary.tex           Related work + background.
│   ├── 3Method.tex                Bottleneck taxonomy, Snap-HyRE, controller.
│   ├── 4Experiment.tex            Setup, benchmarks, models, protocol.
│   ├── 5Analysis.tex              Results (calibration + held-out + per-benchmark
│   │                              + negative controls + cross-model).
│   ├── 6Conclusion.tex            Discussion + limitations + conclusion.
│   └── Appendix.tex               Full result tables, taxonomy worked
│                                  examples, BarExam Tier-3 reference, source
│                                  logs, methodology integrity notes.
├── figures/                       10 PNG figures + captions.md.
├── diagnosing_legal_rag_overleaf.zip   Pre-built bundle for Overleaf upload.
└── _drafting/
    └── context_pack.md            Source-of-truth synthesis. Do NOT submit.
```

## Compilation

Style files are at the project root, so `\usepackage{icml2026}` resolves
without extra path setup. Locally:

```
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

On Overleaf, upload `diagnosing_legal_rag_overleaf.zip` directly (new
project -> Upload Project). Overleaf will detect `main.tex` automatically.
The zip excludes `_drafting/` and the redundant `icml2026_styles/` extract.

## Anonymity

`main.tex` uses `\usepackage{icml2026}` without options, so the style file
automatically blinds the author block at compile time. Names are kept in the
source for the camera-ready pass (switch to `\usepackage[accepted]{icml2026}`).

For preprint distribution before final decision, use
`\usepackage[preprint]{icml2026}`.

## Pending content (stand-ins)

Cells marked with `\pending{...}` in tables refer to N>=500 expansion jobs
still running on the HPC. Specifically:

| Job   | Dataset           | Status                                              |
| ----- | ----------------- | --------------------------------------------------- |
| 67897 | SCALR (N=571)     | `rag_simple` half copied/validated; frontier half running |
| 67911 | BarExam (N=500)   | running                                             |
| 67912 | HousingQA (N=500) | running                                             |
| 67913 | CaseHOLD (N=500)  | running                                             |
| 67915 | SCALR (N=571)     | rewrite retry after 67914 died on bad node          |

When these clear validation (`analyze_detail_flags.py` +
`audit_adaptive_hyre_logs.py` for HyRE-family rows), backfill the
`\pending{}` cells in `sections/5Analysis.tex` and `sections/Appendix.tex`.

## Bibliography

`references.bib` has 33 entries spanning RAG/HyDE, adaptive/agentic RAG,
reasoning + retrieval coupling, legal NLP benchmarks, legal RAG, RAG
evaluation/diagnostic, and one multi-hop reference. All citation keys used
in the section files are present.

## Word counts (main body)

| Section          | Words |
| ---------------- | -----:|
| Abstract         |   242 |
| Introduction     |   694 |
| Preliminary      |   656 |
| Method           | 1,904 |
| Experiment       |   426 |
| Analysis         | 2,017 |
| Conclusion       |   751 |
| **Total body**   | **6,690** |
| Appendix         | 4,380 |

ICML's 8-page main-paper limit accommodates ~6.5k words plus figures and
tables. Adjust each section as you tighten or expand.

## Custom macros (defined in `main.tex`)

| Macro          | Renders as                                           |
| -------------- | ---------------------------------------------------- |
| `\hyre{}`      | "Snap-HyRE"                                          |
| `\controller{}`| "bottleneck-aware diagnostic controller"             |
| `\framework{}` | "diagnostic adaptation framework"                    |
| `\method{x}`   | `\texttt{x}` with detokenized underscores            |
| `\dataset{x}`  | `\textsc{x}`                                         |
| `\pp`          | "pp" superscript (works inside or outside math mode) |
| `\pending{x}`  | gray italic "x" placeholder for unverified rows      |

## Source-of-truth verification

Numbers in the paper are sourced from these audited docs (under
`/Users/hamzaiqbal/grad/LegalRagAgent/`):

- `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md` — primary table
- `docs/meeting_eval_expansion_status_2026-05-11.md` — pending-job status
- `docs/signoff_log.md` — cite-or-not gate
- `docs/snap_hyde_2call_2026-04-28.md` — Snap-HyRE 2-call mechanism
- `docs/casehold_repaired_rerun_2026-05-01.md` — CaseHOLD repaired pair
- `docs/housing_state_filter_followup_2026-05-01.md` — HousingQA state filter
- `docs/compiled_results.md` §1.1 — BarExam Tier 3 historical reference

A condensed version lives in `_drafting/context_pack.md` — convenient for
referencing during edits.
