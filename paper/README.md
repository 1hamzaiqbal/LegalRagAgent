# Paper Workspace

The active ICML draft lives in:

```text
paper/snap_hyre_2025_05_18/
```

The root `paper/` directory is now only a workspace wrapper. Older duplicate
draft files that previously lived at the root were moved to
`archive_pre_current_icml_2026-05-20/root_duplicate_draft/` so there is one
clear active paper tree.

Timestamped Overleaf exports are local handoff artifacts. Older exports are
collected under `archive_overleaf_uploads_2026-05-21/`, while the latest upload
package remains at `overleaf_snap_hyre_20260521_100948/` with the matching zip
beside it.

## Active Scope

- Method: fixed `snap_hyre`, not adaptive HyRE.
- Main benchmarks: BarExamQA, HousingQA, Legal-Link-EU, MASLegalBench.
- Historical only unless explicitly re-added: CaseHOLD and LegalBench-SCALR.
- Main providers: `groq-llama8b`, `or-gemma4-26b`, `groq-llama70b`.
- Main modes: `llm_only`, `rag_simple`, `rag_hyde`, `snap_hyre`,
  `golden_passage`, `golden_plus_neighbors`, `rag_rewrite`.
- Current audited completion: 71/78 exact-scored answer cells.

## Build

From the repository root:

```bash
python3 paper/snap_hyre_2025_05_18/build_current_paper_assets.py
cd paper/snap_hyre_2025_05_18
tectonic main.tex
```

## Evidence Discipline

For paper-facing numerical claims, use `docs/signoff_log.md` first, then
`docs/compiled_results.md`, then `logs/experiments.jsonl` only as supporting
source. Keep run-status/process details in docs, not in the manuscript prose.
