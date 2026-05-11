# Meeting Eval Expansion Status - 2026-05-11

Purpose: source-gated handoff for the May 11, 2026 4pm meeting goal. This
extends the existing diagnostic-adaptation package with the stricter ask for
an inherited ablation ladder, model coverage, full-corpus feasibility, and
cluster-log validity.

## North Star

Build a bottleneck-aware legal RAG package, not a universal Snap-HyRE victory
claim. The meeting story should be:

> Calibration traces expose which bottleneck is active in each legal benchmark;
> the controller then routes to the cheapest plausible intervention among
> baseline RAG, query rewrite, Snap-HyRE/HyRE, metadata filtering, option
> grounding, verifier policies, disagreement arbitration, or reject/escalate.

Current source-gated claim:

> Fixed HyRE is useful but not universally dominant. The stronger research
> direction is diagnostic adaptation: generated reasoning helps when it is
> routed to the bottleneck the task actually exhibits.

## Current Source-Gated Package

These are meeting-ready now:

| Item | Source | Status |
|---|---|---|
| Four legal benchmarks only | `docs/meeting_prep_2026-05-12_diagnostic_adaptation.md` | Done |
| Calibration controller table | `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json` | Done |
| Held-out controller table | `docs/heldout_controller_eval_2026-05-10.json` | Done |
| Query-rewrite held-out control | `docs/heldout_query_rewrite_2026-05-10.json` | Done |
| CaseHOLD direct option-table repair | `docs/casehold_option_table_direct_heldout_2026-05-11.md` | Done, clean negative |
| Figure pack | `scripts/build_meeting_package_figures.py` | Done |
| Method ladder flowchart | `docs/presentation/figures/16_method_ladder_flowchart.png` | Done |

Existing headline table:

| Model & Method | BarExam | HousingQA | CaseHOLD | SCALR | Avg. | Calls |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B + baseline retrieval | 80.0 | 60.5 | 73.0 | 74.0 | 71.9 | 1.00 |
| + legal query rewrite control | 82.0 | 58.0* | 72.0* | 76.0* | 72.0 | 2.00 |
| + fixed HyRE family | 86.0 | 63.5 | 73.5 | 76.0 | 74.8 | 2.00 |
| + diagnostic controller routes | 86.0 | 74.5 | 73.5 | 77.5 | 77.9 | 1.30 |

`*` = N=50 calibration control. Do not hide that caveat; also do not let it
dominate the meeting narrative.

## Expanded Ablation Gap

The professor-facing ablation should look inherited:

1. baseline RAG,
2. snap-only reasoning visible to the final answer,
3. HyRE/HyDE-only retrieval,
4. fixed Snap-HyRE,
5. diagnostic route.

Existing N=200 logs already cover the key baseline/controller rows, but they
do not yet cover every inherited control at N=200. The missing rows are being
run as targeted jobs rather than a broad sweep.

## Active Jobs

All active jobs below use the repaired cluster script from commit `dfa4d8a`.
They set `DATA_REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent`,
`CHROMA_DB_DIR=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/chroma_db`,
skip Chroma count preflight, and exclude the bad A40 nodes that produced
CUDA/ECC failures. Snap-only jobs do not retrieve. Retrieval-bearing retries
use the normal GPU embedding path on the remaining healthy nodes; the CPU
embedding fallback was rejected after an embedding index error.

### Missing Gemma 4 26B Ladder Controls

Provider: `or-gemma4-26b`. Sample: N=200, seed 42, k=5.

| Job | Dataset | Mode | Why it matters |
|---:|---|---|---|
| 67773 | BarExam | `snap_only_in_final` | Snap-only control for inherited ladder. |
| 67792 | BarExam | `rag_hyde` | HyRE/HyDE-only control. |
| 67775 | HousingQA | `snap_only_in_final` | Snap-only control for yes/no statutory task. |
| 67793 | HousingQA | `rag_hyde` | Tests whether plain HyRE helps without state/verifier route. |
| 67777 | CaseHOLD | `snap_only_in_final` | Snap-only control for option conversion. |
| 67794 | CaseHOLD | `rag_hyde` | HyRE-only control for holding retrieval. |
| 67779 | LegalBench-SCALR | `snap_only_in_final` | Snap-only control for candidate disambiguation. |
| 67795 | LegalBench-SCALR | `rag_hyde` | HyRE-only control. |
| 67796 | BarExam | `rag_snap_hyde_2call` | Fixed Snap-HyRE N=200 row missing for this provider. |
| 67797 | HousingQA | `rag_snap_hyde_2call` | Fixed Snap-HyRE N=200 row missing for this provider. |
| 67798 | CaseHOLD | `rag_snap_hyde_2call` | Fixed Snap-HyRE N=200 row missing for this provider. |

### Cross-Model Sanity Layer

Provider: `groq-llama70b`. Sample: same held-out slice rows 200-249
(`N_QUESTIONS=250`, `SAMPLE_START=200`, `SAMPLE_END=250`), seed 42, k=5.

| Job | Dataset | Mode | Why it matters |
|---:|---|---|---|
| 67799 | BarExam | `rag_simple` | Cross-model baseline. |
| 67800 | BarExam | `adaptive_snap_hyre_v2` | Cross-model selected route. |
| 67801 | HousingQA | `rag_state_filter` | Cross-model state-filter baseline. |
| 67802 | HousingQA | `adaptive_snap_hyre_housing_verifier` | Cross-model verifier route. |
| 67803 | CaseHOLD | `rag_simple` | Cross-model baseline. |
| 67804 | CaseHOLD | `adaptive_snap_hyre_diverse` | Cross-model selected route. |
| 67805 | LegalBench-SCALR | `rag_simple` | Cross-model baseline. |
| 67806 | LegalBench-SCALR | `adaptive_snap_hyre_frontier` | Cross-model selected route. |

These are not report numbers until the stdout and detail logs pass the
validation gates below.

## Invalid / Rejected Runs

Do not cite any of these:

| Jobs | Issue | Action |
|---|---|---|
| 67746, 67747, 67748 | Retrieval failed before LLM calls on A40 nodes with CUDA index assertions or ECC errors. | Cancelled / invalid. |
| 67749 | Housing job hung in Chroma count preflight. | Cancelled; script now supports skip gate. |
| 67750-67753 | Cerebras preflight failed from missing credentials. | Invalid; use Groq for second-provider sanity unless Cerebras key is installed. |
| 67754-67772 | Bad launch used `DATA_REPO=$REPO`, creating a self-referential `datasets` symlink. | Cancelled / invalid; script now avoids self-linking. |
| 67774, 67776, 67778, 67780-67791 | CPU embedding fallback launch was rejected after `rag_hyde` failed with an embedding index error. | Cancelled / invalid; retrieval-bearing jobs relaunched with normal GPU embedding on non-A40 nodes. |

## Full-Corpus Feasibility

Harness-level full sizes are:

| Dataset | Harness full N |
|---|---:|
| BarExam | 1,195 |
| HousingQA | 6,853 |
| CaseHOLD | 3,600 |
| LegalBench-SCALR | 571 |

Full-corpus all-method, all-model coverage is not a realistic May 11 4pm gate.
It would be thousands to tens of thousands of LLM calls, with HousingQA and
CaseHOLD dominating cost and runtime. The meeting-safe standard is therefore:

- report existing verified N=200/N=50 source-gated results;
- integrate new N=200 ladder controls only if they land cleanly;
- treat full-corpus expansion as post-meeting paper work unless a smaller full
  dataset such as SCALR lands cleanly under the repaired CPU retrieval path.

## Validation Gates

A result can be promoted only if all gates pass:

1. `sacct` says `COMPLETED` with exit `0:0`.
2. stdout has no `Traceback`, API auth/rate-limit failure, CUDA assertion,
   ECC error, symlink error, parse meltdown, or circuit-breaker abort.
3. The expected detail JSONL exists in
   `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/`.
4. `scripts/analyze_detail_flags.py <detail.jsonl>` reports sane health:
   rows match target, no missing predictions spike, no empty-retrieval failure.
5. Adaptive/HyRE-family rows also pass
   `scripts/audit_adaptive_hyre_logs.py <detail.jsonl>`.
6. Paired comparisons use overlapping labels only; McNemar tests are optional
   for meeting but required before paper-grade claims.

## Meeting Read If Jobs Are Still Pending

If the active jobs do not finish before the meeting, the package is still
usable. Present the verified controller story, then say:

> We launched a source-gated inherited-ladder fill-in: snap-only, HyRE-only,
> and fixed Snap-HyRE controls at N=200, plus a Groq Llama 70B held-out sanity
> layer. Those numbers are intentionally not in the table until logs pass the
> same gates as the current results.

That is stronger than rushing invalid numbers into the deck.
