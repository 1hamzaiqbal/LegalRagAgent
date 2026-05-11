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
| Four legal benchmarks only | `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md` | Done |
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
| + snap-only reasoning | 85.5 | 55.0 | 74.0 | 72.5 | 71.8 | 2.00 |
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
do not yet cover every inherited control at N=200. The snap-only row has now
landed across all four legal benchmarks; HyRE-only and fixed Snap-HyRE fill-in
rows remain live targeted jobs rather than a broad sweep.

## Landed Since This Expansion

These are source-gated additions after the initial package:

| Item | Source | Status |
|---|---|---|
| BarExam snap-only control | SLURM `67773`; `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0346_barexam_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl` | Completed, 171/200 = 85.5%, avg calls 2.00, errors 0, missing prediction 1 |
| BarExam HyRE-only retrieval control | SLURM `67825`; `logs/eval_rag_hyde_or-gemma4-26b_20260511_0526_barexam_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl` | Completed, 164/200 = 82.0%, avg calls 2.00, errors 0 |
| HousingQA snap-only control | SLURM `67775`; `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0259_housing_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl` | Completed, 110/200 = 55.0%, avg calls 2.00, errors 0, missing prediction 1 |
| CaseHOLD snap-only control | SLURM `67777`; `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0418_casehold_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl` | Completed, 148/200 = 74.0%, avg calls 2.00, errors 0 |
| LegalBench-SCALR snap-only control | SLURM `67779`; `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0411_legalbench_scalr_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl` | Completed, 145/200 = 72.5%, avg calls 2.00, errors 0 |
| HousingQA HyRE-only retrieval control | SLURM `67826`; `logs/eval_rag_hyde_or-gemma4-26b_20260511_0443_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl` | Completed, 100/200 = 50.0%, avg calls 2.00, errors 0 |
| CaseHOLD HyRE-only retrieval control | SLURM `67827`; `logs/eval_rag_hyde_or-gemma4-26b_20260511_0511_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl` | Completed, 143/200 = 71.5%, avg calls 2.00, errors 0, missing prediction 1 |
| LegalBench-SCALR HyRE-only retrieval control | SLURM `67828`; `logs/eval_rag_hyde_or-gemma4-26b_20260511_0559_legalbench_scalr_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl` | Completed, 142/200 = 71.0%, but rejected as clean result: one runaway final answer has 267,458 chars / 70,593 output tokens |
| HousingQA fixed Snap-HyRE control | SLURM `67830`; `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0559_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl` | Completed, 103/200 = 51.5%, avg calls 2.00, errors 0 |
| CaseHOLD fixed Snap-HyRE control | SLURM `67831`; `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0602_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl` | Completed, 144/200 = 72.0%, avg calls 2.00, errors 0 |
| SCALR capped HyRE-only rerun | SLURM `67864`; `LLM_MAX_COMPLETION_TOKENS=4096` | Launched to replace rejected uncapped SCALR HyRE-only row |
| GTE query-embedding repair | `rag_utils.py`; direct smoke SLURM `67820` | Completed; repaired `position_ids`, finite 1024-d unit-norm query embeddings |
| Retrieval smoke after repair | SLURM `67821`; `logs/eval_rag_hyde_or-gemma4-26b_20260511_0341_barexam_embedding-fix-smoke2-or-gemma4-26b-n5-k5-rag_hyde_detail.jsonl` | Completed, 5/5; confirms retrieval-bearing jobs can run again |

The completed snap-only row is a useful diagnostic control. BarExam nearly
matches the best verified BarExam controller row (86.0%), so its route decision
should ask whether retrieval is worth the extra latency. HousingQA is the
opposite pattern: visible snap reasoning alone is below the verified state-filter
baseline (60.5%) and far below the Housing verifier route (74.5%). CaseHOLD
slightly beats its baseline but does not resolve the option-conversion story;
SCALR falls below the existing retrieval/Snap-HyRE rows. The macro result is
therefore not a positive method claim; it is evidence that the controller needs
to decide where reasoning is spent.

BarExam HyRE-only is a modest retrieval-only lift: `rag_hyde` reaches 82.0%,
above the baseline retrieval row (80.0%) but below snap-only reasoning (85.5%)
and the stronger fixed Snap-HyRE v2 route (86.0%). This keeps BarExam routed
toward a rewrite-vs-Snap-HyRE selector rather than generic hypothetical
retrieval alone.

The repaired HousingQA HyRE-only row is also negative: `rag_hyde` reaches 50.0%,
below snap-only reasoning (55.0%), state-filter retrieval (60.5%), and the
Housing verifier route (74.5%). This further supports routing HousingQA toward
state scoping and verification rather than generic hypothetical retrieval.
CaseHOLD HyRE-only is also not enough: `rag_hyde` reaches 71.5%, below the
baseline retrieval row (73.0%), snap-only reasoning (74.0%), and diverse HyRE
(73.5%). That keeps the CaseHOLD diagnosis focused on answer-option conversion,
not generic hypothetical retrieval.

The fixed `rag_snap_hyde_2call` fill-ins reinforce the same routing pattern.
HousingQA fixed Snap-HyRE reaches only 51.5%, below snap-only (55.0%),
state-filter retrieval (60.5%), snap-HyRE state retrieval (63.0%), and the
verifier route (74.5%). CaseHOLD fixed Snap-HyRE reaches 72.0%, below the
current baseline (73.0%), snap-only (74.0%), and diverse HyRE-family row
(73.5%). These are useful negative controls: fixed Snap-HyRE is not the
adaptive policy for those bottlenecks.

SCALR HyRE-only technically completed at 71.0%, but it fails the May 11 health
gate because one row produced a 267,458-character final answer. Do not promote
that row as clean evidence; capped rerun `67864` was launched with
`LLM_MAX_COMPLETION_TOKENS=4096`.

## Active And Pending Jobs

All active, pending, and newly completed expansion jobs below use the repaired
cluster script from commit `dfa4d8a` plus the current `rag_utils.py`
embedding-loader repair. They set
`DATA_REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent`,
`CHROMA_DB_DIR=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/chroma_db`,
skip Chroma count preflight, and exclude the bad A40 nodes that produced
CUDA/ECC failures. Retrieval-bearing jobs also run with the repaired online GTE
query embedder: reinitialized RoPE `position_ids`, `max_seq_length=512`, fp16
disabled by default, and finite embedding smoke verified by SLURM `67820`.
If a rerun is needed because a row produces runaway final-answer text, set
`LLM_MAX_COMPLETION_TOKENS` on the launch to cap generations without changing
the default historical rows.

### Gemma 4 26B Ladder Controls

Provider: `or-gemma4-26b`. Sample: N=200, seed 42, k=5.

| Job | Dataset | Mode | Why it matters |
|---:|---|---|---|
| 67773 | BarExam | `snap_only_in_final` | Completed and copied locally: 85.5%, 2.00 calls. |
| 67825 | BarExam | `rag_hyde` | Completed and copied locally: 82.0%, 2.00 calls. |
| 67775 | HousingQA | `snap_only_in_final` | Completed and copied locally: 55.0%, 2.00 calls. |
| 67826 | HousingQA | `rag_hyde` | Completed and copied locally: 50.0%, 2.00 calls. |
| 67777 | CaseHOLD | `snap_only_in_final` | Completed and copied locally: 74.0%, 2.00 calls. |
| 67827 | CaseHOLD | `rag_hyde` | Completed and copied locally: 71.5%, 2.00 calls. |
| 67779 | LegalBench-SCALR | `snap_only_in_final` | Completed and copied locally: 72.5%, 2.00 calls. |
| 67828 | LegalBench-SCALR | `rag_hyde` | Completed and copied locally: 71.0%, but rejected as a clean row due one runaway final answer. |
| 67864 | LegalBench-SCALR | `rag_hyde` | Capped rerun launched with `LLM_MAX_COMPLETION_TOKENS=4096`. |
| 67829 | BarExam | `rag_snap_hyde_2call` | Fixed Snap-HyRE N=200 row missing for this provider. |
| 67830 | HousingQA | `rag_snap_hyde_2call` | Completed and copied locally: 51.5%, 2.00 calls. |
| 67831 | CaseHOLD | `rag_snap_hyde_2call` | Completed and copied locally: 72.0%, 2.00 calls. |

### Cross-Model Sanity Layer

Provider: `groq-llama70b`. Sample: same held-out slice rows 200-249
(`N_QUESTIONS=250`, `SAMPLE_START=200`, `SAMPLE_END=250`), seed 42, k=5.

| Job | Dataset | Mode | Why it matters |
|---:|---|---|---|
| 67832 | BarExam | `rag_simple` | Cross-model baseline. |
| 67833 | BarExam | `adaptive_snap_hyre_v2` | Cross-model selected route. |
| 67834 | HousingQA | `rag_state_filter` | Cross-model state-filter baseline. |
| 67835 | HousingQA | `adaptive_snap_hyre_housing_verifier` | Cross-model verifier route. |
| 67836 | CaseHOLD | `rag_simple` | Cross-model baseline. |
| 67837 | CaseHOLD | `adaptive_snap_hyre_diverse` | Cross-model selected route. |
| 67838 | LegalBench-SCALR | `rag_simple` | Cross-model baseline. |
| 67839 | LegalBench-SCALR | `adaptive_snap_hyre_frontier` | Cross-model selected route. |

These are not report numbers until the stdout and detail logs pass the
validation gates below.

### Full-Corpus Feasibility Probe

Provider: `or-gemma4-26b`. This is a targeted full-SCALR sanity check, not an
all-dataset full-corpus sweep.

| Job | Dataset | N | Modes | Why it matters |
|---:|---|---:|---|---|
| 67863 | LegalBench-SCALR | 571 | `rag_simple`, `adaptive_snap_hyre_frontier` | Tests whether the smaller legal benchmark's controller signal survives beyond the N=200 slice. |

This is not a report number until it completes and passes the validation gates.

## Invalid / Rejected Runs

Do not cite any of these:

| Jobs | Issue | Action |
|---|---|---|
| 67746, 67747, 67748 | Retrieval failed before LLM calls on A40 nodes with CUDA index assertions or ECC errors. | Cancelled / invalid. |
| 67749 | Housing job hung in Chroma count preflight. | Cancelled; script now supports skip gate. |
| 67750-67753 | Cerebras preflight failed from missing credentials. | Invalid; use Groq for second-provider sanity unless Cerebras key is installed. |
| 67754-67772 | Bad launch used `DATA_REPO=$REPO`, creating a self-referential `datasets` symlink. | Cancelled / invalid; script now avoids self-linking. |
| 67774, 67776, 67778, 67780-67791 | CPU embedding fallback launch was rejected after `rag_hyde` failed with an embedding index error. | Cancelled / invalid; retrieval-bearing jobs relaunched with normal GPU embedding on non-A40 nodes. |
| 67792, 67793, 67808 | Default GTE query embedding crashed before valid retrieval due to a corrupted remote-code `position_ids` buffer. | Invalid; root-caused and repaired in `rag_utils.py`, then smoke-tested by `67820` and `67821`. |
| 67794-67806 | Pending retrieval jobs from the pre-repair launch. | Cancelled / invalid to avoid wasting API calls on the broken embedder. |
| 67810-67818 | Embedding debug/smoke attempts, including a failed ONNX backend check because `optimum` is not installed. | Debug only; do not cite as eval results. |

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
CaseHOLD dominating cost and runtime. A targeted full-SCALR probe has been
launched as SLURM `67863` because SCALR is the smallest legal benchmark in the
current four-task set. The meeting-safe standard is therefore:

- report existing verified N=200/N=50 source-gated results;
- integrate new N=200 ladder controls only if they land cleanly;
- treat full-corpus expansion beyond the SCALR probe as post-meeting paper
  work unless additional full runs land cleanly under the same gates.

## Validation Gates

A result can be promoted only if all gates pass:

1. `sacct` says `COMPLETED` with exit `0:0`.
2. stdout has no `Traceback`, API auth/rate-limit failure, CUDA assertion,
   ECC error, symlink error, parse meltdown, or circuit-breaker abort.
3. The expected detail JSONL exists in
   `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/`.
4. `scripts/analyze_detail_flags.py <detail.jsonl>` reports sane health:
   rows match target, no missing predictions spike, no empty-retrieval failure,
   and no runaway final-answer length spike.
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
