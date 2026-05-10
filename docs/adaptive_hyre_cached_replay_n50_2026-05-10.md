# Adaptive HyRE Cached Replay N=50 - 2026-05-10

## Purpose

This run isolates the downstream retrieval/final-answer behavior from fresh
Snap-HyDE/HyRE generation variance. The cache replays the already generated
snap answer and HyRE passage, then spends only one LLM call on the final answer.

The cache was built from the clean N=200 frontier detail logs and contains 800
rows across Barexam, HousingQA, CaseHOLD, and LegalBench-SCALR.

## Implementation Gate

- Commit `5514cba` wires cache replay into the plain `rag_snap_hyde_2call`
  path used by the SCALR frontier route.
- Commit `1ca2c0c` updates `scripts/audit_adaptive_hyre_logs.py` so cached
  rows pass with one fewer LLM call.
- Smoke job `67345` verified the SCALR path: 3/3 cache hits, 1.00 calls/row,
  no fresh `snap_and_hyre` calls, and PASS health checks after the audit update.

## Cluster Run

Manifest:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_mode_matrix_20260510_0830_cached_frontier.tsv`

Cache:
`/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/hyre_cache/frontier_n200_hyre_cache.jsonl`

Jobs:

| Job | Dataset | Mode | N | Status |
|---:|---|---|---:|---|
| 67360 | Barexam | `adaptive_snap_hyre_frontier` | 50 | PASS |
| 67361 | HousingQA | `adaptive_snap_hyre_frontier` | 50 | PASS |
| 67362 | CaseHOLD | `adaptive_snap_hyre_frontier` | 50 | PASS |
| 67363 | LegalBench-SCALR | `adaptive_snap_hyre_frontier` | 50 | PASS |

## Results

| Dataset | Route | Accuracy | Gold Retrieved | Cache Hits | Avg Calls | Health |
|---|---|---:|---:|---:|---:|---|
| Barexam | `frontier_barexam_v2` | 40/50 = 80.0% | 4/50 | 50/50 | 1.00 | PASS |
| HousingQA | `frontier_housing_diverse` | 31/50 = 62.0% | 23/50 | 50/50 | 1.00 | PASS |
| CaseHOLD | `frontier_casehold_diverse` | 35/50 = 70.0% | 14/50 | 50/50 | 1.00 | PASS |
| LegalBench-SCALR | `frontier_scalr_plain_snap_hyde` | 39/50 = 78.0% | 42/50 | 50/50 | 1.00 | PASS |

Detail logs:

- Barexam:
  `logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0839_barexam_adaptive-hyre-or-gemma4-26b-barexam-n50-k5-adaptive_snap_hyre_frontier_detail.jsonl`
- HousingQA:
  `logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0849_housing_adaptive-hyre-or-gemma4-26b-housing-n50-k5-adaptive_snap_hyre_frontier_detail.jsonl`
- CaseHOLD:
  `logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0838_casehold_adaptive-hyre-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_frontier_detail.jsonl`
- LegalBench-SCALR:
  `logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0836_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n50-k5-adaptive_snap_hyre_frontier_detail.jsonl`

## Interpretation

Cached replay confirms that fixed HyRE passages can be reused as an experiment
primitive. This is useful for rapid iteration because it cuts adaptive frontier
runs from two calls to one call per row while preserving the retrieval/query
objects being tested.

The replay results do not show that one fixed frontier policy solves all four
datasets. They show a narrower and more actionable point: once generation
variance is removed, the remaining bottlenecks are mostly downstream conversion
or evidence-use problems.

- LegalBench-SCALR is the cleanest beneficiary of fixed HyRE replay at N=50:
  78.0% with high gold retrieval and one call per row.
- HousingQA stays near the known 62-63% band. Its failures are not caused by
  fresh HyRE stochasticity; the next useful intervention is yes/no calibration
  or state-grounded evidence verification.
- CaseHOLD remains answer-conversion limited. Retrieval improvements alone are
  not enough; the next useful intervention is option-aware reranking or a final
  selector that compares choices directly against retrieved holdings.
- Barexam replay drops below the clean N=200 v2 result. That makes it a poor
  target for cached final-answer-only optimization; its stronger evidence is
  still the full v2 N=200 run.

## Next Targeted Iterations

Do not scale broad method sweeps from this result. Use the cache to run cheaper
targeted probes:

1. CaseHOLD option reranking: rerank retrieved holdings against each candidate
   option before final answer selection.
2. HousingQA yes/no verifier: force a final entailment check over the statute
   snippets and the proposition being asked. This became the strongest targeted
   result from the cached replay path. At N=50,
   `adaptive_snap_hyre_housing_verifier` reached 37/50 = 74.0% at 1.00
   call/row, reducing false-positive Yes errors from 14 to 6 while increasing
   false-negative No errors from 5 to 7. The N=200 follow-up job `67373` held:
   149/200 = 74.5%, 89/200 gold retrieved, 200/200 cache hits, 1.00 calls/row,
   and PASS health checks. Against the N=200 Housing frontier, false-positive
   Yes errors fell from 62 to 29 while false-negative No errors rose from 14 to
   22. Paired comparison versus the N=200 frontier was +12.5pp, b/c=36/11,
   p=0.000346.
3. SCALR low-cost replay: use the cached one-call path for fast prompt
   comparisons because retrieval is already strong.
4. Router analysis: use cached replay logs to identify when the frontier route
   should escalate from a one-call replay to an option verifier.
