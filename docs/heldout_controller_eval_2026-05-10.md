# Held-Out Controller Evaluation - 2026-05-10

This report summarizes the corrected held-out matrix on deterministic rows 200-249. It is a compact held-out check, not a full benchmark sweep.

## Summary

- Exact selected-route macro accuracy: 77.5%
- Baseline macro accuracy: 71.5%
- Exact selected-route macro calls: 1.54
- Baseline macro calls: 1.00

| Dataset | Baseline | Baseline acc | Selected route | Selected acc | Delta pp | b/c | Health |
|---|---|---:|---|---:|---:|---:|---|
| barexam | `rag_simple` | 38/50 = 76.0% | `adaptive_snap_hyre_v2` | 38/50 = 76.0% | +0.0 | 5/5 | PASS |
| housing | `rag_state_filter` | 31/50 = 62.0% | `adaptive_snap_hyre_housing_verifier` | 38/50 = 76.0% | +14.0 | 11/4 | parse_fail=1; adaptive_audit missing_prediction=1 |
| casehold | `rag_simple` | 34/50 = 68.0% | `adaptive_snap_hyre_diverse` | 39/50 = 78.0% | +10.0 | 7/2 | PASS |
| legalbench_scalr | `rag_simple` | 40/50 = 80.0% | `adaptive_snap_hyre_disagreement_majority_prior` | 40/50 = 80.0% | +0.0 | 1/1 | majority-prior replay; frontier component was 42/50 |

## Interpretation

- The exact held-out controller route beats matched baselines by +6.0 macro points on this 50-row slice.
- BarExam ties baseline at 76.0%, so the calibration-slice BarExam lift should not be overclaimed as universal.
- HousingQA again shows the clearest bottleneck-specific lift: verifier routing reaches 76.0% vs 62.0% for state-filter retrieval, with one unparseable verifier answer counted as wrong.
- CaseHOLD improves on this held-out slice with `adaptive_snap_hyre_diverse` at 78.0% vs 68.0% baseline, while the reject/escalate framing remains useful for lower-confidence rows.
- LegalBench-SCALR is the main negative/nuanced held-out result: frontier reaches 84.0%, but the exact majority-prior disagreement replay falls to 80.0%, tying baseline. This argues for keeping disagreement arbitration as a calibration finding rather than a locked held-out policy.

## Validation

- All nine retry jobs completed with `sacct` exit code `0:0`.
- `scripts/analyze_detail_flags.py` loaded 50 rows for each detail log and found zero top-level HyDE/report/knowledge artifacts and zero nested gap artifacts.
- Adaptive audit passed for BarExam `adaptive_snap_hyre_v2`, CaseHOLD `adaptive_snap_hyre_diverse`, and SCALR `adaptive_snap_hyre_frontier`.
- Housing verifier adaptive audit reported `missing_prediction=1`; row `hqa_South Dakota_7198` produced prose without a parseable `Answer: Yes/No` and was counted wrong.
- SCALR majority-prior disagreement replay ran locally from the three held-out component logs and arbitrated 9/50 rows.

## Source Logs

- casehold / `rag_simple`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260510_1709_casehold_heldout-controller-retry-or-gemma4-26b-casehold-q250-start200-end250-k5-rag_simple_detail.jsonl`
- barexam / `rag_simple`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260510_1713_barexam_heldout-controller-retry-or-gemma4-26b-barexam-q250-start200-end250-k5-rag_simple_detail.jsonl`
- legalbench_scalr / `rag_simple`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260510_1714_legalbench_scalr_heldout-controller-retry-or-gemma4-26b-legalbench_scalr-q250-start200-end250-k5-rag_simple_detail.jsonl`
- barexam / `adaptive_snap_hyre_v2`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_v2_or-gemma4-26b_20260510_1716_barexam_heldout-controller-retry-or-gemma4-26b-barexam-q250-start200-end250-k5-adaptive_snap_hyre_v2_detail.jsonl`
- casehold / `adaptive_snap_hyre_diverse`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260510_1721_casehold_heldout-controller-retry-or-gemma4-26b-casehold-q250-start200-end250-k5-adaptive_snap_hyre_diverse_detail.jsonl`
- housing / `rag_state_filter`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_state_filter_or-gemma4-26b_20260510_1722_housing_heldout-controller-retry-or-gemma4-26b-housing-q250-start200-end250-k5-rag_state_filter_detail.jsonl`
- legalbench_scalr / `rag_snap_hyde_2call`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260510_1724_legalbench_scalr_heldout-controller-retry-or-gemma4-26b-legalbench_scalr-q250-start200-end250-k5-rag_snap_hyde_2call_detail.jsonl`
- legalbench_scalr / `adaptive_snap_hyre_frontier`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_1727_legalbench_scalr_heldout-controller-retry-or-gemma4-26b-legalbench_scalr-q250-start200-end250-k5-adaptive_snap_hyre_frontier_detail.jsonl`
- housing / `adaptive_snap_hyre_housing_verifier`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_housing_verifier_or-gemma4-26b_20260510_1728_housing_heldout-controller-retry-or-gemma4-26b-housing-q250-start200-end250-k5-adaptive_snap_hyre_housing_verifier_detail.jsonl`
- legalbench_scalr / `adaptive_snap_hyre_disagreement_majority_prior`: `logs/eval_disagreement_arbitrator_majority_prior_or-gemma4-26b_20260510_scalr_heldout_n50_detail.jsonl`
