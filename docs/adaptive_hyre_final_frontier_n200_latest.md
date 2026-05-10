# Adaptive HyRE Sweep Postprocess

## Latest Logs

| Dataset | Mode | Provider | N | Acc | Gold hit | Empty | Calls | Audit | Detail log |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| barexam | adaptive_snap_hyre | or-gemma4-26b | 200 | 83.0% | 18/200 | 0 | 2.00 | FAIL missing_prediction=1; parse_fail=1 | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_or-gemma4-26b_20260510_0027_barexam_adaptive-hyre-or-gemma4-26b-barexam-n200-k5-cleanretry-adaptive_snap_hyre_detail.jsonl` |
| barexam | adaptive_snap_hyre_v2 | or-gemma4-26b | 200 | 86.0% | 18/200 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_v2_or-gemma4-26b_20260510_0237_barexam_adaptive-hyre-v2-tight-or-gemma4-26b-barexam-n200-k5-repaired_detail.jsonl` |
| barexam | rag_simple | or-gemma4-26b | 200 | 80.0% | 5/200 | 0 | 1.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_2006_barexam_adaptive-hyre-or-gemma4-26b-barexam-n200-k5-rag_simple_detail.jsonl` |
| housing | adaptive_snap_hyre_diverse | or-gemma4-26b | 200 | 63.5% | 92/200 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260509_2059_housing_adaptive-hyre-or-gemma4-26b-housing-n200-k5-adaptive_snap_hyre_diverse_detail.jsonl` |
| housing | rag_state_filter | or-gemma4-26b | 200 | 60.5% | 81/200 | 0 | 1.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_state_filter_or-gemma4-26b_20260509_2006_housing_adaptive-hyre-or-gemma4-26b-housing-n200-k5-rag_state_filter_detail.jsonl` |
| housing | snap_hyre_state | or-gemma4-26b | 200 | 63.0% | 81/200 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_snap_hyre_state_or-gemma4-26b_20260509_2031_housing_adaptive-hyre-or-gemma4-26b-housing-n200-k5-snap_hyre_state_detail.jsonl` |
| casehold | adaptive_snap_hyre_diverse | or-gemma4-26b | 200 | 73.5% | 38/200 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260509_2043_casehold_adaptive-hyre-or-gemma4-26b-casehold-n200-k5-adaptive_snap_hyre_diverse_detail.jsonl` |
| casehold | rag_simple | or-gemma4-26b | 200 | 73.0% | 32/200 | 0 | 1.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_2007_casehold_adaptive-hyre-or-gemma4-26b-casehold-n200-k5-rag_simple_detail.jsonl` |
| legalbench_scalr | adaptive_snap_hyre_anchor | or-gemma4-26b | 200 | 72.5% | 121/200 | 0 | 2.00 | FAIL parse_fail=6 | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_anchor_or-gemma4-26b_20260509_2239_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n200-k5-adaptive_snap_hyre_anchor_detail.jsonl` |
| legalbench_scalr | rag_simple | or-gemma4-26b | 200 | 74.0% | 108/200 | 0 | 1.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_2010_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n200-k5-rag_simple_detail.jsonl` |
| legalbench_scalr | rag_snap_hyde_2call | or-gemma4-26b | 200 | 76.0% | 146/200 | 0 | 2.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260509_2118_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n200-k5-rag_snap_hyde_2call_detail.jsonl` |

## Adaptive Coverage

| Dataset | Provider | Present adaptive modes | Missing adaptive modes | Status |
|---|---|---|---|---|
| barexam | or-gemma4-26b | adaptive_snap_hyre | snap_hyre_option, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse | MISSING |
| housing | or-gemma4-26b | adaptive_snap_hyre_diverse, snap_hyre_state | adaptive_snap_hyre, adaptive_snap_hyre_anchor | MISSING |
| casehold | or-gemma4-26b | adaptive_snap_hyre_diverse | snap_hyre_option, adaptive_snap_hyre, adaptive_snap_hyre_anchor | MISSING |
| legalbench_scalr | or-gemma4-26b | adaptive_snap_hyre_anchor | snap_hyre_option, adaptive_snap_hyre, adaptive_snap_hyre_diverse | MISSING |

## Targeted Run Coverage

| Dataset | Provider | Present modes | Missing modes | Status |
|---|---|---|---|---|
| barexam | or-gemma4-26b | adaptive_snap_hyre_v2, rag_simple | - | READY |
| housing | or-gemma4-26b | adaptive_snap_hyre_diverse, rag_state_filter, snap_hyre_state | - | READY |
| casehold | or-gemma4-26b | adaptive_snap_hyre_diverse, rag_simple | - | READY |
| legalbench_scalr | or-gemma4-26b | rag_simple, rag_snap_hyde_2call | - | READY |

## Adaptive Parity Frontier

| Dataset | Provider | Best control | Acc | Calls | Best adaptive policy | Acc | Calls | Delta pp | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| barexam | or-gemma4-26b | rag_simple | 80.0% | 1.00 | adaptive_snap_hyre_v2 | 86.0% | 2.00 | 6.0 | LEADS |
| housing | or-gemma4-26b | snap_hyre_state | 63.0% | 2.00 | adaptive_snap_hyre_diverse | 63.5% | 2.00 | 0.5 | LEADS |
| casehold | or-gemma4-26b | rag_simple | 73.0% | 1.00 | adaptive_snap_hyre_diverse | 73.5% | 2.00 | 0.5 | LEADS |
| legalbench_scalr | or-gemma4-26b | rag_snap_hyde_2call | 76.0% | 2.00 | - | - | - | - | MISSING |

## Paired Comparisons

| Dataset | Provider | Baseline -> Treatment | N | Baseline Acc | Treatment Acc | Delta pp | b / c | p | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| barexam | or-gemma4-26b | rag_simple -> adaptive_snap_hyre | 200 | 80.0% | 83.0% | 3.0 | 18 / 12 | 0.3616 | [-2.0, 8.5] |
| barexam | or-gemma4-26b | rag_simple -> adaptive_snap_hyre_v2 | 200 | 80.0% | 86.0% | 6.0 | 22 / 10 | 0.0501 | [0.5, 11.5] |
