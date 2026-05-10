# Adaptive HyRE Sweep Postprocess

## Latest Logs

| Dataset | Mode | Provider | N | Acc | Gold hit | Empty | Calls | Audit | Detail log |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| barexam | adaptive_snap_hyre_frontier | or-gemma4-26b | 50 | 96.0% | 5/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0332_barexam_adaptive-hyre-frontier-or-gemma4-26b-barexam-n50-k5-adaptive_snap_hyre_frontier_detail.jsonl` |
| housing | adaptive_snap_hyre_frontier | or-gemma4-26b | 50 | 56.0% | 24/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0337_housing_adaptive-hyre-frontier-or-gemma4-26b-housing-n50-k5-adaptive_snap_hyre_frontier_detail.jsonl` |
| casehold | adaptive_snap_hyre_frontier | or-gemma4-26b | 50 | 72.0% | 14/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0339_casehold-adaptive-hyre-frontier-or-gemma4-26b-casehold-n50-k5-repaired_detail.jsonl` |
| legalbench_scalr | adaptive_snap_hyre_frontier | or-gemma4-26b | 50 | 74.0% | 37/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0330_legalbench_scalr_adaptive-hyre-frontier-or-gemma4-26b-legalbench_scalr-n50-k5-adaptive_snap_hyre_frontier_detail.jsonl` |

## Adaptive Coverage

| Dataset | Provider | Present adaptive modes | Missing adaptive modes | Status |
|---|---|---|---|---|
| barexam | or-gemma4-26b | - | snap_hyre_option, adaptive_snap_hyre, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse | MISSING |
| housing | or-gemma4-26b | - | snap_hyre_state, adaptive_snap_hyre, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse | MISSING |
| casehold | or-gemma4-26b | - | snap_hyre_option, adaptive_snap_hyre, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse | MISSING |
| legalbench_scalr | or-gemma4-26b | - | snap_hyre_option, adaptive_snap_hyre, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse | MISSING |

## Targeted Run Coverage

| Dataset | Provider | Present modes | Missing modes | Status |
|---|---|---|---|---|
| barexam | or-gemma4-26b | adaptive_snap_hyre_frontier | - | READY |
| housing | or-gemma4-26b | adaptive_snap_hyre_frontier | - | READY |
| casehold | or-gemma4-26b | adaptive_snap_hyre_frontier | - | READY |
| legalbench_scalr | or-gemma4-26b | adaptive_snap_hyre_frontier | - | READY |

## Adaptive Parity Frontier

| Dataset | Provider | Best control | Acc | Calls | Best adaptive policy | Acc | Calls | Delta pp | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| barexam | or-gemma4-26b | - | - | - | adaptive_snap_hyre_frontier | 96.0% | 2.00 | - | MISSING |
| housing | or-gemma4-26b | - | - | - | adaptive_snap_hyre_frontier | 56.0% | 2.00 | - | MISSING |
| casehold | or-gemma4-26b | - | - | - | adaptive_snap_hyre_frontier | 72.0% | 2.00 | - | MISSING |
| legalbench_scalr | or-gemma4-26b | - | - | - | adaptive_snap_hyre_frontier | 74.0% | 2.00 | - | MISSING |

## Paired Comparisons

| Dataset | Provider | Baseline -> Treatment | N | Baseline Acc | Treatment Acc | Delta pp | b / c | p | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
