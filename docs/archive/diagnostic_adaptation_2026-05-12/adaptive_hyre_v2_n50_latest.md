# Adaptive HyRE Sweep Postprocess

## Latest Logs

| Dataset | Mode | Provider | N | Acc | Gold hit | Empty | Calls | Audit | Detail log |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| barexam | adaptive_snap_hyre_v2 | or-gemma4-26b | 50 | 84.0% | 5/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_v2_or-gemma4-26b_20260509_2328_barexam_adaptive-hyre-v2-or-gemma4-26b-barexam-n50-k5-adaptive_snap_hyre_v2_detail.jsonl` |
| housing | adaptive_snap_hyre_v2 | or-gemma4-26b | 50 | 62.0% | 21/50 | 0 | 2.00 | FAIL parse_fail=1 | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_v2_or-gemma4-26b_20260509_2327_housing_adaptive-hyre-v2-or-gemma4-26b-housing-n50-k5-adaptive_snap_hyre_v2_detail.jsonl` |
| casehold | adaptive_snap_hyre_v2 | or-gemma4-26b | 50 | 72.0% | 15/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_v2_or-gemma4-26b_20260509_2318_casehold_adaptive-hyre-v2-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_v2_detail.jsonl` |
| legalbench_scalr | adaptive_snap_hyre_v2 | or-gemma4-26b | 50 | 80.0% | 36/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_v2_or-gemma4-26b_20260509_2313_legalbench_scalr_adaptive-hyre-v2-or-gemma4-26b-legalbench_scalr-n50-k5-adaptive_snap_hyre_v2_detail.jsonl` |

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
| barexam | or-gemma4-26b | adaptive_snap_hyre_v2 | - | READY |
| housing | or-gemma4-26b | adaptive_snap_hyre_v2 | - | MISSING |
| casehold | or-gemma4-26b | adaptive_snap_hyre_v2 | - | READY |
| legalbench_scalr | or-gemma4-26b | adaptive_snap_hyre_v2 | - | READY |

## Adaptive Parity Frontier

| Dataset | Provider | Best control | Acc | Calls | Best adaptive policy | Acc | Calls | Delta pp | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| barexam | or-gemma4-26b | - | - | - | adaptive_snap_hyre_v2 | 84.0% | 2.00 | - | MISSING |
| housing | or-gemma4-26b | - | - | - | adaptive_snap_hyre_v2 | 62.0% | 2.00 | - | MISSING |
| casehold | or-gemma4-26b | - | - | - | adaptive_snap_hyre_v2 | 72.0% | 2.00 | - | MISSING |
| legalbench_scalr | or-gemma4-26b | - | - | - | adaptive_snap_hyre_v2 | 80.0% | 2.00 | - | MISSING |

## Paired Comparisons

| Dataset | Provider | Baseline -> Treatment | N | Baseline Acc | Treatment Acc | Delta pp | b / c | p | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
