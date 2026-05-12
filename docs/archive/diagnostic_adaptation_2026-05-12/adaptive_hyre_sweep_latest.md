# Adaptive HyRE Sweep Postprocess

## Latest Logs

| Dataset | Mode | Provider | N | Acc | Gold hit | Empty | Calls | Audit | Detail log |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| barexam | adaptive_snap_hyre | or-gemma4-26b | 50 | 86.0% | 4/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_or-gemma4-26b_20260509_1746_detail.jsonl` |
| barexam | adaptive_snap_hyre_anchor | or-gemma4-26b | 50 | 78.0% | 2/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_anchor_or-gemma4-26b_20260509_1750_detail.jsonl` |
| barexam | adaptive_snap_hyre_diverse | or-gemma4-26b | 50 | 82.0% | 2/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260509_1755_detail.jsonl` |
| barexam | rag_simple | or-gemma4-26b | 50 | 86.0% | 1/50 | 0 | 1.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_1736_detail.jsonl` |
| barexam | rag_snap_hyde_2call | or-gemma4-26b | 50 | 78.0% | 3/50 | 0 | 2.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260509_1749_detail.jsonl` |
| barexam | snap_hyre_option | or-gemma4-26b | 50 | 80.0% | 5/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_snap_hyre_option_or-gemma4-26b_20260509_1835_barexam_adaptive-hyre-or-gemma4-26b-barexam-n50-k5-snap_hyre_option_detail.jsonl` |
| housing | adaptive_snap_hyre | or-gemma4-26b | 50 | 62.0% | 20/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_or-gemma4-26b_20260509_1753_detail.jsonl` |
| housing | adaptive_snap_hyre_anchor | or-gemma4-26b | 50 | 60.0% | 25/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_anchor_or-gemma4-26b_20260509_1754_detail.jsonl` |
| housing | adaptive_snap_hyre_diverse | or-gemma4-26b | 50 | 64.0% | 22/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260509_1833_housing_adaptive-hyre-or-gemma4-26b-housing-n50-k5-adaptive_snap_hyre_diverse_detail.jsonl` |
| housing | rag_state_filter | or-gemma4-26b | 50 | 62.0% | 21/50 | 0 | 1.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_state_filter_or-gemma4-26b_20260509_1752_detail.jsonl` |
| housing | snap_hyre_state | or-gemma4-26b | 50 | 64.0% | 19/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_snap_hyre_state_or-gemma4-26b_20260509_1825_detail.jsonl` |
| casehold | adaptive_snap_hyre | or-gemma4-26b | 50 | 64.0% | 16/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_or-gemma4-26b_20260509_1741_detail.jsonl` |
| casehold | adaptive_snap_hyre_anchor | or-gemma4-26b | 50 | 70.0% | 15/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_anchor_or-gemma4-26b_20260509_1742_detail.jsonl` |
| casehold | adaptive_snap_hyre_diverse | or-gemma4-26b | 50 | 70.0% | 13/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260509_1828_casehold_adaptive-hyre-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_diverse_detail.jsonl` |
| casehold | rag_simple | or-gemma4-26b | 50 | 70.0% | 13/50 | 0 | 1.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_1745_detail.jsonl` |
| casehold | rag_snap_hyde_2call | or-gemma4-26b | 50 | 66.0% | 32/50 | 0 | 2.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260509_1746_detail.jsonl` |
| casehold | snap_hyre_option | or-gemma4-26b | 50 | 66.0% | 27/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_snap_hyre_option_or-gemma4-26b_20260509_1740_detail.jsonl` |
| legalbench_scalr | adaptive_snap_hyre | or-gemma4-26b | 50 | 72.0% | 29/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_or-gemma4-26b_20260509_1758_detail.jsonl` |
| legalbench_scalr | adaptive_snap_hyre_anchor | or-gemma4-26b | 50 | 78.0% | 30/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_anchor_or-gemma4-26b_20260509_1807_detail.jsonl` |
| legalbench_scalr | adaptive_snap_hyre_diverse | or-gemma4-26b | 50 | 72.0% | 30/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260509_1806_detail.jsonl` |
| legalbench_scalr | rag_simple | or-gemma4-26b | 50 | 72.0% | 29/50 | 0 | 1.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_1749_detail.jsonl` |
| legalbench_scalr | rag_snap_hyde_2call | or-gemma4-26b | 50 | 76.0% | 37/50 | 0 | 2.00 | - | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260509_1759_detail.jsonl` |
| legalbench_scalr | snap_hyre_option | or-gemma4-26b | 50 | 72.0% | 38/50 | 0 | 2.00 | PASS | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_snap_hyre_option_or-gemma4-26b_20260509_1842_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n50-k5-snap_hyre_option_detail.jsonl` |

## Adaptive Coverage

| Dataset | Provider | Present adaptive modes | Missing adaptive modes | Status |
|---|---|---|---|---|
| barexam | or-gemma4-26b | adaptive_snap_hyre, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse, snap_hyre_option | - | READY |
| housing | or-gemma4-26b | adaptive_snap_hyre, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse, snap_hyre_state | - | READY |
| casehold | or-gemma4-26b | adaptive_snap_hyre, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse, snap_hyre_option | - | READY |
| legalbench_scalr | or-gemma4-26b | adaptive_snap_hyre, adaptive_snap_hyre_anchor, adaptive_snap_hyre_diverse, snap_hyre_option | - | READY |

## Adaptive Parity Frontier

| Dataset | Provider | Best control | Acc | Calls | Best adaptive policy | Acc | Calls | Delta pp | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| barexam | or-gemma4-26b | rag_simple | 86.0% | 1.00 | adaptive_snap_hyre | 86.0% | 2.00 | 0.0 | PARITY |
| housing | or-gemma4-26b | snap_hyre_state | 64.0% | 2.00 | adaptive_snap_hyre_diverse | 64.0% | 2.00 | 0.0 | PARITY |
| casehold | or-gemma4-26b | rag_simple | 70.0% | 1.00 | adaptive_snap_hyre_diverse | 70.0% | 2.00 | 0.0 | PARITY |
| legalbench_scalr | or-gemma4-26b | rag_snap_hyde_2call | 76.0% | 2.00 | adaptive_snap_hyre_anchor | 78.0% | 2.00 | 2.0 | LEADS |

## Paired Comparisons

| Dataset | Provider | Baseline -> Treatment | N | Baseline Acc | Treatment Acc | Delta pp | b / c | p | 95% CI pp |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| barexam | or-gemma4-26b | rag_simple -> adaptive_snap_hyre | 50 | 86.0% | 86.0% | 0.0 | 5 / 5 | 1 | [-12.0, 12.0] |
| barexam | or-gemma4-26b | rag_snap_hyde_2call -> adaptive_snap_hyre | 50 | 78.0% | 86.0% | 8.0 | 6 / 2 | 0.2891 | [-2.0, 20.0] |
| barexam | or-gemma4-26b | snap_hyre_option -> adaptive_snap_hyre | 50 | 80.0% | 86.0% | 6.0 | 6 / 3 | 0.5078 | [-6.0, 18.0] |
| barexam | or-gemma4-26b | adaptive_snap_hyre -> adaptive_snap_hyre_anchor | 50 | 86.0% | 78.0% | -8.0 | 3 / 7 | 0.3438 | [-20.0, 4.0] |
| barexam | or-gemma4-26b | adaptive_snap_hyre_anchor -> adaptive_snap_hyre_diverse | 50 | 78.0% | 82.0% | 4.0 | 6 / 4 | 0.7539 | [-8.0, 16.0] |
| casehold | or-gemma4-26b | rag_simple -> adaptive_snap_hyre | 50 | 70.0% | 64.0% | -6.0 | 3 / 6 | 0.5078 | [-18.0, 4.0] |
| casehold | or-gemma4-26b | rag_snap_hyde_2call -> adaptive_snap_hyre | 50 | 66.0% | 64.0% | -2.0 | 1 / 2 | 1 | [-10.0, 4.0] |
| casehold | or-gemma4-26b | snap_hyre_option -> adaptive_snap_hyre | 50 | 66.0% | 64.0% | -2.0 | 2 / 3 | 1 | [-10.0, 6.0] |
| casehold | or-gemma4-26b | adaptive_snap_hyre -> adaptive_snap_hyre_anchor | 50 | 64.0% | 70.0% | 6.0 | 4 / 1 | 0.375 | [-2.0, 16.0] |
| casehold | or-gemma4-26b | adaptive_snap_hyre_anchor -> adaptive_snap_hyre_diverse | 50 | 70.0% | 70.0% | 0.0 | 2 / 2 | 1 | [-8.0, 8.0] |
| legalbench_scalr | or-gemma4-26b | rag_simple -> adaptive_snap_hyre | 50 | 72.0% | 72.0% | 0.0 | 6 / 6 | 1 | [-14.0, 14.0] |
| legalbench_scalr | or-gemma4-26b | rag_snap_hyde_2call -> adaptive_snap_hyre | 50 | 76.0% | 72.0% | -4.0 | 2 / 4 | 0.6875 | [-14.0, 6.0] |
| legalbench_scalr | or-gemma4-26b | snap_hyre_option -> adaptive_snap_hyre | 50 | 72.0% | 72.0% | 0.0 | 3 / 3 | 1 | [-10.0, 10.0] |
| legalbench_scalr | or-gemma4-26b | adaptive_snap_hyre -> adaptive_snap_hyre_anchor | 50 | 72.0% | 78.0% | 6.0 | 7 / 4 | 0.5488 | [-6.0, 18.0] |
| legalbench_scalr | or-gemma4-26b | adaptive_snap_hyre_anchor -> adaptive_snap_hyre_diverse | 50 | 78.0% | 72.0% | -6.0 | 1 / 4 | 0.375 | [-14.0, 2.0] |
| housing | or-gemma4-26b | rag_state_filter -> adaptive_snap_hyre | 50 | 62.0% | 62.0% | 0.0 | 4 / 4 | 1 | [-10.0, 12.0] |
| housing | or-gemma4-26b | snap_hyre_state -> adaptive_snap_hyre | 50 | 64.0% | 62.0% | -2.0 | 4 / 5 | 1 | [-12.0, 8.0] |
| housing | or-gemma4-26b | adaptive_snap_hyre -> adaptive_snap_hyre_anchor | 50 | 62.0% | 60.0% | -2.0 | 3 / 4 | 1 | [-12.0, 8.0] |
| housing | or-gemma4-26b | adaptive_snap_hyre_anchor -> adaptive_snap_hyre_diverse | 50 | 60.0% | 64.0% | 4.0 | 4 / 2 | 0.6875 | [-6.0, 14.0] |
