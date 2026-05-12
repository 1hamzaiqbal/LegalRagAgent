# Legal RAG Diagnostic Table

This table is generated from detail logs. It separates retrieval exposure, answer conversion, and call budget for bottleneck-aware routing.

| Dataset | Method | N | Acc | Gold retrieved | Gold retrieved but wrong | Gold missing but correct | Acc if gold retrieved | Acc if gold missing | R@1 | R@5 | R@10 | MRR | Calls | Health |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| barexam | `rag_simple` | 200 | 80.0% | 5/200 | 2 | 157 | 60.0% | 80.5% | 20.0% | 100.0% | 100.0% | 0.533 | 1.00 | PASS |
| barexam | `adaptive_snap_hyre_v2` | 200 | 86.0% | 18/200 | 1 | 155 | 94.4% | 85.2% | 38.9% | 100.0% | 100.0% | 0.594 | 2.00 | PASS |
| housing | `rag_state_filter` | 200 | 60.5% | 81/200 | 33 | 73 | 59.3% | 61.3% | 43.3% | 100.0% | 100.0% | 0.626 | 1.00 | PASS |
| housing | `adaptive_snap_hyre_diverse` | 200 | 63.5% | 92/200 | 34 | 69 | 63.0% | 63.9% | 44.4% | 100.0% | 100.0% | 0.650 | 2.00 | PASS |
| housing | `adaptive_snap_hyre_housing_verifier` | 200 | 74.5% | 89/200 | 25 | 85 | 71.9% | 76.6% | 45.7% | 100.0% | 100.0% | 0.630 | 1.00 | PASS |
| casehold | `rag_simple` | 200 | 73.0% | 32/200 | 3 | 117 | 90.6% | 69.6% | 34.4% | 100.0% | 100.0% | 0.566 | 1.00 | PASS |
| casehold | `adaptive_snap_hyre_diverse` | 200 | 73.5% | 38/200 | 3 | 112 | 92.1% | 69.1% | 36.8% | 100.0% | 100.0% | 0.570 | 2.00 | PASS |
| casehold | `adaptive_snap_hyre_frontier` | 200 | 70.5% | 42/200 | 3 | 102 | 92.9% | 64.6% | 31.0% | 100.0% | 100.0% | 0.544 | 2.00 | PASS |
| legalbench_scalr | `rag_simple` | 200 | 74.0% | 108/200 | 13 | 53 | 88.0% | 57.6% | 51.9% | 100.0% | 100.0% | 0.705 | 1.00 | PASS |
| legalbench_scalr | `rag_snap_hyde_2call` | 200 | 76.0% | 146/200 | 15 | 21 | 89.7% | 38.9% | 73.3% | 100.0% | 100.0% | 0.829 | 2.00 | PASS |
| legalbench_scalr | `adaptive_snap_hyre_frontier` | 200 | 76.5% | 153/200 | 13 | 13 | 91.5% | 27.7% | 75.8% | 100.0% | 100.0% | 0.844 | 2.00 | PASS |
| legalbench_scalr | `adaptive_snap_hyre_disagreement_replay` | 200 | 77.5% | 0/0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0.19 | PASS |
| legalbench_scalr | `adaptive_snap_hyre_disagreement_majority_prior` | 200 | 77.5% | 0/0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0.19 | PASS |

## Source Logs

- barexam / `rag_simple`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_2006_barexam_adaptive-hyre-or-gemma4-26b-barexam-n200-k5-rag_simple_detail.jsonl`
- barexam / `adaptive_snap_hyre_v2`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_v2_or-gemma4-26b_20260510_0237_barexam_adaptive-hyre-v2-tight-or-gemma4-26b-barexam-n200-k5-repaired_detail.jsonl`
- housing / `rag_state_filter`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_state_filter_or-gemma4-26b_20260509_2006_housing_adaptive-hyre-or-gemma4-26b-housing-n200-k5-rag_state_filter_detail.jsonl`
- housing / `adaptive_snap_hyre_diverse`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260509_2059_housing_adaptive-hyre-or-gemma4-26b-housing-n200-k5-adaptive_snap_hyre_diverse_detail.jsonl`
- housing / `adaptive_snap_hyre_housing_verifier`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_housing_verifier_or-gemma4-26b_20260510_1037_housing_housing-verifier-cached-or-gemma4-26b-housing-n200-k5-adaptive_snap_hyre_housing_verifier_detail.jsonl`
- casehold / `rag_simple`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_2007_casehold_adaptive-hyre-or-gemma4-26b-casehold-n200-k5-rag_simple_detail.jsonl`
- casehold / `adaptive_snap_hyre_diverse`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260509_2043_casehold_adaptive-hyre-or-gemma4-26b-casehold-n200-k5-adaptive_snap_hyre_diverse_detail.jsonl`
- casehold / `adaptive_snap_hyre_frontier`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0546_casehold_adaptive-hyre-or-gemma4-26b-casehold-n200-k5-adaptive_snap_hyre_frontier_detail.jsonl`
- legalbench_scalr / `rag_simple`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_simple_or-gemma4-26b_20260509_2010_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n200-k5-rag_simple_detail.jsonl`
- legalbench_scalr / `rag_snap_hyde_2call`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260509_2118_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n200-k5-rag_snap_hyde_2call_detail.jsonl`
- legalbench_scalr / `adaptive_snap_hyre_frontier`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0542_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n200-k5-adaptive_snap_hyre_frontier_detail.jsonl`
- legalbench_scalr / `adaptive_snap_hyre_disagreement_replay`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_disagreement_arbitrator_or-gemma4-26b_20260510_scalr_n200_detail.jsonl`
- legalbench_scalr / `adaptive_snap_hyre_disagreement_majority_prior`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_disagreement_arbitrator_majority_prior_or-gemma4-26b_20260510_scalr_n200_detail.jsonl`
