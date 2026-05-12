# Router Baseline Report

This report tests whether cheap static task features can predict the oracle method arm.
It is intentionally lightweight: fixed arms, a one-rule stump, and optional small sklearn baselines.

Rows: `800`
Arms: `rag, two_call`
Label column: `oracle_reward_arm`
Include dataset feature: `False`
Include provider feature: `False`
Include subject feature: `False`
Include retrieval-probe features: `True`
Include sklearn baselines: `True`
Policy reward: `correct - 0.02*calls - 0*sec`
Features: `answer_format, question_chars, question_tokens, intermediate_chars, choice_count, avg_choice_chars, max_choice_chars, max_choice_jaccard, avg_choice_question_overlap, named_entity_count, date_number_count, legal_term_count, multi_hop_cue_count, rag_evidence_count, rag_max_ce_score, rag_ce_top1, rag_ce_top2, rag_ce_top5, rag_ce_top1_top2_margin, rag_ce_top1_top5_margin, rag_ce_score_entropy, rag_unique_source_count, rag_source_diversity, rag_retrieved_chars, rag_question_evidence_overlap, rag_choice_evidence_overlap_max, rag_choice_evidence_overlap_margin, two_call_evidence_count, two_call_max_ce_score, two_call_ce_top1, two_call_ce_top2, two_call_ce_top5, two_call_ce_top1_top2_margin, two_call_ce_top1_top5_margin, two_call_ce_score_entropy, two_call_unique_source_count, two_call_source_diversity, two_call_retrieved_chars, two_call_question_evidence_overlap, two_call_choice_evidence_overlap_max, two_call_choice_evidence_overlap_margin`

### Random Split

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 71.5% | 7.5% | 2.00 | two_call=200 |
| majority_oracle_label | 200 | 69.0% | 92.5% | 1.00 | rag=200 |
| decision_stump | 200 | 72.0% | 71.5% | 1.30 | rag=140, two_call=60 |
| logreg | 200 | 68.5% | 61.5% | 1.36 | rag=128, two_call=72 |
| random_forest | 200 | 70.5% | 86.0% | 1.10 | rag=179, two_call=21 |
| oracle_label | 200 | 76.5% | 100.0% | 1.07 | rag=185, two_call=15 |

- static arm from train: `two_call`
- majority oracle label from train: `rag`
- decision stump: `two_call_choice_evidence_overlap_max <= 0.478 -> two_call; else rag`

## Leave-One-Dataset-Out

### Hold Out `barexam`

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 85.5% | 9.5% | 2.00 | two_call=200 |
| majority_oracle_label | 200 | 82.5% | 90.5% | 1.00 | rag=200 |
| decision_stump | 200 | 83.5% | 84.0% | 1.09 | rag=181, two_call=19 |
| logreg | 200 | 83.5% | 84.0% | 1.09 | rag=181, two_call=19 |
| random_forest | 200 | 83.0% | 89.0% | 1.02 | rag=195, two_call=5 |
| oracle_label | 200 | 92.0% | 100.0% | 1.09 | rag=181, two_call=19 |

- static arm from train: `two_call`
- majority oracle label from train: `rag`
- decision stump: `two_call_choice_evidence_overlap_max <= 0.467 -> two_call; else rag`

### Hold Out `casehold`

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 69.5% | 7.0% | 2.00 | two_call=200 |
| majority_oracle_label | 200 | 72.0% | 93.0% | 1.00 | rag=200 |
| decision_stump | 200 | 71.0% | 59.5% | 1.36 | rag=127, two_call=73 |
| logreg | 200 | 70.5% | 67.5% | 1.28 | rag=143, two_call=57 |
| random_forest | 200 | 72.0% | 93.0% | 1.00 | rag=200 |
| oracle_label | 200 | 79.0% | 100.0% | 1.07 | rag=186, two_call=14 |

- static arm from train: `two_call`
- majority oracle label from train: `rag`
- decision stump: `two_call_choice_evidence_overlap_max <= 0.722 -> two_call; else rag`

### Hold Out `legalbench_scalr`

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 75.0% | 4.0% | 2.00 | two_call=200 |
| majority_oracle_label | 200 | 77.0% | 96.0% | 1.00 | rag=200 |
| decision_stump | 200 | 76.5% | 53.0% | 1.47 | rag=106, two_call=94 |
| logreg | 200 | 76.0% | 53.0% | 1.45 | rag=110, two_call=90 |
| random_forest | 200 | 77.0% | 96.0% | 1.00 | rag=200 |
| oracle_label | 200 | 81.0% | 100.0% | 1.04 | rag=192, two_call=8 |

- static arm from train: `two_call`
- majority oracle label from train: `rag`
- decision stump: `rag_ce_top2 <= -1.71 -> rag; else two_call`

### Hold Out `musique`

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 27.5% | 83.5% | 1.00 | rag=200 |
| majority_oracle_label | 200 | 27.5% | 83.5% | 1.00 | rag=200 |
| decision_stump | 200 | 27.5% | 82.5% | 1.01 | rag=198, two_call=2 |
| logreg | 200 | 32.0% | 50.5% | 1.48 | rag=104, two_call=96 |
| random_forest | 200 | 27.0% | 82.5% | 1.01 | rag=198, two_call=2 |
| oracle_label | 200 | 44.0% | 100.0% | 1.17 | rag=167, two_call=33 |

- static arm from train: `rag`
- majority oracle label from train: `rag`
- decision stump: `two_call_ce_score_entropy <= 0.933 -> rag; else two_call`
