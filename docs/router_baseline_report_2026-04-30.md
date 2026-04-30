# Router Baseline Report

This report tests whether cheap static task features can predict the oracle method arm.
It is intentionally lightweight: a fixed static arm, a majority-oracle label, and a one-rule decision stump.

Rows: `800`
Arms: `rag, two_call`
Label column: `oracle_reward_arm`
Include dataset feature: `False`
Include provider feature: `False`
Include subject feature: `False`
Include retrieval-probe features: `True`
Policy reward: `correct - 0.02*calls - 0*sec`
Features: `answer_format, question_chars, question_tokens, intermediate_chars, choice_count, avg_choice_chars, max_choice_chars, named_entity_count, date_number_count, legal_term_count, multi_hop_cue_count, rag_evidence_count, rag_max_ce_score, two_call_evidence_count, two_call_max_ce_score`

### Random Split

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 71.5% | 7.5% | 2.00 | two_call=200 |
| majority_oracle_label | 200 | 69.0% | 92.5% | 1.00 | rag=200 |
| decision_stump | 200 | 74.5% | 49.0% | 1.56 | rag=87, two_call=113 |
| oracle_label | 200 | 76.5% | 100.0% | 1.07 | rag=185, two_call=15 |

- static arm from train: `two_call`
- majority oracle label from train: `rag`
- decision stump: `max_choice_chars <= 188 -> two_call; else rag`

## Leave-One-Dataset-Out

### Hold Out `barexam`

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 85.5% | 9.5% | 2.00 | two_call=200 |
| majority_oracle_label | 200 | 82.5% | 90.5% | 1.00 | rag=200 |
| decision_stump | 200 | 81.0% | 70.0% | 1.25 | rag=151, two_call=49 |
| oracle_label | 200 | 92.0% | 100.0% | 1.09 | rag=181, two_call=19 |

- static arm from train: `two_call`
- majority oracle label from train: `rag`
- decision stump: `question_chars <= 150 -> two_call; else rag`

### Hold Out `casehold`

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 69.5% | 7.0% | 2.00 | two_call=200 |
| majority_oracle_label | 200 | 72.0% | 93.0% | 1.00 | rag=200 |
| decision_stump | 200 | 71.5% | 66.0% | 1.32 | rag=136, two_call=64 |
| oracle_label | 200 | 79.0% | 100.0% | 1.07 | rag=186, two_call=14 |

- static arm from train: `two_call`
- majority oracle label from train: `rag`
- decision stump: `max_choice_chars <= 198 -> two_call; else rag`

### Hold Out `legalbench_scalr`

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 75.0% | 4.0% | 2.00 | two_call=200 |
| majority_oracle_label | 200 | 77.0% | 96.0% | 1.00 | rag=200 |
| decision_stump | 200 | 76.0% | 40.0% | 1.60 | rag=80, two_call=120 |
| oracle_label | 200 | 81.0% | 100.0% | 1.04 | rag=192, two_call=8 |

- static arm from train: `two_call`
- majority oracle label from train: `rag`
- decision stump: `rag_max_ce_score <= -1.14 -> rag; else two_call`

### Hold Out `musique`

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 200 | 27.5% | 83.5% | 1.00 | rag=200 |
| majority_oracle_label | 200 | 27.5% | 83.5% | 1.00 | rag=200 |
| decision_stump | 200 | 37.0% | 16.5% | 2.00 | two_call=200 |
| oracle_label | 200 | 44.0% | 100.0% | 1.17 | rag=167, two_call=33 |

- static arm from train: `rag`
- majority oracle label from train: `rag`
- decision stump: `two_call_max_ce_score <= 7.23 -> rag; else two_call`
