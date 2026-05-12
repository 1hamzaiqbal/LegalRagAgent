# CaseHOLD Offline Calibrated Selector Evaluation

## Source Logs

- `frontier`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260510_0838_casehold_adaptive-hyre-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_frontier_detail.jsonl`
- `candidate`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_candidate_verifier_or-gemma4-26b_20260510_1058_casehold_adaptive-hyre-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_candidate_verifier_detail.jsonl`
- `reranker`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_reranker_or-gemma4-26b_20260510_1201_casehold_casehold-option-reranker-cached-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_option_reranker_detail.jsonl`
- `score`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_score_or-gemma4-26b_20260510_1211_casehold_casehold-option-score-cached-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_option_score_detail.jsonl`
- `replay`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_replay_minimal_rule_or-gemma4-26b_20260510_casehold_n50_detail.jsonl`

## Policy Results

| Policy | Accuracy | Route counts |
|---|---:|---|
| `score_margin_2_else_candidate` | 38/50 = 76.0% | candidate=46, score_high_margin=4 |
| `score_margin_2_else_reranker` | 38/50 = 76.0% | reranker=46, score_high_margin=4 |
| `always_candidate` | 37/50 = 74.0% | candidate=50 |
| `always_reranker` | 37/50 = 74.0% | reranker=50 |
| `candidate_reranker_agree_else_candidate` | 37/50 = 74.0% | candidate=7, candidate_reranker_agree=43 |
| `candidate_reranker_snap_agree_else_candidate` | 37/50 = 74.0% | candidate=17, candidate_reranker_snap_agree=33 |
| `majority_candidate_reranker_replay_else_candidate` | 37/50 = 74.0% | majority_candidate_reranker_replay=50 |
| `always_frontier` | 35/50 = 70.0% | frontier=50 |
| `always_replay` | 31/50 = 62.0% | replay=50 |
| `always_score` | 11/50 = 22.0% | score=50 |

## Interpretation

- These policies spend no new retrieval or LLM calls; they only route among existing CaseHOLD selector outputs.
- Any improvement here is a calibration signal, not a new end-to-end method claim until validated on held-out rows.
- A high score-margin override is useful only if it beats the best individual selector and survives held-out validation.
