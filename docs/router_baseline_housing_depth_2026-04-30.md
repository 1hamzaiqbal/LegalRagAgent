# Router Baseline Report

This report tests whether cheap static task features can predict the oracle method arm.
It is intentionally lightweight: fixed arms, a one-rule stump, and optional small sklearn baselines.

Rows: `200`
Arms: `top1, top10, top5, two_call`
Label column: `oracle_reward_arm`
Include dataset feature: `False`
Include provider feature: `False`
Include subject feature: `False`
Include retrieval-probe features: `True`
Include sklearn baselines: `True`
Policy reward: `correct - 0.02*calls - 0*sec`
Features: `answer_format, question_chars, question_tokens, intermediate_chars, choice_count, avg_choice_chars, max_choice_chars, max_choice_jaccard, avg_choice_question_overlap, named_entity_count, date_number_count, legal_term_count, multi_hop_cue_count, top1_evidence_count, top1_max_ce_score, top1_ce_top1, top1_ce_top2, top1_ce_top5, top1_ce_top1_top2_margin, top1_ce_top1_top5_margin, top1_ce_score_entropy, top1_unique_source_count, top1_source_diversity, top1_retrieved_chars, top1_question_evidence_overlap, top1_choice_evidence_overlap_max, top1_choice_evidence_overlap_margin, top1_top1_state_match, top1_any_state_match, top1_all_state_match, top1_state_match_frac, top1_unique_retrieved_states, top1_state_diversity, top5_evidence_count, top5_max_ce_score, top5_ce_top1, top5_ce_top2, top5_ce_top5, top5_ce_top1_top2_margin, top5_ce_top1_top5_margin, top5_ce_score_entropy, top5_unique_source_count, top5_source_diversity, top5_retrieved_chars, top5_question_evidence_overlap, top5_choice_evidence_overlap_max, top5_choice_evidence_overlap_margin, top5_top1_state_match, top5_any_state_match, top5_all_state_match, top5_state_match_frac, top5_unique_retrieved_states, top5_state_diversity, top10_evidence_count, top10_max_ce_score, top10_ce_top1, top10_ce_top2, top10_ce_top5, top10_ce_top1_top2_margin, top10_ce_top1_top5_margin, top10_ce_score_entropy, top10_unique_source_count, top10_source_diversity, top10_retrieved_chars, top10_question_evidence_overlap, top10_choice_evidence_overlap_max, top10_choice_evidence_overlap_margin, top10_top1_state_match, top10_any_state_match, top10_all_state_match, top10_state_match_frac, top10_unique_retrieved_states, top10_state_diversity, two_call_evidence_count, two_call_max_ce_score, two_call_ce_top1, two_call_ce_top2, two_call_ce_top5, two_call_ce_top1_top2_margin, two_call_ce_top1_top5_margin, two_call_ce_score_entropy, two_call_unique_source_count, two_call_source_diversity, two_call_retrieved_chars, two_call_question_evidence_overlap, two_call_choice_evidence_overlap_max, two_call_choice_evidence_overlap_margin, two_call_top1_state_match, two_call_any_state_match, two_call_all_state_match, two_call_state_match_frac, two_call_unique_retrieved_states, two_call_state_diversity`

## Read

- Oracle headroom is large on this N=200 slice: random-split oracle reaches
  80.0% accuracy at 1.06 calls/q.
- Simple learned baselines are still unstable. Logistic regression reaches
  62.0% on the random split, but this is a small single-dataset split and not a
  deployable controller result.
- The table is still useful because state metadata features are now present in
  the router CSV. The next fair controller test should add the new
  `rag_state_filter` arm once its cluster logs land.

### Random Split

| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |
|---|---:|---:|---:|---:|---|
| static_best_train | 50 | 54.0% | 6.0% | 2.00 | two_call=50 |
| majority_oracle_label | 50 | 52.0% | 36.0% | 1.00 | top1=50 |
| decision_stump | 50 | 54.0% | 12.0% | 1.64 | top10=18, two_call=32 |
| logreg | 50 | 62.0% | 30.0% | 1.34 | top1=15, top10=9, top5=9, two_call=17 |
| random_forest | 50 | 56.0% | 40.0% | 1.20 | top1=16, top10=9, top5=15, two_call=10 |
| oracle_label | 50 | 80.0% | 100.0% | 1.06 | top1=18, top10=10, top5=19, two_call=3 |

- static arm from train: `two_call`
- majority oracle label from train: `top1`
- decision stump: `top5_ce_top5 <= 0.772 -> top10; else two_call`
