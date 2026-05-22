# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | barexam | model_invariant | snap_hyre | 1 | 1195 | 0.0226 | 0.0226 | 0.0226 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | snap_hyre | 3 | 1195 | 0.0628 | 0.0628 | 0.0395 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | snap_hyre | 5 | 1195 | 0.0954 | 0.0954 | 0.0469 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | snap_hyre | 10 | 1195 | 0.1481 | 0.1481 | 0.0538 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 1195 | 0.0226 | 0.0226 | 0.0226 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 1195 | 0.0628 | 0.0628 | 0.0395 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 1195 | 0.0954 | 0.0954 | 0.0469 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 1195 | 0.1481 | 0.1481 | 0.0538 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
