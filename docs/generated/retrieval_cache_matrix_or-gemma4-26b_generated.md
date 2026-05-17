# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | barexam | gemma4-26b | snap_hyre | 1 | 1195 | 0.0318 | 0.0318 | 0.0318 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | gemma4-26b | snap_hyre | 3 | 1195 | 0.0795 | 0.0795 | 0.0509 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | gemma4-26b | snap_hyre | 5 | 1195 | 0.1205 | 0.1205 | 0.0602 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | gemma4-26b | snap_hyre | 10 | 1195 | 0.1866 | 0.1866 | 0.0691 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 1195 | 0.0318 | 0.0318 | 0.0318 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 1195 | 0.0795 | 0.0795 | 0.0509 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 1195 | 0.1205 | 0.1205 | 0.0602 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 1195 | 0.1866 | 0.1866 | 0.0691 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
