# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | legal_link_eu | gemma4-26b | rag_hyde | 1 | 1127 | 0.2866 | 0.0575 | 0.2866 | alignment_missing () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legal_link_eu | gemma4-26b | rag_hyde | 3 | 1127 | 0.4312 | 0.1394 | 0.3503 | alignment_missing () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legal_link_eu | gemma4-26b | rag_hyde | 5 | 1127 | 0.4898 | 0.1885 | 0.3637 | alignment_missing () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legal_link_eu | gemma4-26b | rag_hyde | 10 | 1127 | 0.5892 | 0.2580 | 0.3771 | alignment_missing () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legal_link_eu | gemma4-26b | snap_hyre | 1 | 1127 | 0.4685 | 0.0938 | 0.4685 | alignment_missing () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legal_link_eu | gemma4-26b | snap_hyre | 3 | 1127 | 0.6149 | 0.2153 | 0.5328 | alignment_missing () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legal_link_eu | gemma4-26b | snap_hyre | 5 | 1127 | 0.6788 | 0.2896 | 0.5473 | alignment_missing () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legal_link_eu | gemma4-26b | snap_hyre | 10 | 1127 | 0.7684 | 0.3827 | 0.5588 | alignment_missing () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 2254 | 0.3776 | 0.0757 | 0.3776 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 2254 | 0.5231 | 0.1774 | 0.4416 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 2254 | 0.5843 | 0.2390 | 0.4555 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 2254 | 0.6788 | 0.3203 | 0.4679 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
