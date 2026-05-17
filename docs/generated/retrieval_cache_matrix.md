# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | casehold | model_invariant | golden_plus_neighbors | 1 | 3600 | 1.0000 | 1.0000 | 1.0000 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | golden_plus_neighbors | 3 | 3600 | 1.0000 | 1.0000 | 1.0000 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | golden_plus_neighbors | 5 | 3600 | 1.0000 | 1.0000 | 1.0000 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | golden_plus_neighbors | 10 | 3600 | 1.0000 | 1.0000 | 1.0000 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | rag_simple | 1 | 3600 | 0.0606 | 0.0606 | 0.0606 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | rag_simple | 3 | 3600 | 0.1308 | 0.1308 | 0.0905 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | rag_simple | 5 | 3600 | 0.1794 | 0.1794 | 0.1015 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | rag_simple | 10 | 3600 | 0.2639 | 0.2639 | 0.1125 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 7200 | 0.5303 | 0.5303 | 0.5303 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 7200 | 0.5654 | 0.5654 | 0.5452 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 7200 | 0.5897 | 0.5897 | 0.5508 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 7200 | 0.6319 | 0.6319 | 0.5563 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
