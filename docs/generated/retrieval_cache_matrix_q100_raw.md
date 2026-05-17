# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | barexam | model_invariant | rag_simple | 1 | 100 | 0.0000 | 0.0000 | 0.0000 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | rag_simple | 3 | 100 | 0.0000 | 0.0000 | 0.0000 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | rag_simple | 5 | 100 | 0.0200 | 0.0200 | 0.0045 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | rag_simple | 10 | 100 | 0.0200 | 0.0200 | 0.0045 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | rag_simple | 1 | 100 | 0.0600 | 0.0600 | 0.0600 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | rag_simple | 3 | 100 | 0.0900 | 0.0900 | 0.0750 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | rag_simple | 5 | 100 | 0.1500 | 0.1500 | 0.0885 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | model_invariant | rag_simple | 10 | 100 | 0.2000 | 0.2000 | 0.0951 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | housing | model_invariant | rag_simple | 1 | 100 | 0.0000 | 0.0000 | 0.0000 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | housing | model_invariant | rag_simple | 3 | 100 | 0.0100 | 0.0033 | 0.0050 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | housing | model_invariant | rag_simple | 5 | 100 | 0.0200 | 0.0083 | 0.0075 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | housing | model_invariant | rag_simple | 10 | 100 | 0.0400 | 0.0283 | 0.0097 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_simple | 1 | 100 | 0.2500 | 0.2500 | 0.2500 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_simple | 3 | 100 | 0.4600 | 0.4600 | 0.3400 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_simple | 5 | 100 | 0.5200 | 0.5200 | 0.3535 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_simple | 10 | 100 | 0.6000 | 0.6000 | 0.3642 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 400 | 0.0775 | 0.0775 | 0.0775 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 400 | 0.1400 | 0.1383 | 0.1050 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 400 | 0.1775 | 0.1746 | 0.1135 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 400 | 0.2150 | 0.2121 | 0.1184 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
