# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | barexam | model_invariant | rag_hyde | 1 | 1195 | 0.0117 | 0.0117 | 0.0117 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | rag_hyde | 3 | 1195 | 0.0393 | 0.0393 | 0.0234 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | rag_hyde | 5 | 1195 | 0.0586 | 0.0586 | 0.0277 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | rag_hyde | 10 | 1195 | 0.1046 | 0.1046 | 0.0337 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | snap_hyre | 1 | 1195 | 0.0234 | 0.0234 | 0.0234 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | snap_hyre | 3 | 1195 | 0.0611 | 0.0611 | 0.0389 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | snap_hyre | 5 | 1195 | 0.0862 | 0.0862 | 0.0445 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | model_invariant | snap_hyre | 10 | 1195 | 0.1381 | 0.1381 | 0.0511 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_hyde | 1 | 571 | 0.3415 | 0.3415 | 0.3415 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_hyde | 3 | 571 | 0.5131 | 0.5131 | 0.4189 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_hyde | 5 | 571 | 0.6025 | 0.6025 | 0.4397 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_hyde | 10 | 571 | 0.6865 | 0.6865 | 0.4506 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | snap_hyre | 1 | 571 | 0.4256 | 0.4256 | 0.4256 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | snap_hyre | 3 | 571 | 0.5692 | 0.5692 | 0.4877 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | snap_hyre | 5 | 571 | 0.6200 | 0.6200 | 0.4996 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | snap_hyre | 10 | 571 | 0.7040 | 0.7040 | 0.5110 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 3532 | 0.2006 | 0.2006 | 0.2006 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 3532 | 0.2957 | 0.2957 | 0.2422 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 3532 | 0.3418 | 0.3418 | 0.2529 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 3532 | 0.4083 | 0.4083 | 0.2616 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
