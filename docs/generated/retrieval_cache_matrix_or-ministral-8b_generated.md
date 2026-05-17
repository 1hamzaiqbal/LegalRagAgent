# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | legalbench_scalr | model_invariant | rag_hyde | 1 | 571 | 0.3415 | 0.3415 | 0.3415 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_hyde | 3 | 571 | 0.5131 | 0.5131 | 0.4189 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_hyde | 5 | 571 | 0.6025 | 0.6025 | 0.4397 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | rag_hyde | 10 | 571 | 0.6865 | 0.6865 | 0.4506 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | snap_hyre | 1 | 571 | 0.4256 | 0.4256 | 0.4256 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | snap_hyre | 3 | 571 | 0.5692 | 0.5692 | 0.4877 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | snap_hyre | 5 | 571 | 0.6200 | 0.6200 | 0.4996 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | model_invariant | snap_hyre | 10 | 571 | 0.7040 | 0.7040 | 0.5110 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 1142 | 0.3835 | 0.3835 | 0.3835 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 1142 | 0.5412 | 0.5412 | 0.4533 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 1142 | 0.6112 | 0.6112 | 0.4696 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 1142 | 0.6953 | 0.6953 | 0.4808 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
