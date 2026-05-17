# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | legalbench_scalr | gemma4-26b | rag_hyde | 1 | 50 | 0.5800 | 0.5800 | 0.5800 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | gemma4-26b | rag_hyde | 3 | 50 | 0.7000 | 0.7000 | 0.6367 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | gemma4-26b | rag_hyde | 5 | 50 | 0.7400 | 0.7400 | 0.6447 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | gemma4-26b | rag_hyde | 10 | 50 | 0.8400 | 0.8400 | 0.6565 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | gemma4-26b | snap_hyre | 1 | 50 | 0.5200 | 0.5200 | 0.5200 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | gemma4-26b | snap_hyre | 3 | 50 | 0.6400 | 0.6400 | 0.5733 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | gemma4-26b | snap_hyre | 5 | 50 | 0.6800 | 0.6800 | 0.5833 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | gemma4-26b | snap_hyre | 10 | 50 | 0.8200 | 0.8200 | 0.6017 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 100 | 0.5500 | 0.5500 | 0.5500 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 100 | 0.6700 | 0.6700 | 0.6050 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 100 | 0.7100 | 0.7100 | 0.6140 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 100 | 0.8300 | 0.8300 | 0.6291 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
