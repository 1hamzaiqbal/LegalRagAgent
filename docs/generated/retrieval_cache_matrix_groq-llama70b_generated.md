# Retrieval Cache Matrix

| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | qrels | health |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| cache | barexam | llama70b | rag_hyde | 1 | 1195 | 0.0259 | 0.0259 | 0.0259 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | llama70b | rag_hyde | 3 | 1195 | 0.0653 | 0.0653 | 0.0427 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | llama70b | rag_hyde | 5 | 1195 | 0.1046 | 0.1046 | 0.0515 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | llama70b | rag_hyde | 10 | 1195 | 0.1757 | 0.1757 | 0.0609 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | llama70b | snap_hyre | 1 | 1195 | 0.0301 | 0.0301 | 0.0301 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | llama70b | snap_hyre | 3 | 1195 | 0.0720 | 0.0720 | 0.0477 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | llama70b | snap_hyre | 5 | 1195 | 0.1105 | 0.1105 | 0.0564 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | barexam | llama70b | snap_hyre | 10 | 1195 | 0.1849 | 0.1849 | 0.0663 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | llama70b | rag_hyde | 1 | 3600 | 0.3303 | 0.3303 | 0.3303 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | llama70b | rag_hyde | 3 | 3600 | 0.4517 | 0.4517 | 0.3845 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | llama70b | rag_hyde | 5 | 3600 | 0.5122 | 0.5122 | 0.3983 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | llama70b | rag_hyde | 10 | 3600 | 0.5914 | 0.5914 | 0.4090 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | llama70b | snap_hyre | 1 | 3600 | 0.2608 | 0.2608 | 0.2608 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | llama70b | snap_hyre | 3 | 3600 | 0.3844 | 0.3844 | 0.3139 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | llama70b | snap_hyre | 5 | 3600 | 0.4497 | 0.4497 | 0.3286 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | casehold | llama70b | snap_hyre | 10 | 3600 | 0.5289 | 0.5289 | 0.3390 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | llama70b | rag_hyde | 1 | 571 | 0.4046 | 0.4046 | 0.4046 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | llama70b | rag_hyde | 3 | 571 | 0.5762 | 0.5762 | 0.4816 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | llama70b | rag_hyde | 5 | 571 | 0.6147 | 0.6147 | 0.4905 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | llama70b | rag_hyde | 10 | 571 | 0.6953 | 0.6953 | 0.5015 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | llama70b | snap_hyre | 1 | 571 | 0.3135 | 0.3135 | 0.3135 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | llama70b | snap_hyre | 3 | 571 | 0.4658 | 0.4658 | 0.3809 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | llama70b | snap_hyre | 5 | 571 | 0.5517 | 0.5517 | 0.4003 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| cache | legalbench_scalr | llama70b | snap_hyre | 10 | 571 | 0.6462 | 0.6462 | 0.4126 | aligned (1.0000) | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 1 | 10732 | 0.2275 | 0.2275 | 0.2275 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 3 | 10732 | 0.3359 | 0.3359 | 0.2752 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 5 | 10732 | 0.3906 | 0.3906 | 0.2876 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
| macro | macro | macro | all_supplied_caches | 10 | 10732 | 0.4704 | 0.4704 | 0.2982 | mixed () | dup=0, missing_idx=0, empty=0, short=0, no_gold=0 |
