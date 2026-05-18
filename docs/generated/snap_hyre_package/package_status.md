# Snap-HyRE Package Status

This file is generated from local artifacts only. Missing cells are not inferred from older docs.

- Experiments tag prefix: `local-snap-hyre`
- Minimum answer-row questions: `50`
- Latest answer rows found: 46 / 84 expected cells
- Retrieval matrix rows found: 188

## Answer Ladder

| provider | dataset | llm_only | rag_simple | rag_rewrite | rag_hyde | snap_hyre | golden_passage | golden_plus_neighbors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| or-ministral-8b | barexam | 56.8% | 56.9% | missing | missing | missing | 64.6% | missing |
| or-ministral-8b | housing | missing | missing | missing | missing | missing | missing | missing |
| or-ministral-8b | casehold | missing | missing | missing | missing | missing | missing | missing |
| or-ministral-8b | legalbench_scalr | 67.2% | 68.0% | 69.9% | 71.1% | 69.9% | 93.2% | 77.1% |
| or-gemma4-26b | barexam | 80.8% | 78.0% | 80.7% | 80.2% | 82.0% | 78.6% | 80.7% |
| or-gemma4-26b | housing | missing | missing | missing | missing | missing | missing | missing |
| or-gemma4-26b | casehold | missing | missing | missing | missing | missing | missing | missing |
| or-gemma4-26b | legalbench_scalr | 73.0% | 73.4% | 73.9% | 72.2% | 73.9% | 97.9% | 81.3% |
| groq-llama70b | barexam | 78.7% | 74.6% | 77.2% | 80.2% | 79.8% | 79.2% | 77.8% |
| groq-llama70b | housing | 44.8% | missing | missing | missing | missing | missing | missing |
| groq-llama70b | casehold | 71.8% | 70.8% | 70.6% | 70.3% | 70.5% | 97.5% | 79.4% |
| groq-llama70b | legalbench_scalr | 74.4% | 72.9% | 71.6% | 70.4% | 71.3% | 93.5% | 83.0% |

## Retrieval Top-k Rows

| dataset | model | method | k | rows | Hit@k | MRR@k | qrels | health |
|---|---|---|---:|---:|---:|---:|---|---|
| casehold | model_invariant | golden_plus_neighbors | 1 | 3600 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | golden_plus_neighbors | 3 | 3600 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | golden_plus_neighbors | 5 | 3600 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | golden_plus_neighbors | 10 | 3600 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 1 | 3600 | 6.1% | 0.061 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 3 | 3600 | 13.1% | 0.090 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 5 | 3600 | 17.9% | 0.102 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 10 | 3600 | 26.4% | 0.113 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 1 | 50 | 58.0% | 0.580 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 3 | 50 | 70.0% | 0.637 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 5 | 50 | 74.0% | 0.645 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 10 | 50 | 84.0% | 0.657 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 1 | 50 | 52.0% | 0.520 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 3 | 50 | 64.0% | 0.573 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 5 | 50 | 68.0% | 0.583 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 10 | 50 | 82.0% | 0.602 | aligned | empty=0, short=0, no_gold=0 |
| barexam | llama70b | rag_hyde | 1 | 1195 | 2.6% | 0.026 | aligned | empty=0, short=0, no_gold=0 |
| barexam | llama70b | rag_hyde | 3 | 1195 | 6.5% | 0.043 | aligned | empty=0, short=0, no_gold=0 |
| barexam | llama70b | rag_hyde | 5 | 1195 | 10.5% | 0.052 | aligned | empty=0, short=0, no_gold=0 |
| barexam | llama70b | rag_hyde | 10 | 1195 | 17.6% | 0.061 | aligned | empty=0, short=0, no_gold=0 |
| barexam | llama70b | snap_hyre | 1 | 1195 | 3.0% | 0.030 | aligned | empty=0, short=0, no_gold=0 |
| barexam | llama70b | snap_hyre | 3 | 1195 | 7.2% | 0.048 | aligned | empty=0, short=0, no_gold=0 |
| barexam | llama70b | snap_hyre | 5 | 1195 | 11.0% | 0.056 | aligned | empty=0, short=0, no_gold=0 |
| barexam | llama70b | snap_hyre | 10 | 1195 | 18.5% | 0.066 | aligned | empty=0, short=0, no_gold=0 |
| casehold | llama70b | rag_hyde | 1 | 3600 | 33.0% | 0.330 | aligned | empty=0, short=0, no_gold=0 |
| casehold | llama70b | rag_hyde | 3 | 3600 | 45.2% | 0.384 | aligned | empty=0, short=0, no_gold=0 |
| casehold | llama70b | rag_hyde | 5 | 3600 | 51.2% | 0.398 | aligned | empty=0, short=0, no_gold=0 |
| casehold | llama70b | rag_hyde | 10 | 3600 | 59.1% | 0.409 | aligned | empty=0, short=0, no_gold=0 |
| casehold | llama70b | snap_hyre | 1 | 3600 | 26.1% | 0.261 | aligned | empty=0, short=0, no_gold=0 |
| casehold | llama70b | snap_hyre | 3 | 3600 | 38.4% | 0.314 | aligned | empty=0, short=0, no_gold=0 |
| casehold | llama70b | snap_hyre | 5 | 3600 | 45.0% | 0.329 | aligned | empty=0, short=0, no_gold=0 |
| casehold | llama70b | snap_hyre | 10 | 3600 | 52.9% | 0.339 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | llama70b | rag_hyde | 1 | 571 | 40.5% | 0.405 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | llama70b | rag_hyde | 3 | 571 | 57.6% | 0.482 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | llama70b | rag_hyde | 5 | 571 | 61.5% | 0.491 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | llama70b | rag_hyde | 10 | 571 | 69.5% | 0.502 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | llama70b | snap_hyre | 1 | 571 | 31.3% | 0.313 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | llama70b | snap_hyre | 3 | 571 | 46.6% | 0.381 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | llama70b | snap_hyre | 5 | 571 | 55.2% | 0.400 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | llama70b | snap_hyre | 10 | 571 | 64.6% | 0.413 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 1 | 1195 | 3.2% | 0.032 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 3 | 1195 | 7.9% | 0.051 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 5 | 1195 | 12.1% | 0.060 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 10 | 1195 | 18.7% | 0.069 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 1 | 100 | 3.0% | 0.030 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 2 | 100 | 6.0% | 0.045 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 3 | 100 | 8.0% | 0.052 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 4 | 100 | 14.0% | 0.067 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 5 | 100 | 16.0% | 0.071 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 6 | 100 | 19.0% | 0.076 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 7 | 100 | 20.0% | 0.077 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 8 | 100 | 22.0% | 0.080 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 9 | 100 | 22.0% | 0.080 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | rag_hyde | 10 | 100 | 22.0% | 0.080 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 1 | 100 | 4.0% | 0.040 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 2 | 100 | 6.0% | 0.050 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 3 | 100 | 9.0% | 0.060 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 4 | 100 | 11.0% | 0.065 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 5 | 100 | 14.0% | 0.071 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 6 | 100 | 15.0% | 0.073 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 7 | 100 | 17.0% | 0.076 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 8 | 100 | 20.0% | 0.079 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 9 | 100 | 21.0% | 0.080 | aligned | empty=0, short=0, no_gold=0 |
| barexam | gemma4-26b | snap_hyre | 10 | 100 | 23.0% | 0.082 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 1 | 100 | 0.0% | 0.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 2 | 100 | 0.0% | 0.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 3 | 100 | 0.0% | 0.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 4 | 100 | 1.0% | 0.003 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 5 | 100 | 2.0% | 0.005 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 6 | 100 | 2.0% | 0.005 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 7 | 100 | 2.0% | 0.005 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 8 | 100 | 2.0% | 0.005 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 9 | 100 | 2.0% | 0.005 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 10 | 100 | 2.0% | 0.005 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 1 | 100 | 38.0% | 0.380 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 2 | 100 | 47.0% | 0.425 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 3 | 100 | 52.0% | 0.442 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 4 | 100 | 55.0% | 0.449 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 5 | 100 | 58.0% | 0.455 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 6 | 100 | 61.0% | 0.460 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 7 | 100 | 62.0% | 0.462 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 8 | 100 | 63.0% | 0.463 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 9 | 100 | 63.0% | 0.463 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | rag_hyde | 10 | 100 | 64.0% | 0.464 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 1 | 100 | 41.0% | 0.410 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 2 | 100 | 50.0% | 0.455 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 3 | 100 | 55.0% | 0.472 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 4 | 100 | 56.0% | 0.474 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 5 | 100 | 56.0% | 0.474 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 6 | 100 | 56.0% | 0.474 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 7 | 100 | 56.0% | 0.474 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 8 | 100 | 58.0% | 0.477 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 9 | 100 | 58.0% | 0.477 | aligned | empty=0, short=0, no_gold=0 |
| casehold | gemma4-26b | snap_hyre | 10 | 100 | 59.0% | 0.478 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 1 | 100 | 6.0% | 0.060 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 2 | 100 | 9.0% | 0.075 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 3 | 100 | 9.0% | 0.075 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 4 | 100 | 12.0% | 0.083 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 5 | 100 | 15.0% | 0.089 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 6 | 100 | 16.0% | 0.090 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 7 | 100 | 17.0% | 0.092 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 8 | 100 | 18.0% | 0.093 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 9 | 100 | 20.0% | 0.095 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 10 | 100 | 20.0% | 0.095 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 1 | 100 | 6.0% | 0.060 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 2 | 100 | 12.0% | 0.090 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 3 | 100 | 14.0% | 0.097 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 4 | 100 | 16.0% | 0.102 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 5 | 100 | 18.0% | 0.106 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 6 | 100 | 18.0% | 0.106 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 7 | 100 | 19.0% | 0.107 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 8 | 100 | 20.0% | 0.108 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 9 | 100 | 20.0% | 0.108 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | rag_hyde | 10 | 100 | 21.0% | 0.109 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 1 | 100 | 2.0% | 0.020 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 2 | 100 | 2.0% | 0.020 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 3 | 100 | 2.0% | 0.020 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 4 | 100 | 3.0% | 0.022 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 5 | 100 | 5.0% | 0.027 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 6 | 100 | 5.0% | 0.027 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 7 | 100 | 5.0% | 0.027 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 8 | 100 | 5.0% | 0.027 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 9 | 100 | 5.0% | 0.027 | aligned | empty=0, short=0, no_gold=0 |
| housing | gemma4-26b | snap_hyre | 10 | 100 | 5.0% | 0.027 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 1 | 100 | 0.0% | 0.000 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 2 | 100 | 1.0% | 0.005 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 3 | 100 | 1.0% | 0.005 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 4 | 100 | 2.0% | 0.007 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 5 | 100 | 2.0% | 0.007 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 6 | 100 | 2.0% | 0.007 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 7 | 100 | 2.0% | 0.007 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 8 | 100 | 3.0% | 0.009 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 9 | 100 | 3.0% | 0.009 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 10 | 100 | 4.0% | 0.010 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 1 | 100 | 55.0% | 0.550 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 2 | 100 | 60.0% | 0.575 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 3 | 100 | 63.0% | 0.585 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 4 | 100 | 64.0% | 0.588 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 5 | 100 | 70.0% | 0.600 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 6 | 100 | 73.0% | 0.605 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 7 | 100 | 74.0% | 0.606 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 8 | 100 | 75.0% | 0.607 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 9 | 100 | 75.0% | 0.607 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 10 | 100 | 76.0% | 0.608 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 1 | 100 | 54.0% | 0.540 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 2 | 100 | 62.0% | 0.580 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 3 | 100 | 67.0% | 0.597 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 4 | 100 | 71.0% | 0.607 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 5 | 100 | 73.0% | 0.611 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 6 | 100 | 74.0% | 0.612 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 7 | 100 | 75.0% | 0.614 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 8 | 100 | 77.0% | 0.616 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 9 | 100 | 78.0% | 0.617 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 10 | 100 | 78.0% | 0.617 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 1 | 100 | 25.0% | 0.250 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 2 | 100 | 37.0% | 0.310 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 3 | 100 | 46.0% | 0.340 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 4 | 100 | 49.0% | 0.347 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 5 | 100 | 52.0% | 0.353 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 6 | 100 | 55.0% | 0.358 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 7 | 100 | 55.0% | 0.358 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 8 | 100 | 58.0% | 0.362 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 9 | 100 | 58.0% | 0.362 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 10 | 100 | 60.0% | 0.364 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_hyde | 1 | 571 | 34.2% | 0.342 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_hyde | 3 | 571 | 51.3% | 0.419 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_hyde | 5 | 571 | 60.2% | 0.440 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_hyde | 10 | 571 | 68.7% | 0.451 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | snap_hyre | 1 | 571 | 42.6% | 0.426 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | snap_hyre | 3 | 571 | 56.9% | 0.488 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | snap_hyre | 5 | 571 | 62.0% | 0.500 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | snap_hyre | 10 | 571 | 70.4% | 0.511 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | golden_plus_neighbors | 1 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | golden_plus_neighbors | 3 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | golden_plus_neighbors | 5 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | golden_plus_neighbors | 10 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | golden_plus_neighbors | 1 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | golden_plus_neighbors | 3 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | golden_plus_neighbors | 5 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | golden_plus_neighbors | 10 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | golden_plus_neighbors | 1 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | golden_plus_neighbors | 3 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | golden_plus_neighbors | 5 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | golden_plus_neighbors | 10 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | golden_plus_neighbors | 1 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | golden_plus_neighbors | 3 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | golden_plus_neighbors | 5 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | golden_plus_neighbors | 10 | 100 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |

## Missing Answer Cells

| provider | dataset | mode |
|---|---|---|
| or-ministral-8b | barexam | rag_rewrite |
| or-ministral-8b | barexam | rag_hyde |
| or-ministral-8b | barexam | snap_hyre |
| or-ministral-8b | barexam | golden_plus_neighbors |
| or-ministral-8b | housing | llm_only |
| or-ministral-8b | housing | rag_simple |
| or-ministral-8b | housing | rag_rewrite |
| or-ministral-8b | housing | rag_hyde |
| or-ministral-8b | housing | snap_hyre |
| or-ministral-8b | housing | golden_passage |
| or-ministral-8b | housing | golden_plus_neighbors |
| or-ministral-8b | casehold | llm_only |
| or-ministral-8b | casehold | rag_simple |
| or-ministral-8b | casehold | rag_rewrite |
| or-ministral-8b | casehold | rag_hyde |
| or-ministral-8b | casehold | snap_hyre |
| or-ministral-8b | casehold | golden_passage |
| or-ministral-8b | casehold | golden_plus_neighbors |
| or-gemma4-26b | housing | llm_only |
| or-gemma4-26b | housing | rag_simple |
| or-gemma4-26b | housing | rag_rewrite |
| or-gemma4-26b | housing | rag_hyde |
| or-gemma4-26b | housing | snap_hyre |
| or-gemma4-26b | housing | golden_passage |
| or-gemma4-26b | housing | golden_plus_neighbors |
| or-gemma4-26b | casehold | llm_only |
| or-gemma4-26b | casehold | rag_simple |
| or-gemma4-26b | casehold | rag_rewrite |
| or-gemma4-26b | casehold | rag_hyde |
| or-gemma4-26b | casehold | snap_hyre |
| or-gemma4-26b | casehold | golden_passage |
| or-gemma4-26b | casehold | golden_plus_neighbors |
| groq-llama70b | housing | rag_simple |
| groq-llama70b | housing | rag_rewrite |
| groq-llama70b | housing | rag_hyde |
| groq-llama70b | housing | snap_hyre |
| groq-llama70b | housing | golden_passage |
| groq-llama70b | housing | golden_plus_neighbors |

## Retrieval Coverage Notes
