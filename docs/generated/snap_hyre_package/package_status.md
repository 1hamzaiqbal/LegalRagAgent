# Snap-HyRE Package Status

This file is generated from local artifacts only. Missing cells are not inferred from older docs.

- Experiments tag prefix: `local-snap-hyre`
- Minimum answer-row questions: `50`
- Latest answer rows found: 7 / 84 expected cells
- Retrieval matrix rows found: 28

## Answer Ladder

| provider | dataset | llm_only | rag_simple | rag_rewrite | rag_hyde | snap_hyre | golden_passage | golden_plus_neighbors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| or-gemma3n-e4b | barexam | missing | missing | missing | missing | missing | missing | missing |
| or-gemma3n-e4b | housing | missing | missing | missing | missing | missing | missing | missing |
| or-gemma3n-e4b | casehold | missing | missing | missing | missing | missing | missing | missing |
| or-gemma3n-e4b | legalbench_scalr | missing | missing | missing | missing | missing | missing | missing |
| or-gemma4-26b | barexam | missing | missing | missing | missing | missing | missing | missing |
| or-gemma4-26b | housing | missing | missing | missing | missing | missing | missing | missing |
| or-gemma4-26b | casehold | missing | missing | missing | missing | missing | missing | missing |
| or-gemma4-26b | legalbench_scalr | 76.0% | 76.0% | 74.0% | 78.0% | 80.0% | 74.0% (caveated) | 78.0% (failed_tag) |
| groq-llama70b | barexam | missing | missing | missing | missing | missing | missing | missing |
| groq-llama70b | housing | missing | missing | missing | missing | missing | missing | missing |
| groq-llama70b | casehold | missing | missing | missing | missing | missing | missing | missing |
| groq-llama70b | legalbench_scalr | missing | missing | missing | missing | missing | missing | missing |

## Retrieval Top-k Rows

| dataset | model | method | k | rows | Hit@k | MRR@k | qrels | health |
|---|---|---|---:|---:|---:|---:|---|---|
| barexam | model_invariant | golden_plus_neighbors | 1 | 1195 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | golden_plus_neighbors | 3 | 1195 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | golden_plus_neighbors | 5 | 1195 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | golden_plus_neighbors | 10 | 1195 | 100.0% | 1.000 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 1 | 1195 | 0.3% | 0.003 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 3 | 1195 | 0.9% | 0.006 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 5 | 1195 | 1.4% | 0.007 | aligned | empty=0, short=0, no_gold=0 |
| barexam | model_invariant | rag_simple | 10 | 1195 | 2.2% | 0.008 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 1 | 3600 | 6.1% | 0.061 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 3 | 3600 | 13.1% | 0.091 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 5 | 3600 | 17.9% | 0.102 | aligned | empty=0, short=0, no_gold=0 |
| casehold | model_invariant | rag_simple | 10 | 3600 | 26.3% | 0.113 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 1 | 6853 | 0.8% | 0.008 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 3 | 6853 | 1.9% | 0.013 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 5 | 6853 | 2.8% | 0.015 | aligned | empty=0, short=0, no_gold=0 |
| housing | model_invariant | rag_simple | 10 | 6853 | 5.1% | 0.018 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 1 | 571 | 25.4% | 0.254 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 3 | 571 | 42.6% | 0.329 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 5 | 571 | 49.6% | 0.345 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | model_invariant | rag_simple | 10 | 571 | 59.4% | 0.358 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 1 | 50 | 58.0% | 0.580 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 3 | 50 | 70.0% | 0.637 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 5 | 50 | 74.0% | 0.645 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | rag_hyde | 10 | 50 | 84.0% | 0.657 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 1 | 50 | 52.0% | 0.520 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 3 | 50 | 64.0% | 0.573 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 5 | 50 | 68.0% | 0.583 | aligned | empty=0, short=0, no_gold=0 |
| legalbench_scalr | gemma4-26b | snap_hyre | 10 | 50 | 82.0% | 0.602 | aligned | empty=0, short=0, no_gold=0 |

## Missing Answer Cells

| provider | dataset | mode |
|---|---|---|
| or-gemma3n-e4b | barexam | llm_only |
| or-gemma3n-e4b | barexam | rag_simple |
| or-gemma3n-e4b | barexam | rag_rewrite |
| or-gemma3n-e4b | barexam | rag_hyde |
| or-gemma3n-e4b | barexam | snap_hyre |
| or-gemma3n-e4b | barexam | golden_passage |
| or-gemma3n-e4b | barexam | golden_plus_neighbors |
| or-gemma3n-e4b | housing | llm_only |
| or-gemma3n-e4b | housing | rag_simple |
| or-gemma3n-e4b | housing | rag_rewrite |
| or-gemma3n-e4b | housing | rag_hyde |
| or-gemma3n-e4b | housing | snap_hyre |
| or-gemma3n-e4b | housing | golden_passage |
| or-gemma3n-e4b | housing | golden_plus_neighbors |
| or-gemma3n-e4b | casehold | llm_only |
| or-gemma3n-e4b | casehold | rag_simple |
| or-gemma3n-e4b | casehold | rag_rewrite |
| or-gemma3n-e4b | casehold | rag_hyde |
| or-gemma3n-e4b | casehold | snap_hyre |
| or-gemma3n-e4b | casehold | golden_passage |
| or-gemma3n-e4b | casehold | golden_plus_neighbors |
| or-gemma3n-e4b | legalbench_scalr | llm_only |
| or-gemma3n-e4b | legalbench_scalr | rag_simple |
| or-gemma3n-e4b | legalbench_scalr | rag_rewrite |
| or-gemma3n-e4b | legalbench_scalr | rag_hyde |
| or-gemma3n-e4b | legalbench_scalr | snap_hyre |
| or-gemma3n-e4b | legalbench_scalr | golden_passage |
| or-gemma3n-e4b | legalbench_scalr | golden_plus_neighbors |
| or-gemma4-26b | barexam | llm_only |
| or-gemma4-26b | barexam | rag_simple |
| or-gemma4-26b | barexam | rag_rewrite |
| or-gemma4-26b | barexam | rag_hyde |
| or-gemma4-26b | barexam | snap_hyre |
| or-gemma4-26b | barexam | golden_passage |
| or-gemma4-26b | barexam | golden_plus_neighbors |
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
| groq-llama70b | barexam | llm_only |
| groq-llama70b | barexam | rag_simple |
| groq-llama70b | barexam | rag_rewrite |
| groq-llama70b | barexam | rag_hyde |
| groq-llama70b | barexam | snap_hyre |
| groq-llama70b | barexam | golden_passage |
| groq-llama70b | barexam | golden_plus_neighbors |
| groq-llama70b | housing | llm_only |
| groq-llama70b | housing | rag_simple |
| groq-llama70b | housing | rag_rewrite |
| groq-llama70b | housing | rag_hyde |
| groq-llama70b | housing | snap_hyre |
| groq-llama70b | housing | golden_passage |
| groq-llama70b | housing | golden_plus_neighbors |
| groq-llama70b | casehold | llm_only |
| groq-llama70b | casehold | rag_simple |
| groq-llama70b | casehold | rag_rewrite |
| groq-llama70b | casehold | rag_hyde |
| groq-llama70b | casehold | snap_hyre |
| groq-llama70b | casehold | golden_passage |
| groq-llama70b | casehold | golden_plus_neighbors |
| groq-llama70b | legalbench_scalr | llm_only |
| groq-llama70b | legalbench_scalr | rag_simple |
| groq-llama70b | legalbench_scalr | rag_rewrite |
| groq-llama70b | legalbench_scalr | rag_hyde |
| groq-llama70b | legalbench_scalr | snap_hyre |
| groq-llama70b | legalbench_scalr | golden_passage |
| groq-llama70b | legalbench_scalr | golden_plus_neighbors |

## Retrieval Coverage Notes

- `barexam` missing retrieval rows for: rag_hyde, snap_hyre
- `housing` missing retrieval rows for: golden_plus_neighbors, rag_hyde, snap_hyre
- `casehold` missing retrieval rows for: golden_plus_neighbors, rag_hyde, snap_hyre
- `legalbench_scalr` missing retrieval rows for: golden_plus_neighbors
