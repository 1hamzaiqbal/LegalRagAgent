# BEIR Phase 1b Model Breadth - 2026-05-26

## Scope

Model-breadth analysis over the five BEIR Phase 1 datasets using the same raw-question retrieval caches, Chroma collections, gte-large embeddings, and MiniLM cross-encoder reranking. No files under `paper/` were edited.

Models included: `or-gemma4-26b` (Gemma 4 26B), `or-qwen3p5-9b` (Qwen 3.5 9B), `or-mistral-small-3p2-24b` (Mistral Small 3.2 24B).

## Cross-Model Verdicts

| Claim | Verdict | Key numbers |
|---|---|---|
| Gold-affinity mechanism (HyDE) | **supported** | Gemma 4 26B rho=0.501, Mistral Small 3.2 24B rho=0.512, Qwen 3.5 9B rho=0.478 |
| P4 geometry-not-hallucination (HyDE) | **supported** | Gemma 4 26B geom=0.944/quality=0.520, Mistral Small 3.2 24B geom=0.944/quality=0.499, Qwen 3.5 9B geom=0.945/quality=0.492 |
| Gold-affinity mechanism (SCOPE) | **supported** | Gemma 4 26B rho=0.426, Mistral Small 3.2 24B rho=0.390, Qwen 3.5 9B rho=0.393 |
| P4 geometry-not-hallucination (SCOPE) | **supported** | Gemma 4 26B geom=0.909/quality=0.509, Mistral Small 3.2 24B geom=0.910/quality=0.508, Qwen 3.5 9B geom=0.922/quality=0.528 |
| SCOPE robustness over HyDE | **supported** | Gemma 4 26B net_gap=19.0%, closer0=3.807, Mistral Small 3.2 24B net_gap=17.0%, closer0=2.716, Qwen 3.5 9B net_gap=19.6%, closer0=4.348 |

## Clean-Output And Cache Health

Generation status: **clean**. Retrieval status: **clean**.

Execution notes:
- Qwen generation used `OPENROUTER_DISABLE_REASONING=1` after the sanity check exposed hidden-reasoning / blank-visible-content routing; the committed caches passed the clean-output checks.
- Mistral generation hit an upstream 429 on the default route and was resumed on the OpenRouter `Mistral` provider with `EVAL_LLM_MIN_CALL_INTERVAL_SEC=0.75`; the completed caches are clean and are not marked provisional.

| Model | Dataset | Expansion | Generation rows | Errors | Missing passage | Parse bad | Answer artifacts | Format retries | Max output tokens | Retrieval rows | Short retrieval rows | Provider mismatches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B | SciFact | HyDE | 300 | 0 | 0 | 0 | 0 | 0 | 129 | 300 | 0 | 0 |
| Gemma 4 26B | SciFact | SCOPE | 300 | 0 | 0 | 0 | 0 | 0 | 234 | 300 | 0 | 0 |
| Gemma 4 26B | NFCorpus | HyDE | 323 | 0 | 0 | 0 | 0 | 0 | 121 | 323 | 0 | 0 |
| Gemma 4 26B | NFCorpus | SCOPE | 323 | 0 | 0 | 0 | 0 | 0 | 214 | 323 | 0 | 0 |
| Gemma 4 26B | FiQA | HyDE | 648 | 0 | 0 | 0 | 0 | 0 | 130 | 648 | 0 | 0 |
| Gemma 4 26B | FiQA | SCOPE | 648 | 0 | 0 | 0 | 0 | 2 | 356 | 648 | 0 | 0 |
| Gemma 4 26B | TREC-COVID | HyDE | 50 | 0 | 0 | 0 | 0 | 0 | 118 | 50 | 0 | 0 |
| Gemma 4 26B | TREC-COVID | SCOPE | 50 | 0 | 0 | 0 | 0 | 0 | 237 | 50 | 0 | 0 |
| Gemma 4 26B | SciDocs | HyDE | 1000 | 0 | 0 | 0 | 0 | 0 | 135 | 1000 | 0 | 0 |
| Gemma 4 26B | SciDocs | SCOPE | 1000 | 0 | 0 | 0 | 0 | 0 | 225 | 1000 | 0 | 0 |
| Qwen 3.5 9B | SciFact | HyDE | 300 | 0 | 0 | 0 | 0 | 0 | 170 | 300 | 0 | 0 |
| Qwen 3.5 9B | SciFact | SCOPE | 300 | 0 | 0 | 0 | 0 | 0 | 272 | 300 | 0 | 0 |
| Qwen 3.5 9B | NFCorpus | HyDE | 323 | 0 | 0 | 0 | 0 | 0 | 152 | 323 | 0 | 0 |
| Qwen 3.5 9B | NFCorpus | SCOPE | 323 | 0 | 0 | 0 | 0 | 0 | 239 | 323 | 0 | 0 |
| Qwen 3.5 9B | FiQA | HyDE | 648 | 0 | 0 | 0 | 0 | 0 | 160 | 648 | 0 | 0 |
| Qwen 3.5 9B | FiQA | SCOPE | 648 | 0 | 0 | 0 | 0 | 0 | 276 | 648 | 0 | 0 |
| Qwen 3.5 9B | TREC-COVID | HyDE | 50 | 0 | 0 | 0 | 0 | 0 | 140 | 50 | 0 | 0 |
| Qwen 3.5 9B | TREC-COVID | SCOPE | 50 | 0 | 0 | 0 | 0 | 0 | 269 | 50 | 0 | 0 |
| Qwen 3.5 9B | SciDocs | HyDE | 1000 | 0 | 0 | 0 | 0 | 1 | 181 | 1000 | 0 | 0 |
| Qwen 3.5 9B | SciDocs | SCOPE | 1000 | 0 | 0 | 0 | 0 | 0 | 242 | 1000 | 0 | 0 |
| Mistral Small 3.2 24B | SciFact | HyDE | 300 | 0 | 0 | 0 | 0 | 0 | 144 | 300 | 0 | 0 |
| Mistral Small 3.2 24B | SciFact | SCOPE | 300 | 0 | 0 | 0 | 0 | 0 | 272 | 300 | 0 | 0 |
| Mistral Small 3.2 24B | NFCorpus | HyDE | 323 | 0 | 0 | 0 | 0 | 0 | 150 | 323 | 0 | 0 |
| Mistral Small 3.2 24B | NFCorpus | SCOPE | 323 | 0 | 0 | 0 | 0 | 0 | 231 | 323 | 0 | 0 |
| Mistral Small 3.2 24B | FiQA | HyDE | 648 | 0 | 0 | 0 | 0 | 0 | 151 | 648 | 0 | 0 |
| Mistral Small 3.2 24B | FiQA | SCOPE | 648 | 0 | 0 | 0 | 0 | 0 | 247 | 648 | 0 | 0 |
| Mistral Small 3.2 24B | TREC-COVID | HyDE | 50 | 0 | 0 | 0 | 0 | 0 | 131 | 50 | 0 | 0 |
| Mistral Small 3.2 24B | TREC-COVID | SCOPE | 50 | 0 | 0 | 0 | 0 | 0 | 224 | 50 | 0 | 0 |
| Mistral Small 3.2 24B | SciDocs | HyDE | 1000 | 0 | 0 | 0 | 0 | 0 | 151 | 1000 | 0 | 0 |
| Mistral Small 3.2 24B | SciDocs | SCOPE | 1000 | 0 | 0 | 0 | 0 | 0 | 250 | 1000 | 0 | 0 |

## Retrieval Outcomes

| Model | Dataset | Expansion | N | Raw Hit@5 | Expansion Hit@5 | Net Hit@5 | Help | Hurt | RI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B | FiQA | HyDE | 648 | 66.2% | 32.3% | -34.0% | 38 | 258 | -0.340 |
| Gemma 4 26B | FiQA | SCOPE | 648 | 66.2% | 35.2% | -31.0% | 25 | 226 | -0.310 |
| Gemma 4 26B | NFCorpus | HyDE | 323 | 69.3% | 33.4% | -35.9% | 6 | 122 | -0.359 |
| Gemma 4 26B | NFCorpus | SCOPE | 323 | 69.3% | 65.0% | -4.3% | 20 | 34 | -0.043 |
| Gemma 4 26B | SciDocs | HyDE | 1000 | 49.0% | 25.5% | -23.5% | 58 | 293 | -0.235 |
| Gemma 4 26B | SciDocs | SCOPE | 1000 | 49.0% | 47.1% | -1.9% | 87 | 106 | -0.019 |
| Gemma 4 26B | SciFact | HyDE | 300 | 82.0% | 35.0% | -47.0% | 12 | 153 | -0.470 |
| Gemma 4 26B | SciFact | SCOPE | 300 | 82.0% | 65.7% | -16.3% | 12 | 61 | -0.163 |
| Gemma 4 26B | TREC-COVID | HyDE | 50 | 98.0% | 70.0% | -28.0% | 1 | 15 | -0.280 |
| Gemma 4 26B | TREC-COVID | SCOPE | 50 | 98.0% | 96.0% | -2.0% | 1 | 2 | -0.020 |
| Mistral Small 3.2 24B | FiQA | HyDE | 648 | 66.2% | 32.9% | -33.3% | 30 | 246 | -0.333 |
| Mistral Small 3.2 24B | FiQA | SCOPE | 648 | 66.2% | 39.5% | -26.7% | 27 | 200 | -0.267 |
| Mistral Small 3.2 24B | NFCorpus | HyDE | 323 | 69.3% | 36.8% | -32.5% | 12 | 117 | -0.325 |
| Mistral Small 3.2 24B | NFCorpus | SCOPE | 323 | 69.3% | 66.6% | -2.8% | 23 | 32 | -0.028 |
| Mistral Small 3.2 24B | SciDocs | HyDE | 1000 | 49.0% | 25.4% | -23.6% | 54 | 290 | -0.236 |
| Mistral Small 3.2 24B | SciDocs | SCOPE | 1000 | 49.0% | 43.5% | -5.5% | 79 | 134 | -0.055 |
| Mistral Small 3.2 24B | SciFact | HyDE | 300 | 82.0% | 50.0% | -32.0% | 14 | 110 | -0.320 |
| Mistral Small 3.2 24B | SciFact | SCOPE | 300 | 82.0% | 72.0% | -10.0% | 20 | 50 | -0.100 |
| Mistral Small 3.2 24B | TREC-COVID | HyDE | 50 | 98.0% | 76.0% | -22.0% | 1 | 12 | -0.220 |
| Mistral Small 3.2 24B | TREC-COVID | SCOPE | 50 | 98.0% | 92.0% | -6.0% | 1 | 4 | -0.060 |
| Qwen 3.5 9B | FiQA | HyDE | 648 | 66.2% | 28.9% | -37.3% | 24 | 266 | -0.373 |
| Qwen 3.5 9B | FiQA | SCOPE | 648 | 66.2% | 38.0% | -28.2% | 34 | 217 | -0.282 |
| Qwen 3.5 9B | NFCorpus | HyDE | 323 | 69.3% | 41.8% | -27.6% | 10 | 99 | -0.276 |
| Qwen 3.5 9B | NFCorpus | SCOPE | 323 | 69.3% | 66.3% | -3.1% | 19 | 29 | -0.031 |
| Qwen 3.5 9B | SciDocs | HyDE | 1000 | 49.0% | 23.7% | -25.3% | 54 | 307 | -0.253 |
| Qwen 3.5 9B | SciDocs | SCOPE | 1000 | 49.0% | 45.7% | -3.3% | 83 | 116 | -0.033 |
| Qwen 3.5 9B | SciFact | HyDE | 300 | 82.0% | 42.7% | -39.3% | 10 | 128 | -0.393 |
| Qwen 3.5 9B | SciFact | SCOPE | 300 | 82.0% | 70.0% | -12.0% | 16 | 52 | -0.120 |
| Qwen 3.5 9B | TREC-COVID | HyDE | 50 | 98.0% | 64.0% | -34.0% | 0 | 17 | -0.340 |
| Qwen 3.5 9B | TREC-COVID | SCOPE | 50 | 98.0% | 92.0% | -6.0% | 1 | 4 | -0.060 |
| Gemma 4 26B | Pooled | HyDE | 2321 | 62.0% | 30.7% | -31.3% | 115 | 841 | -0.313 |
| Gemma 4 26B | Pooled | SCOPE | 2321 | 62.0% | 49.7% | -12.2% | 145 | 429 | -0.122 |
| Mistral Small 3.2 24B | Pooled | HyDE | 2321 | 62.0% | 33.3% | -28.6% | 111 | 775 | -0.286 |
| Mistral Small 3.2 24B | Pooled | SCOPE | 2321 | 62.0% | 50.3% | -11.6% | 150 | 420 | -0.116 |
| Qwen 3.5 9B | Pooled | HyDE | 2321 | 62.0% | 31.0% | -31.0% | 98 | 817 | -0.310 |
| Qwen 3.5 9B | Pooled | SCOPE | 2321 | 62.0% | 50.5% | -11.4% | 153 | 418 | -0.114 |

## Mechanism Correlations

| Model | Dataset | Expansion | N | Mean CE gold delta | Gold rho | Gold tau | Margin-valid N | Mean deltaM | DeltaM rho | DeltaM tau |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B | FiQA | HyDE | 648 | -4.055 | 0.565 | 0.457 | 648 | -1.352 | 0.543 | 0.441 |
| Gemma 4 26B | FiQA | SCOPE | 648 | -2.947 | 0.505 | 0.411 | 648 | -1.798 | 0.537 | 0.440 |
| Gemma 4 26B | NFCorpus | HyDE | 323 | -5.005 | 0.406 | 0.331 | 310 | -2.313 | 0.419 | 0.341 |
| Gemma 4 26B | NFCorpus | SCOPE | 323 | -0.919 | 0.296 | 0.241 | 307 | -0.983 | 0.198 | 0.159 |
| Gemma 4 26B | SciDocs | HyDE | 1000 | -3.269 | 0.476 | 0.382 | 1000 | 1.208 | 0.501 | 0.402 |
| Gemma 4 26B | SciDocs | SCOPE | 1000 | 1.302 | 0.299 | 0.240 | 1000 | 1.179 | 0.349 | 0.281 |
| Gemma 4 26B | SciFact | HyDE | 300 | -7.360 | 0.475 | 0.388 | 300 | -2.879 | 0.455 | 0.373 |
| Gemma 4 26B | SciFact | SCOPE | 300 | -0.909 | 0.329 | 0.270 | 300 | -2.408 | 0.411 | 0.337 |
| Gemma 4 26B | TREC-COVID | HyDE | 50 | -7.662 | 0.313 | 0.255 | 27 | -0.663 | 0.336 | 0.284 |
| Gemma 4 26B | TREC-COVID | SCOPE | 50 | -1.824 | 0.108 | 0.088 | 27 | -0.066 | 0.516 | 0.434 |
| Mistral Small 3.2 24B | FiQA | HyDE | 648 | -2.602 | 0.510 | 0.414 | 648 | -1.544 | 0.525 | 0.425 |
| Mistral Small 3.2 24B | FiQA | SCOPE | 648 | -1.347 | 0.428 | 0.347 | 648 | -1.510 | 0.490 | 0.399 |
| Mistral Small 3.2 24B | NFCorpus | HyDE | 323 | -3.562 | 0.505 | 0.410 | 309 | -2.270 | 0.437 | 0.353 |
| Mistral Small 3.2 24B | NFCorpus | SCOPE | 323 | -0.323 | 0.277 | 0.225 | 310 | -0.885 | 0.252 | 0.204 |
| Mistral Small 3.2 24B | SciDocs | HyDE | 1000 | -2.341 | 0.495 | 0.395 | 1000 | 1.231 | 0.504 | 0.404 |
| Mistral Small 3.2 24B | SciDocs | SCOPE | 1000 | 1.466 | 0.351 | 0.283 | 1000 | 1.076 | 0.393 | 0.317 |
| Mistral Small 3.2 24B | SciFact | HyDE | 300 | -4.550 | 0.582 | 0.477 | 300 | -2.580 | 0.534 | 0.438 |
| Mistral Small 3.2 24B | SciFact | SCOPE | 300 | -0.027 | 0.379 | 0.310 | 300 | -1.872 | 0.464 | 0.380 |
| Mistral Small 3.2 24B | TREC-COVID | HyDE | 50 | -3.529 | 0.287 | 0.230 | 25 | -0.248 | 0.537 | 0.442 |
| Mistral Small 3.2 24B | TREC-COVID | SCOPE | 50 | -1.324 | 0.037 | 0.028 | 21 | 0.245 | 0.029 | 0.015 |
| Qwen 3.5 9B | FiQA | HyDE | 648 | -4.219 | 0.509 | 0.413 | 648 | -1.590 | 0.503 | 0.408 |
| Qwen 3.5 9B | FiQA | SCOPE | 648 | -2.172 | 0.500 | 0.407 | 648 | -1.699 | 0.515 | 0.419 |
| Qwen 3.5 9B | NFCorpus | HyDE | 323 | -4.230 | 0.384 | 0.312 | 310 | -2.062 | 0.343 | 0.276 |
| Qwen 3.5 9B | NFCorpus | SCOPE | 323 | -0.865 | 0.234 | 0.191 | 310 | -0.750 | 0.167 | 0.134 |
| Qwen 3.5 9B | SciDocs | HyDE | 1000 | -4.293 | 0.473 | 0.377 | 1000 | 1.545 | 0.494 | 0.395 |
| Qwen 3.5 9B | SciDocs | SCOPE | 1000 | 1.143 | 0.292 | 0.234 | 1000 | 1.234 | 0.334 | 0.268 |
| Qwen 3.5 9B | SciFact | HyDE | 300 | -7.139 | 0.547 | 0.449 | 300 | -2.395 | 0.451 | 0.369 |
| Qwen 3.5 9B | SciFact | SCOPE | 300 | -0.603 | 0.324 | 0.264 | 300 | -2.096 | 0.440 | 0.362 |
| Qwen 3.5 9B | TREC-COVID | HyDE | 50 | -7.695 | 0.414 | 0.341 | 28 | -1.033 | 0.052 | 0.043 |
| Qwen 3.5 9B | TREC-COVID | SCOPE | 50 | -2.074 | -0.118 | -0.096 | 25 | 0.149 | 0.167 | 0.130 |
| Gemma 4 26B | Pooled | HyDE | 2321 | -4.354 | 0.501 | 0.404 | 2285 | -0.555 | 0.517 | 0.418 |
| Gemma 4 26B | Pooled | SCOPE | 2321 | -0.546 | 0.426 | 0.346 | 2282 | -0.444 | 0.442 | 0.358 |
| Mistral Small 3.2 24B | Pooled | HyDE | 2321 | -2.895 | 0.512 | 0.413 | 2282 | -0.548 | 0.509 | 0.410 |
| Mistral Small 3.2 24B | Pooled | SCOPE | 2321 | 0.179 | 0.390 | 0.316 | 2279 | -0.321 | 0.432 | 0.350 |
| Qwen 3.5 9B | Pooled | HyDE | 2321 | -4.705 | 0.478 | 0.385 | 2286 | -0.381 | 0.473 | 0.381 |
| Qwen 3.5 9B | Pooled | SCOPE | 2321 | -0.357 | 0.393 | 0.318 | 2283 | -0.317 | 0.417 | 0.338 |

## P4 Failure Model

| Model | Dataset | Expansion | Target | N | Failures | AUC OOV/logPPL | AUC geometry | Pseudo-R2 OOV/logPPL | Pseudo-R2 geometry |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B | FiQA | HyDE | deltaM<0 | 648 | 412 | 0.544 | 0.936 | 0.006 | 0.534 |
| Gemma 4 26B | FiQA | HyDE | retrieval hurt | 648 | 258 | 0.548 | 0.815 | 0.004 | 0.241 |
| Gemma 4 26B | FiQA | SCOPE | deltaM<0 | 648 | 423 | 0.527 | 0.923 | 0.003 | 0.491 |
| Gemma 4 26B | FiQA | SCOPE | retrieval hurt | 648 | 226 | 0.563 | 0.751 | 0.005 | 0.135 |
| Gemma 4 26B | NFCorpus | HyDE | deltaM<0 | 310 | 202 | 0.574 | 0.956 | 0.009 | 0.602 |
| Gemma 4 26B | NFCorpus | HyDE | retrieval hurt | 311 | 121 | 0.566 | 0.762 | 0.019 | 0.156 |
| Gemma 4 26B | NFCorpus | SCOPE | deltaM<0 | 307 | 188 | 0.548 | 0.881 | 0.008 | 0.378 |
| Gemma 4 26B | NFCorpus | SCOPE | retrieval hurt | 311 | 34 | 0.536 | 0.674 | 0.003 | 0.047 |
| Gemma 4 26B | SciDocs | HyDE | deltaM<0 | 1000 | 400 | 0.508 | 0.949 | 0.000 | 0.588 |
| Gemma 4 26B | SciDocs | HyDE | retrieval hurt | 1000 | 293 | 0.536 | 0.816 | 0.002 | 0.230 |
| Gemma 4 26B | SciDocs | SCOPE | deltaM<0 | 1000 | 369 | 0.507 | 0.909 | 0.001 | 0.443 |
| Gemma 4 26B | SciDocs | SCOPE | retrieval hurt | 1000 | 106 | 0.503 | 0.701 | 0.000 | 0.077 |
| Gemma 4 26B | SciFact | HyDE | deltaM<0 | 300 | 225 | 0.570 | 0.964 | 0.016 | 0.625 |
| Gemma 4 26B | SciFact | HyDE | retrieval hurt | 300 | 153 | 0.532 | 0.839 | 0.003 | 0.271 |
| Gemma 4 26B | SciFact | SCOPE | deltaM<0 | 300 | 213 | 0.572 | 0.889 | 0.024 | 0.391 |
| Gemma 4 26B | SciFact | SCOPE | retrieval hurt | 300 | 61 | 0.538 | 0.819 | 0.004 | 0.209 |
| Gemma 4 26B | TREC-COVID | HyDE | deltaM<0 | 27 | 17 | 0.488 | 0.753 | 0.001 | 0.148 |
| Gemma 4 26B | TREC-COVID | HyDE | retrieval hurt | 28 | 7 | 0.638 | 0.741 | 0.044 | 0.099 |
| Gemma 4 26B | TREC-COVID | SCOPE | deltaM<0 | 27 | 13 | 0.665 | 0.835 | 0.069 | 0.261 |
| Gemma 4 26B | TREC-COVID | SCOPE | retrieval hurt | 28 | 2 | 0.635 | 0.865 | 0.007 | 0.150 |
| Mistral Small 3.2 24B | FiQA | HyDE | deltaM<0 | 648 | 418 | 0.525 | 0.943 | 0.001 | 0.555 |
| Mistral Small 3.2 24B | FiQA | HyDE | retrieval hurt | 648 | 246 | 0.515 | 0.799 | 0.001 | 0.216 |
| Mistral Small 3.2 24B | FiQA | SCOPE | deltaM<0 | 648 | 423 | 0.516 | 0.911 | 0.000 | 0.460 |
| Mistral Small 3.2 24B | FiQA | SCOPE | retrieval hurt | 648 | 200 | 0.547 | 0.714 | 0.003 | 0.092 |
| Mistral Small 3.2 24B | NFCorpus | HyDE | deltaM<0 | 309 | 210 | 0.536 | 0.933 | 0.003 | 0.517 |
| Mistral Small 3.2 24B | NFCorpus | HyDE | retrieval hurt | 311 | 112 | 0.558 | 0.791 | 0.019 | 0.199 |
| Mistral Small 3.2 24B | NFCorpus | SCOPE | deltaM<0 | 310 | 196 | 0.558 | 0.868 | 0.008 | 0.351 |
| Mistral Small 3.2 24B | NFCorpus | SCOPE | retrieval hurt | 311 | 32 | 0.535 | 0.705 | 0.004 | 0.063 |
| Mistral Small 3.2 24B | SciDocs | HyDE | deltaM<0 | 1000 | 388 | 0.526 | 0.962 | 0.002 | 0.634 |
| Mistral Small 3.2 24B | SciDocs | HyDE | retrieval hurt | 1000 | 290 | 0.550 | 0.838 | 0.003 | 0.271 |
| Mistral Small 3.2 24B | SciDocs | SCOPE | deltaM<0 | 1000 | 372 | 0.502 | 0.928 | 0.000 | 0.498 |
| Mistral Small 3.2 24B | SciDocs | SCOPE | retrieval hurt | 1000 | 134 | 0.518 | 0.761 | 0.001 | 0.119 |
| Mistral Small 3.2 24B | SciFact | HyDE | deltaM<0 | 300 | 209 | 0.512 | 0.942 | 0.001 | 0.533 |
| Mistral Small 3.2 24B | SciFact | HyDE | retrieval hurt | 300 | 110 | 0.510 | 0.887 | 0.006 | 0.383 |
| Mistral Small 3.2 24B | SciFact | SCOPE | deltaM<0 | 300 | 195 | 0.542 | 0.889 | 0.013 | 0.390 |
| Mistral Small 3.2 24B | SciFact | SCOPE | retrieval hurt | 300 | 50 | 0.561 | 0.749 | 0.002 | 0.112 |
| Mistral Small 3.2 24B | TREC-COVID | HyDE | deltaM<0 | 25 | 14 | 0.753 | 0.844 | 0.100 | 0.328 |
| Mistral Small 3.2 24B | TREC-COVID | HyDE | retrieval hurt | 28 | 6 | 0.533 | 0.803 | 0.006 | 0.254 |
| Mistral Small 3.2 24B | TREC-COVID | SCOPE | deltaM<0 | 21 | 7 | 0.673 | 0.714 | 0.046 | 0.115 |
| Mistral Small 3.2 24B | TREC-COVID | SCOPE | retrieval hurt | 28 | 4 | 0.565 | 0.802 | 0.002 | 0.186 |
| Qwen 3.5 9B | FiQA | HyDE | deltaM<0 | 648 | 419 | 0.501 | 0.944 | 0.000 | 0.555 |
| Qwen 3.5 9B | FiQA | HyDE | retrieval hurt | 648 | 266 | 0.533 | 0.796 | 0.002 | 0.208 |
| Qwen 3.5 9B | FiQA | SCOPE | deltaM<0 | 648 | 414 | 0.526 | 0.925 | 0.002 | 0.496 |
| Qwen 3.5 9B | FiQA | SCOPE | retrieval hurt | 648 | 217 | 0.532 | 0.749 | 0.000 | 0.138 |
| Qwen 3.5 9B | NFCorpus | HyDE | deltaM<0 | 310 | 197 | 0.570 | 0.950 | 0.007 | 0.579 |
| Qwen 3.5 9B | NFCorpus | HyDE | retrieval hurt | 311 | 95 | 0.550 | 0.734 | 0.012 | 0.123 |
| Qwen 3.5 9B | NFCorpus | SCOPE | deltaM<0 | 310 | 179 | 0.548 | 0.895 | 0.008 | 0.419 |
| Qwen 3.5 9B | NFCorpus | SCOPE | retrieval hurt | 311 | 29 | 0.543 | 0.664 | 0.002 | 0.046 |
| Qwen 3.5 9B | SciDocs | HyDE | deltaM<0 | 1000 | 378 | 0.520 | 0.961 | 0.001 | 0.630 |
| Qwen 3.5 9B | SciDocs | HyDE | retrieval hurt | 1000 | 307 | 0.523 | 0.835 | 0.001 | 0.266 |
| Qwen 3.5 9B | SciDocs | SCOPE | deltaM<0 | 1000 | 348 | 0.502 | 0.935 | 0.000 | 0.524 |
| Qwen 3.5 9B | SciDocs | SCOPE | retrieval hurt | 1000 | 116 | 0.533 | 0.694 | 0.002 | 0.058 |
| Qwen 3.5 9B | SciFact | HyDE | deltaM<0 | 300 | 206 | 0.538 | 0.968 | 0.005 | 0.640 |
| Qwen 3.5 9B | SciFact | HyDE | retrieval hurt | 300 | 128 | 0.516 | 0.881 | 0.001 | 0.365 |
| Qwen 3.5 9B | SciFact | SCOPE | deltaM<0 | 300 | 214 | 0.521 | 0.919 | 0.009 | 0.475 |
| Qwen 3.5 9B | SciFact | SCOPE | retrieval hurt | 300 | 52 | 0.532 | 0.773 | 0.001 | 0.150 |
| Qwen 3.5 9B | TREC-COVID | HyDE | deltaM<0 | 28 | 19 | 0.684 | 0.865 | 0.048 | 0.335 |
| Qwen 3.5 9B | TREC-COVID | HyDE | retrieval hurt | 28 | 9 | 0.569 | 0.708 | 0.020 | 0.109 |
| Qwen 3.5 9B | TREC-COVID | SCOPE | deltaM<0 | 25 | 13 | 0.590 | 0.769 | 0.008 | 0.196 |
| Qwen 3.5 9B | TREC-COVID | SCOPE | retrieval hurt | 28 | 3 | 0.647 | 0.787 | 0.047 | 0.094 |
| Gemma 4 26B | Pooled | HyDE | deltaM<0 | 2285 | 1256 | 0.520 | 0.944 | 0.000 | 0.571 |
| Gemma 4 26B | Pooled | HyDE | retrieval hurt | 2287 | 832 | 0.490 | 0.798 | 0.001 | 0.206 |
| Gemma 4 26B | Pooled | SCOPE | deltaM<0 | 2282 | 1206 | 0.509 | 0.909 | 0.000 | 0.450 |
| Gemma 4 26B | Pooled | SCOPE | retrieval hurt | 2287 | 429 | 0.598 | 0.743 | 0.016 | 0.119 |
| Mistral Small 3.2 24B | Pooled | HyDE | deltaM<0 | 2282 | 1239 | 0.499 | 0.944 | 0.001 | 0.565 |
| Mistral Small 3.2 24B | Pooled | HyDE | retrieval hurt | 2287 | 764 | 0.501 | 0.807 | 0.001 | 0.221 |
| Mistral Small 3.2 24B | Pooled | SCOPE | deltaM<0 | 2279 | 1193 | 0.508 | 0.910 | 0.000 | 0.456 |
| Mistral Small 3.2 24B | Pooled | SCOPE | retrieval hurt | 2287 | 420 | 0.590 | 0.708 | 0.012 | 0.078 |
| Qwen 3.5 9B | Pooled | HyDE | deltaM<0 | 2286 | 1219 | 0.492 | 0.945 | 0.000 | 0.568 |
| Qwen 3.5 9B | Pooled | HyDE | retrieval hurt | 2287 | 805 | 0.533 | 0.792 | 0.004 | 0.193 |
| Qwen 3.5 9B | Pooled | SCOPE | deltaM<0 | 2283 | 1168 | 0.528 | 0.922 | 0.000 | 0.489 |
| Qwen 3.5 9B | Pooled | SCOPE | retrieval hurt | 2287 | 417 | 0.591 | 0.714 | 0.014 | 0.088 |

## SCOPE-vs-HyDE Robustness Gap

Positive `SCOPE-HyDE net` means SCOPE loses less retrieval exposure than HyDE. Positive `closer-to-zero CE delta` means SCOPE's mean gold-affinity movement is closer to raw than HyDE's.

| Model | Dataset | HyDE net Hit@5 | SCOPE net Hit@5 | SCOPE-HyDE net | HyDE mean CE gold delta | SCOPE mean CE gold delta | Closer-to-zero CE delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B | FiQA | -34.0% | -31.0% | 2.9% | -4.055 | -2.947 | 1.108 |
| Gemma 4 26B | NFCorpus | -35.9% | -4.3% | 31.6% | -5.005 | -0.919 | 4.086 |
| Gemma 4 26B | SciDocs | -23.5% | -1.9% | 21.6% | -3.269 | 1.302 | 1.967 |
| Gemma 4 26B | SciFact | -47.0% | -16.3% | 30.7% | -7.360 | -0.909 | 6.451 |
| Gemma 4 26B | TREC-COVID | -28.0% | -2.0% | 26.0% | -7.662 | -1.824 | 5.838 |
| Mistral Small 3.2 24B | FiQA | -33.3% | -26.7% | 6.6% | -2.602 | -1.347 | 1.255 |
| Mistral Small 3.2 24B | NFCorpus | -32.5% | -2.8% | 29.7% | -3.562 | -0.323 | 3.240 |
| Mistral Small 3.2 24B | SciDocs | -23.6% | -5.5% | 18.1% | -2.341 | 1.466 | 0.875 |
| Mistral Small 3.2 24B | SciFact | -32.0% | -10.0% | 22.0% | -4.550 | -0.027 | 4.523 |
| Mistral Small 3.2 24B | TREC-COVID | -22.0% | -6.0% | 16.0% | -3.529 | -1.324 | 2.204 |
| Qwen 3.5 9B | FiQA | -37.3% | -28.2% | 9.1% | -4.219 | -2.172 | 2.047 |
| Qwen 3.5 9B | NFCorpus | -27.6% | -3.1% | 24.5% | -4.230 | -0.865 | 3.365 |
| Qwen 3.5 9B | SciDocs | -25.3% | -3.3% | 22.0% | -4.293 | 1.143 | 3.150 |
| Qwen 3.5 9B | SciFact | -39.3% | -12.0% | 27.3% | -7.139 | -0.603 | 6.536 |
| Qwen 3.5 9B | TREC-COVID | -34.0% | -6.0% | 28.0% | -7.695 | -2.074 | 5.620 |
| Gemma 4 26B | Pooled | -31.3% | -12.2% | 19.0% | -4.354 | -0.546 | 3.807 |
| Mistral Small 3.2 24B | Pooled | -28.6% | -11.6% | 17.0% | -2.895 | 0.179 | 2.716 |
| Qwen 3.5 9B | Pooled | -31.0% | -11.4% | 19.6% | -4.705 | -0.357 | 4.348 |

## Reading

- SCOPE is consistently less destructive than HyDE in pooled Hit@5: Gemma 4 26B gap 19.0%; Mistral Small 3.2 24B gap 17.0%; Qwen 3.5 9B gap 19.6%.
- SCOPE's row-level gold-affinity mechanism remains positive across included models: Gemma 4 26B rho 0.426; Mistral Small 3.2 24B rho 0.390; Qwen 3.5 9B rho 0.393.
- The operational implication is still gated expansion: the mechanism predicts which rows can move, but ungated expansion remains risky when raw retrieval already has strong gold exposure.

## Sources

- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-qwen3p5-9b_rag_hyde.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-qwen3p5-9b_snap_hyre.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-qwen3p5-9b_rag_hyde.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-qwen3p5-9b_snap_hyre.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-qwen3p5-9b_rag_hyde.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-qwen3p5-9b_snap_hyre.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-qwen3p5-9b_rag_hyde.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-qwen3p5-9b_snap_hyre.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-qwen3p5-9b_rag_hyde.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-qwen3p5-9b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-qwen3p5-9b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-qwen3p5-9b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-qwen3p5-9b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-qwen3p5-9b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-qwen3p5-9b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-qwen3p5-9b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-qwen3p5-9b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-qwen3p5-9b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-mistral-small-3p2-24b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-mistral-small-3p2-24b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-qwen3p5-9b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-qwen3p5-9b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_raw_question_k10.jsonl`

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CROSS_ENCODER_DEVICE=cuda \
uv run python scripts/analyze_beir_phase1b.py \
  --providers or-gemma4-26b or-qwen3p5-9b or-mistral-small-3p2-24b \
  --output docs/generated/beir_phase1b_model_breadth_2026-05-26.md
```
