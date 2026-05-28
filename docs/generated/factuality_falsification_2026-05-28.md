# Factuality Falsification - 2026-05-28

## Scope

Phase: `q200`. This is a results-lane analysis over existing generation and retrieval caches plus LLM-as-judge factuality records. No files under `paper/` were edited.

Definitions:
- `retrieval hurt` is `1[generated-passage Hit@5 < raw-question Hit@5]`; this is the headline target because it is less circular than the margin target.
- `deltaM < 0` is the continuity target from the affinity-margin analysis.
- Primary factuality is gold-grounded corpus-supportedness; the no-gold proxy uses the top-3 raw-retrieved passages.
- Multi-gold rows with at most the configured cap are judged against the full gold passage set in one prompt. High-cardinality BEIR qrel rows use the CE-best gold passage proxy and are flagged by `gold_strategy` in the judge JSONL.

## Verdict

Headline verdict: **mixed**. On pooled `retrieval hurt`, AUC is 0.514 for OOV/logPPL, 0.581 for gold factuality, 0.529 for raw-top3 factuality, 0.791 for geometry, and 0.792 for gold factuality plus geometry.

High-gold-factuality rows still have retrieval hurt rate 8.5% over N=59; within that stratum, Spearman rho(deltaM, retrieval delta) is 0.389.

## AUC Table

| Dataset | Target | Feature set | N | Failures | AUC | Pseudo-R2 |
|---|---|---|---:|---:|---:|---:|
| BarExamQA | retrieval hurt | OOV + logPPL | 400 | 5 | 0.662 | 0.045 |
| BarExamQA | retrieval hurt | Factuality gold | 400 | 5 | 0.657 | 0.053 |
| BarExamQA | retrieval hurt | Factuality raw-top3 | 400 | 5 | 0.757 | 0.140 |
| BarExamQA | retrieval hurt | Geometry | 400 | 5 | 0.894 | 0.173 |
| BarExamQA | retrieval hurt | Gold factuality + geometry | 400 | 5 | 0.977 | 0.349 |
| BarExamQA | retrieval hurt | Raw-top3 factuality + geometry | 400 | 5 | 0.941 | 0.241 |
| BarExamQA | deltaM < 0 | OOV + logPPL | 400 | 285 | 0.541 | 0.002 |
| BarExamQA | deltaM < 0 | Factuality gold | 400 | 285 | 0.728 | 0.166 |
| BarExamQA | deltaM < 0 | Factuality raw-top3 | 400 | 285 | 0.508 | 0.001 |
| BarExamQA | deltaM < 0 | Geometry | 400 | 285 | 0.939 | 0.544 |
| BarExamQA | deltaM < 0 | Gold factuality + geometry | 400 | 285 | 0.942 | 0.556 |
| BarExamQA | deltaM < 0 | Raw-top3 factuality + geometry | 400 | 285 | 0.939 | 0.545 |
| FiQA | retrieval hurt | OOV + logPPL | 400 | 155 | 0.583 | 0.009 |
| FiQA | retrieval hurt | Factuality gold | 400 | 155 | 0.586 | 0.029 |
| FiQA | retrieval hurt | Factuality raw-top3 | 400 | 155 | 0.565 | 0.015 |
| FiQA | retrieval hurt | Geometry | 400 | 155 | 0.803 | 0.225 |
| FiQA | retrieval hurt | Gold factuality + geometry | 400 | 155 | 0.804 | 0.227 |
| FiQA | retrieval hurt | Raw-top3 factuality + geometry | 400 | 155 | 0.803 | 0.227 |
| FiQA | deltaM < 0 | OOV + logPPL | 400 | 276 | 0.540 | 0.013 |
| FiQA | deltaM < 0 | Factuality gold | 400 | 276 | 0.497 | 0.000 |
| FiQA | deltaM < 0 | Factuality raw-top3 | 400 | 276 | 0.541 | 0.006 |
| FiQA | deltaM < 0 | Geometry | 400 | 276 | 0.937 | 0.520 |
| FiQA | deltaM < 0 | Gold factuality + geometry | 400 | 276 | 0.938 | 0.521 |
| FiQA | deltaM < 0 | Raw-top3 factuality + geometry | 400 | 276 | 0.940 | 0.530 |
| NFCorpus | retrieval hurt | OOV + logPPL | 400 | 91 | 0.583 | 0.010 |
| NFCorpus | retrieval hurt | Factuality gold | 400 | 91 | 0.571 | 0.045 |
| NFCorpus | retrieval hurt | Factuality raw-top3 | 400 | 91 | 0.553 | 0.022 |
| NFCorpus | retrieval hurt | Geometry | 384 | 91 | 0.777 | 0.161 |
| NFCorpus | retrieval hurt | Gold factuality + geometry | 384 | 91 | 0.779 | 0.166 |
| NFCorpus | retrieval hurt | Raw-top3 factuality + geometry | 384 | 91 | 0.777 | 0.161 |
| NFCorpus | deltaM < 0 | OOV + logPPL | 380 | 238 | 0.562 | 0.008 |
| NFCorpus | deltaM < 0 | Factuality gold | 380 | 238 | 0.525 | 0.006 |
| NFCorpus | deltaM < 0 | Factuality raw-top3 | 380 | 238 | 0.527 | 0.004 |
| NFCorpus | deltaM < 0 | Geometry | 380 | 238 | 0.910 | 0.453 |
| NFCorpus | deltaM < 0 | Gold factuality + geometry | 380 | 238 | 0.911 | 0.456 |
| NFCorpus | deltaM < 0 | Raw-top3 factuality + geometry | 380 | 238 | 0.916 | 0.473 |
| SciDocs | retrieval hurt | OOV + logPPL | 400 | 67 | 0.513 | 0.000 |
| SciDocs | retrieval hurt | Factuality gold | 400 | 67 | 0.536 | 0.024 |
| SciDocs | retrieval hurt | Factuality raw-top3 | 400 | 67 | 0.581 | 0.046 |
| SciDocs | retrieval hurt | Geometry | 400 | 67 | 0.824 | 0.243 |
| SciDocs | retrieval hurt | Gold factuality + geometry | 400 | 67 | 0.832 | 0.254 |
| SciDocs | retrieval hurt | Raw-top3 factuality + geometry | 400 | 67 | 0.832 | 0.253 |
| SciDocs | deltaM < 0 | OOV + logPPL | 400 | 144 | 0.603 | 0.039 |
| SciDocs | deltaM < 0 | Factuality gold | 400 | 144 | 0.502 | 0.000 |
| SciDocs | deltaM < 0 | Factuality raw-top3 | 400 | 144 | 0.508 | 0.000 |
| SciDocs | deltaM < 0 | Geometry | 400 | 144 | 0.907 | 0.443 |
| SciDocs | deltaM < 0 | Gold factuality + geometry | 400 | 144 | 0.906 | 0.443 |
| SciDocs | deltaM < 0 | Raw-top3 factuality + geometry | 400 | 144 | 0.913 | 0.453 |
| SciFact | retrieval hurt | OOV + logPPL | 400 | 146 | 0.559 | 0.009 |
| SciFact | retrieval hurt | Factuality gold | 400 | 146 | 0.661 | 0.106 |
| SciFact | retrieval hurt | Factuality raw-top3 | 400 | 146 | 0.611 | 0.049 |
| SciFact | retrieval hurt | Geometry | 400 | 146 | 0.850 | 0.303 |
| SciFact | retrieval hurt | Gold factuality + geometry | 400 | 146 | 0.851 | 0.305 |
| SciFact | retrieval hurt | Raw-top3 factuality + geometry | 400 | 146 | 0.851 | 0.306 |
| SciFact | deltaM < 0 | OOV + logPPL | 400 | 294 | 0.548 | 0.009 |
| SciFact | deltaM < 0 | Factuality gold | 400 | 294 | 0.517 | 0.001 |
| SciFact | deltaM < 0 | Factuality raw-top3 | 400 | 294 | 0.545 | 0.008 |
| SciFact | deltaM < 0 | Geometry | 400 | 294 | 0.916 | 0.443 |
| SciFact | deltaM < 0 | Gold factuality + geometry | 400 | 294 | 0.916 | 0.446 |
| SciFact | deltaM < 0 | Raw-top3 factuality + geometry | 400 | 294 | 0.920 | 0.455 |
| TREC-COVID | retrieval hurt | OOV + logPPL | 100 | 17 | 0.584 | 0.017 |
| TREC-COVID | retrieval hurt | Factuality gold | 100 | 17 | 0.669 | 0.123 |
| TREC-COVID | retrieval hurt | Factuality raw-top3 | 100 | 17 | 0.633 | 0.091 |
| TREC-COVID | retrieval hurt | Geometry | 56 | 9 | 0.797 | 0.158 |
| TREC-COVID | retrieval hurt | Gold factuality + geometry | 56 | 9 | 0.816 | 0.188 |
| TREC-COVID | retrieval hurt | Raw-top3 factuality + geometry | 56 | 9 | 0.797 | 0.179 |
| TREC-COVID | deltaM < 0 | OOV + logPPL | 54 | 30 | 0.587 | 0.015 |
| TREC-COVID | deltaM < 0 | Factuality gold | 54 | 30 | 0.496 | 0.000 |
| TREC-COVID | deltaM < 0 | Factuality raw-top3 | 54 | 30 | 0.538 | 0.007 |
| TREC-COVID | deltaM < 0 | Geometry | 54 | 30 | 0.785 | 0.202 |
| TREC-COVID | deltaM < 0 | Gold factuality + geometry | 54 | 30 | 0.785 | 0.203 |
| TREC-COVID | deltaM < 0 | Raw-top3 factuality + geometry | 54 | 30 | 0.790 | 0.208 |
| HousingQA state-filtered | retrieval hurt | OOV + logPPL | 400 | 97 | 0.629 | 0.047 |
| HousingQA state-filtered | retrieval hurt | Factuality gold | 400 | 97 | 0.577 | 0.014 |
| HousingQA state-filtered | retrieval hurt | Factuality raw-top3 | 400 | 97 | 0.501 | 0.000 |
| HousingQA state-filtered | retrieval hurt | Geometry | 400 | 97 | 0.766 | 0.157 |
| HousingQA state-filtered | retrieval hurt | Gold factuality + geometry | 400 | 97 | 0.772 | 0.160 |
| HousingQA state-filtered | retrieval hurt | Raw-top3 factuality + geometry | 400 | 97 | 0.766 | 0.157 |
| HousingQA state-filtered | deltaM < 0 | OOV + logPPL | 400 | 199 | 0.678 | 0.088 |
| HousingQA state-filtered | deltaM < 0 | Factuality gold | 400 | 199 | 0.514 | 0.001 |
| HousingQA state-filtered | deltaM < 0 | Factuality raw-top3 | 400 | 199 | 0.571 | 0.019 |
| HousingQA state-filtered | deltaM < 0 | Geometry | 400 | 199 | 0.928 | 0.511 |
| HousingQA state-filtered | deltaM < 0 | Gold factuality + geometry | 400 | 199 | 0.928 | 0.511 |
| HousingQA state-filtered | deltaM < 0 | Raw-top3 factuality + geometry | 400 | 199 | 0.930 | 0.516 |
| Pooled | retrieval hurt | OOV + logPPL | 2500 | 578 | 0.514 | 0.002 |
| Pooled | retrieval hurt | Factuality gold | 2500 | 578 | 0.581 | 0.026 |
| Pooled | retrieval hurt | Factuality raw-top3 | 2500 | 578 | 0.529 | 0.004 |
| Pooled | retrieval hurt | Geometry | 2440 | 570 | 0.791 | 0.193 |
| Pooled | retrieval hurt | Gold factuality + geometry | 2440 | 570 | 0.792 | 0.193 |
| Pooled | retrieval hurt | Raw-top3 factuality + geometry | 2440 | 570 | 0.791 | 0.193 |
| Pooled | deltaM < 0 | OOV + logPPL | 2434 | 1466 | 0.524 | 0.001 |
| Pooled | deltaM < 0 | Factuality gold | 2434 | 1466 | 0.517 | 0.001 |
| Pooled | deltaM < 0 | Factuality raw-top3 | 2434 | 1466 | 0.533 | 0.005 |
| Pooled | deltaM < 0 | Geometry | 2434 | 1466 | 0.870 | 0.359 |
| Pooled | deltaM < 0 | Gold factuality + geometry | 2434 | 1466 | 0.871 | 0.359 |
| Pooled | deltaM < 0 | Raw-top3 factuality + geometry | 2434 | 1466 | 0.871 | 0.360 |

## Stratified Analysis

| Dataset | Stratum | N | Mean factuality | Retrieval hurt | deltaM<0 | Mean retrieval delta | Mean deltaM | rho(deltaM, retrieval delta) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | high factuality | 8 | 1.000 | 0.0% | 0.0% | 25.0% | 1.624 | 0.378 |
| BarExamQA | low/mid factuality | 392 | 0.148 | 1.3% | 72.7% | 5.6% | -2.863 | 0.294 |
| FiQA | high factuality | 15 | 1.000 | 6.7% | 53.3% | 0.0% | -0.510 | 0.042 |
| FiQA | low/mid factuality | 385 | 0.136 | 40.0% | 69.6% | -35.8% | -2.382 | 0.493 |
| NFCorpus | high factuality | 4 | 1.000 | 0.0% | 100.0% | 0.0% | -5.224 | -- |
| NFCorpus | low/mid factuality | 396 | 0.056 | 23.0% | 62.3% | -19.4% | -1.457 | 0.336 |
| SciDocs | low/mid factuality | 400 | 0.030 | 16.8% | 36.0% | -8.5% | 1.460 | 0.387 |
| SciFact | high factuality | 10 | 1.000 | 0.0% | 70.0% | 10.0% | -2.073 | 0.522 |
| SciFact | low/mid factuality | 390 | 0.129 | 37.4% | 73.6% | -34.1% | -2.705 | 0.426 |
| TREC-COVID | high factuality | 2 | 1.000 | 0.0% | 100.0% | 0.0% | -0.053 | -- |
| TREC-COVID | low/mid factuality | 98 | 0.133 | 17.3% | 54.7% | -15.3% | -0.370 | 0.419 |
| HousingQA state-filtered | high factuality | 20 | 1.000 | 20.0% | 40.0% | 0.0% | 0.263 | 0.562 |
| HousingQA state-filtered | low/mid factuality | 380 | 0.183 | 24.5% | 50.3% | -11.8% | -0.193 | 0.543 |
| Pooled | high factuality | 59 | 1.000 | 8.5% | 47.4% | 5.1% | -0.454 | 0.389 |
| Pooled | low/mid factuality | 2441 | 0.114 | 23.5% | 60.5% | -17.2% | -1.326 | 0.411 |

## Joint Coefficients

Standardized logistic coefficients for `{factuality_gold_score, ce_margin_raw, ce_exp_gold}`. Partial-R2 is the drop in pseudo-R2 when that feature is removed.

| Dataset | Target | N | Failures | AUC | Pseudo-R2 | Coefficients |
|---|---|---:|---:|---:|---:|---|
| BarExamQA | retrieval hurt | 400 | 5 | 0.977 | 0.349 | `factuality_gold_score` beta=-1.253, partial-R2=0.176; `ce_margin_raw` beta=0.817, partial-R2=0.068; `ce_exp_gold` beta=1.538, partial-R2=0.219 |
| BarExamQA | deltaM < 0 | 400 | 285 | 0.942 | 0.556 | `factuality_gold_score` beta=-0.457, partial-R2=0.012; `ce_margin_raw` beta=1.563, partial-R2=0.177; `ce_exp_gold` beta=-2.626, partial-R2=0.262 |
| FiQA | retrieval hurt | 400 | 155 | 0.804 | 0.227 | `factuality_gold_score` beta=-0.132, partial-R2=0.001; `ce_margin_raw` beta=0.868, partial-R2=0.093; `ce_exp_gold` beta=-1.181, partial-R2=0.146 |
| FiQA | deltaM < 0 | 400 | 276 | 0.938 | 0.521 | `factuality_gold_score` beta=-0.148, partial-R2=0.002; `ce_margin_raw` beta=3.113, partial-R2=0.521; `ce_exp_gold` beta=-0.793, partial-R2=0.044 |
| NFCorpus | retrieval hurt | 384 | 91 | 0.779 | 0.166 | `factuality_gold_score` beta=-0.427, partial-R2=0.006; `ce_margin_raw` beta=0.656, partial-R2=0.047; `ce_exp_gold` beta=-1.161, partial-R2=0.111 |
| NFCorpus | deltaM < 0 | 380 | 238 | 0.911 | 0.456 | `factuality_gold_score` beta=0.197, partial-R2=0.003; `ce_margin_raw` beta=2.967, partial-R2=0.449; `ce_exp_gold` beta=-1.026, partial-R2=0.083 |
| SciDocs | retrieval hurt | 400 | 67 | 0.832 | 0.254 | `factuality_gold_score` beta=-0.656, partial-R2=0.011; `ce_margin_raw` beta=1.190, partial-R2=0.155; `ce_exp_gold` beta=-1.078, partial-R2=0.136 |
| SciDocs | deltaM < 0 | 400 | 144 | 0.906 | 0.443 | `factuality_gold_score` beta=-0.023, partial-R2=0.000; `ce_margin_raw` beta=2.882, partial-R2=0.439; `ce_exp_gold` beta=-0.940, partial-R2=0.080 |
| SciFact | retrieval hurt | 400 | 146 | 0.851 | 0.305 | `factuality_gold_score` beta=-0.247, partial-R2=0.002; `ce_margin_raw` beta=0.683, partial-R2=0.056; `ce_exp_gold` beta=-1.566, partial-R2=0.172 |
| SciFact | deltaM < 0 | 400 | 294 | 0.916 | 0.446 | `factuality_gold_score` beta=-0.214, partial-R2=0.003; `ce_margin_raw` beta=2.603, partial-R2=0.440; `ce_exp_gold` beta=-0.978, partial-R2=0.055 |
| TREC-COVID | retrieval hurt | 56 | 9 | 0.816 | 0.188 | `factuality_gold_score` beta=-0.674, partial-R2=0.030; `ce_margin_raw` beta=0.292, partial-R2=0.000; `ce_exp_gold` beta=-0.674, partial-R2=0.062 |
| TREC-COVID | deltaM < 0 | 54 | 30 | 0.785 | 0.203 | `factuality_gold_score` beta=0.112, partial-R2=0.002; `ce_margin_raw` beta=1.096, partial-R2=0.168; `ce_exp_gold` beta=-0.352, partial-R2=0.014 |
| HousingQA state-filtered | retrieval hurt | 400 | 97 | 0.772 | 0.160 | `factuality_gold_score` beta=-0.185, partial-R2=0.003; `ce_margin_raw` beta=0.800, partial-R2=0.085; `ce_exp_gold` beta=-0.891, partial-R2=0.091 |
| HousingQA state-filtered | deltaM < 0 | 400 | 199 | 0.928 | 0.511 | `factuality_gold_score` beta=0.074, partial-R2=0.001; `ce_margin_raw` beta=2.552, partial-R2=0.411; `ce_exp_gold` beta=-1.978, partial-R2=0.248 |
| Pooled | retrieval hurt | 2440 | 570 | 0.792 | 0.193 | `factuality_gold_score` beta=-0.104, partial-R2=0.001; `ce_margin_raw` beta=0.762, partial-R2=0.077; `ce_exp_gold` beta=-1.042, partial-R2=0.106 |
| Pooled | deltaM < 0 | 2434 | 1466 | 0.871 | 0.359 | `factuality_gold_score` beta=-0.087, partial-R2=0.001; `ce_margin_raw` beta=2.253, partial-R2=0.332; `ce_exp_gold` beta=-0.887, partial-R2=0.061 |

## Judge Score Distribution

| Premise | Verdict | All | barexam | beir_fiqa | beir_nfcorpus | beir_scidocs | beir_scifact | beir_trec_covid | housing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold | contradicted | 21 | 1 | 2 | 1 | 0 | 12 | 0 | 5 |
| gold | entailed | 59 | 8 | 15 | 4 | 0 | 10 | 2 | 20 |
| gold | not_entailed | 1865 | 275 | 278 | 351 | 376 | 277 | 72 | 236 |
| gold | partially | 555 | 116 | 105 | 44 | 24 | 101 | 26 | 139 |
| raw_top3 | contradicted | 23 | 0 | 2 | 0 | 0 | 18 | 0 | 3 |
| raw_top3 | entailed | 56 | 7 | 11 | 6 | 2 | 18 | 3 | 9 |
| raw_top3 | not_entailed | 1953 | 359 | 260 | 350 | 340 | 263 | 78 | 303 |
| raw_top3 | partially | 468 | 34 | 127 | 44 | 58 | 101 | 19 | 85 |

## Reading

- The real factuality signal does beat the old OOV/log-perplexity proxy on the headline retrieval-hurt target.
- Geometry remains the stronger failure predictor on the pooled headline target.
- The LLM judge is itself model-biased, but the same prompt and judge model are used across HyDE/SCOPE and premise arms.

## Sources

- Judge cache: `docs/generated/factuality_judge_q200_2026-05-28.jsonl`
- Feature points: `docs/generated/factuality_feature_points_q200_2026-05-28.jsonl`
- BEIR geometry points: `/tmp/beir_phase1_verification_2026-05-26_points.jsonl`
- Legal SCOPE geometry points: `/tmp/affinity_margin_oncache_2026-05-26_points.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`
- `caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`

## Reproduction

```bash
NO_SILENT_FALLBACK=1 OPENROUTER_PROVIDER_ONLY=Cloudflare EVAL_CONCURRENCY=8 uv run python scripts/build_factuality_judge_cache.py --limit 200 --resume --output docs/generated/factuality_judge_q200_2026-05-28.jsonl
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_factuality_falsification.py --phase q200 --judge-cache docs/generated/factuality_judge_q200_2026-05-28.jsonl --features-out docs/generated/factuality_feature_points_q200_2026-05-28.jsonl --output docs/generated/factuality_falsification_2026-05-28.md
```
