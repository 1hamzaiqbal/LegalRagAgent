# Credibility D - OOD No-Gold QPP Predictor

No `paper/` files were edited.

## Verdict

- Status: **useful negative**.
- Best model family by held-out-generator tau: `gb` with mean tau `0.090`.
- Held-out-generator mean Kendall tau: `0.090`; held-out-dataset mean tau: `0.052`.
- Datta-style reliability bar `|tau| >= 0.5`: not reached in the tested calibration budget.

## Coverage

| Dataset | Generator | Rows | Help rate | Hurt rate | Mean retrieval delta |
|---|---|---:|---:|---:|---:|
| FiQA | DeepSeek V3.2 | 648 | 6.2% | 24.1% | -17.9% |
| FiQA | Gemma 4 26B | 648 | 3.9% | 34.9% | -31.0% |
| FiQA | Mistral Small 3.2 24B | 648 | 4.2% | 30.9% | -26.7% |
| FiQA | Qwen 3.5 9B | 648 | 5.2% | 33.5% | -28.2% |
| NFCorpus | DeepSeek V3.2 | 323 | 6.2% | 8.7% | -2.5% |
| NFCorpus | Gemma 4 26B | 323 | 6.2% | 10.5% | -4.3% |
| NFCorpus | Mistral Small 3.2 24B | 323 | 7.1% | 9.9% | -2.8% |
| NFCorpus | Qwen 3.5 9B | 323 | 5.9% | 9.0% | -3.1% |
| SciDocs | DeepSeek V3.2 | 1000 | 9.8% | 11.2% | -1.4% |
| SciDocs | Gemma 4 26B | 1000 | 8.7% | 10.6% | -1.9% |
| SciDocs | Mistral Small 3.2 24B | 1000 | 7.9% | 13.4% | -5.5% |
| SciDocs | Qwen 3.5 9B | 1000 | 8.3% | 11.6% | -3.3% |
| SciFact | DeepSeek V3.2 | 300 | 5.3% | 10.3% | -5.0% |
| SciFact | Gemma 4 26B | 300 | 4.0% | 20.3% | -16.3% |
| SciFact | Mistral Small 3.2 24B | 300 | 6.7% | 16.7% | -10.0% |
| SciFact | Qwen 3.5 9B | 300 | 5.3% | 17.3% | -12.0% |
| TREC-COVID | DeepSeek V3.2 | 50 | 2.0% | 4.0% | -2.0% |
| TREC-COVID | Gemma 4 26B | 50 | 2.0% | 4.0% | -2.0% |
| TREC-COVID | Mistral Small 3.2 24B | 50 | 2.0% | 8.0% | -6.0% |
| TREC-COVID | Qwen 3.5 9B | 50 | 2.0% | 8.0% | -6.0% |

## OOD Splits

| Split | Held out | Model | Train N | Test N | Kendall tau | Spearman | AUC(help) | Help rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| in_sample | none | logistic | 9284 | 9284 | 0.050 | 0.063 | 0.683 | 6.7% |
| heldout_generator | DeepSeek V3.2 | logistic | 6963 | 2321 | 0.069 | 0.087 | 0.696 | 7.5% |
| heldout_generator | Gemma 4 26B | logistic | 6963 | 2321 | 0.048 | 0.061 | 0.697 | 6.2% |
| heldout_generator | Mistral Small 3.2 24B | logistic | 6963 | 2321 | 0.040 | 0.052 | 0.649 | 6.5% |
| heldout_generator | Qwen 3.5 9B | logistic | 6963 | 2321 | 0.040 | 0.050 | 0.676 | 6.6% |
| heldout_dataset_lodo | FiQA | logistic | 6692 | 2592 | 0.083 | 0.105 | 0.661 | 4.9% |
| heldout_dataset_lodo | NFCorpus | logistic | 7992 | 1292 | 0.078 | 0.099 | 0.688 | 6.3% |
| heldout_dataset_lodo | SciDocs | logistic | 5284 | 4000 | 0.037 | 0.047 | 0.628 | 8.7% |
| heldout_dataset_lodo | SciFact | logistic | 8084 | 1200 | -0.008 | -0.007 | 0.789 | 5.3% |
| heldout_dataset_lodo | TREC-COVID | logistic | 9084 | 200 | 0.065 | 0.077 | 0.918 | 2.0% |
| heldout_dataset_leave2_proxy | FiQA + NFCorpus | logistic | 5400 | 3884 | 0.131 | 0.165 | 0.635 | 5.4% |
| heldout_dataset_leave2_proxy | FiQA + SciDocs | logistic | 2692 | 6592 | 0.018 | 0.023 | 0.609 | 7.2% |
| heldout_dataset_leave2_proxy | FiQA + SciFact | logistic | 5492 | 3792 | 0.011 | 0.015 | 0.681 | 5.0% |
| heldout_dataset_leave2_proxy | FiQA + TREC-COVID | logistic | 6492 | 2792 | 0.102 | 0.128 | 0.651 | 4.7% |
| heldout_dataset_leave2_proxy | NFCorpus + SciDocs | logistic | 3992 | 5292 | 0.052 | 0.066 | 0.650 | 8.1% |
| heldout_dataset_leave2_proxy | NFCorpus + SciFact | logistic | 6792 | 2492 | 0.068 | 0.087 | 0.691 | 5.9% |
| heldout_dataset_leave2_proxy | NFCorpus + TREC-COVID | logistic | 7792 | 1492 | 0.087 | 0.109 | 0.719 | 5.8% |
| heldout_dataset_leave2_proxy | SciDocs + SciFact | logistic | 4084 | 5200 | 0.042 | 0.053 | 0.659 | 7.9% |
| heldout_dataset_leave2_proxy | SciDocs + TREC-COVID | logistic | 5084 | 4200 | 0.032 | 0.040 | 0.605 | 8.4% |
| heldout_dataset_leave2_proxy | SciFact + TREC-COVID | logistic | 7884 | 1400 | 0.026 | 0.035 | 0.722 | 4.9% |
| in_sample | none | gb | 9284 | 9284 | 0.094 | 0.119 | 0.778 | 6.7% |
| heldout_generator | DeepSeek V3.2 | gb | 6963 | 2321 | 0.109 | 0.138 | 0.777 | 7.5% |
| heldout_generator | Gemma 4 26B | gb | 6963 | 2321 | 0.091 | 0.115 | 0.774 | 6.2% |
| heldout_generator | Mistral Small 3.2 24B | gb | 6963 | 2321 | 0.076 | 0.097 | 0.719 | 6.5% |
| heldout_generator | Qwen 3.5 9B | gb | 6963 | 2321 | 0.083 | 0.105 | 0.743 | 6.6% |
| heldout_dataset_lodo | FiQA | gb | 6692 | 2592 | 0.068 | 0.085 | 0.617 | 4.9% |
| heldout_dataset_lodo | NFCorpus | gb | 7992 | 1292 | 0.099 | 0.124 | 0.703 | 6.3% |
| heldout_dataset_lodo | SciDocs | gb | 5284 | 4000 | 0.037 | 0.046 | 0.605 | 8.7% |
| heldout_dataset_lodo | SciFact | gb | 8084 | 1200 | -0.017 | -0.018 | 0.790 | 5.3% |
| heldout_dataset_lodo | TREC-COVID | gb | 9084 | 200 | 0.023 | 0.026 | 0.857 | 2.0% |
| heldout_dataset_leave2_proxy | FiQA + NFCorpus | gb | 5400 | 3884 | 0.118 | 0.148 | 0.665 | 5.4% |
| heldout_dataset_leave2_proxy | FiQA + SciDocs | gb | 2692 | 6592 | 0.037 | 0.047 | 0.593 | 7.2% |
| heldout_dataset_leave2_proxy | FiQA + SciFact | gb | 5492 | 3792 | 0.018 | 0.024 | 0.648 | 5.0% |
| heldout_dataset_leave2_proxy | FiQA + TREC-COVID | gb | 6492 | 2792 | 0.085 | 0.106 | 0.621 | 4.7% |
| heldout_dataset_leave2_proxy | NFCorpus + SciDocs | gb | 3992 | 5292 | 0.056 | 0.071 | 0.626 | 8.1% |
| heldout_dataset_leave2_proxy | NFCorpus + SciFact | gb | 6792 | 2492 | 0.071 | 0.090 | 0.683 | 5.9% |
| heldout_dataset_leave2_proxy | NFCorpus + TREC-COVID | gb | 7792 | 1492 | 0.097 | 0.121 | 0.721 | 5.8% |
| heldout_dataset_leave2_proxy | SciDocs + SciFact | gb | 4084 | 5200 | 0.034 | 0.043 | 0.634 | 7.9% |
| heldout_dataset_leave2_proxy | SciDocs + TREC-COVID | gb | 5084 | 4200 | 0.029 | 0.036 | 0.589 | 8.4% |
| heldout_dataset_leave2_proxy | SciFact + TREC-COVID | gb | 7884 | 1400 | 0.018 | 0.025 | 0.745 | 4.9% |

## Calibration Budget Curve

Budget curve uses `gb` and adds labeled examples from the held-out generator before evaluating on the remaining held-out rows.

| Held-out generator | Calibration labels | Mean tau | Max tau |
|---|---:|---:|---:|
| DeepSeek V3.2 | 0 | 0.109 | 0.109 |
| DeepSeek V3.2 | 25 | 0.107 | 0.110 |
| DeepSeek V3.2 | 50 | 0.109 | 0.113 |
| DeepSeek V3.2 | 100 | 0.105 | 0.107 |
| DeepSeek V3.2 | 200 | 0.107 | 0.114 |
| DeepSeek V3.2 | 500 | 0.110 | 0.125 |
| DeepSeek V3.2 | 1000 | 0.110 | 0.121 |
| Gemma 4 26B | 0 | 0.091 | 0.091 |
| Gemma 4 26B | 25 | 0.092 | 0.093 |
| Gemma 4 26B | 50 | 0.093 | 0.096 |
| Gemma 4 26B | 100 | 0.095 | 0.097 |
| Gemma 4 26B | 200 | 0.093 | 0.097 |
| Gemma 4 26B | 500 | 0.091 | 0.100 |
| Gemma 4 26B | 1000 | 0.093 | 0.117 |
| Mistral Small 3.2 24B | 0 | 0.076 | 0.076 |
| Mistral Small 3.2 24B | 25 | 0.075 | 0.075 |
| Mistral Small 3.2 24B | 50 | 0.077 | 0.079 |
| Mistral Small 3.2 24B | 100 | 0.076 | 0.080 |
| Mistral Small 3.2 24B | 200 | 0.076 | 0.087 |
| Mistral Small 3.2 24B | 500 | 0.079 | 0.098 |
| Mistral Small 3.2 24B | 1000 | 0.080 | 0.109 |
| Qwen 3.5 9B | 0 | 0.083 | 0.083 |
| Qwen 3.5 9B | 25 | 0.083 | 0.083 |
| Qwen 3.5 9B | 50 | 0.084 | 0.088 |
| Qwen 3.5 9B | 100 | 0.086 | 0.088 |
| Qwen 3.5 9B | 200 | 0.086 | 0.091 |
| Qwen 3.5 9B | 500 | 0.084 | 0.092 |
| Qwen 3.5 9B | 1000 | 0.082 | 0.103 |

## Notes

- The true four-generator OOD battery is available only for the five BEIR Phase 1b datasets. BarExamQA and HousingQA have richer answer/QPP rows but not the same four-generator breadth, so they are not mixed into the generator-OOD estimate.
- The requested `5 train / 2 held-out datasets` split is impossible within the five-dataset, four-generator BEIR slice. This report uses leave-one-dataset-out plus leave-two-datasets-out as the available proxy.
- Features are no-gold raw-retrieval predictors: NQC, WIG, SMV, CE score/spread/entropy, dense query-top1 cosine, dense top-5 coherence/centroid norm, log perplexity, question length, and OOV rate.
- Row-level points: `docs/generated/credibility_D_ood_predictor_2026-05-29_points.jsonl`

