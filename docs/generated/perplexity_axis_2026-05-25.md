# Perplexity Axis Probe - 2026-05-25

## Scope

This results-lane analysis builds add-1 smoothed unigram language models over each retrieval corpus, scores every eval question by corpus perplexity, and joins that per-query score to signed raw-vs-SCOPE retrieval and answer outcomes. No files under `paper/` were edited.

- BarExamQA LM: `legal_passages` collection, one corpus-wide unigram model.
- HousingQA LM: `housing_statutes` collection, one unigram model per `state` metadata value to match state-filtered retrieval.
- Question text: `eval_harness._fmt_intermediate`, so BarExam includes shared prompts and answer-choice text without choice letters; Housing includes the state-framed question.
- Correlations use `log(perplexity)` because raw perplexity is heavy-tailed.

## Dataset Separation

| Dataset | Questions | LM scope | Median PPL | IQR PPL | Mean log PPL | Mean OOV rate | Median tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| BarExamQA | 1195 | corpus-wide | 1898.4 | 1403.4-2597.1 | 7.564 | 0.9% | 193 |
| HousingQA state-filtered | 6853 | per state | 1434.8 | 1027.5-2096.4 | 7.320 | 0.7% | 22 |

Separation check: probability that a random BarExamQA question has higher log-perplexity than a random HousingQA question is 0.652; Cohen's d on log-perplexity is 0.30. This indicates weak dataset separation on this axis.

## Correlations

Each model row uses one point per question. Pooled rows use one point per question-model pair. Retrieval delta is `SCOPE Hit@5 - raw Hit@5`; answer delta is `SCOPE correct - raw correct`.

| Dataset | Model | N | Pearson retrieval | Spearman retrieval | Pearson answer | Spearman answer | Mean retrieval delta | Mean answer delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | Groq Llama 8B | 1195 | 0.014 | 0.017 | -0.003 | -0.000 | 8.1% | 2.4% |
| BarExamQA | Gemma 4 26B | 1195 | -0.045 | -0.043 | 0.020 | 0.015 | 10.6% | 4.0% |
| BarExamQA | Groq Llama 70B | 1195 | -0.031 | -0.027 | 0.037 | 0.038 | 9.6% | 5.2% |
| BarExamQA | Pooled | 3585 | -0.021 | -0.019 | 0.016 | 0.016 | 9.5% | 3.9% |
| HousingQA state-filtered | Groq Llama 8B | 6853 | -0.036 | -0.051 | -0.049 | -0.049 | -7.4% | -3.3% |
| HousingQA state-filtered | Gemma 4 26B | 6853 | -0.049 | -0.060 | -0.052 | -0.030 | 1.1% | -1.1% |
| HousingQA state-filtered | Groq Llama 70B | 6853 | -0.054 | -0.090 | -0.038 | -0.016 | -13.8% | -2.5% |
| HousingQA state-filtered | Pooled | 20559 | -0.046 | -0.066 | -0.046 | -0.033 | -6.7% | -2.3% |

## BarExamQA Binned Curve

Bins are within-dataset quintiles of question perplexity, pooled across the three model rows.

| Bin | N | Median PPL | PPL range | SCOPE retrieval win | Raw retrieval win | Net retrieval delta | SCOPE answer win | Raw answer win | Net answer delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 717 | 1098.6 | 492.4-1315.1 | 11.7% | 1.5% | 10.2% | 11.6% | 8.4% | 3.2% |
| 2 | 717 | 1493.3 | 1315.5-1699.8 | 11.9% | 2.1% | 9.8% | 11.9% | 9.6% | 2.2% |
| 3 | 717 | 1898.4 | 1700.8-2160.7 | 10.6% | 0.0% | 10.6% | 12.4% | 8.8% | 3.6% |
| 4 | 717 | 2433.3 | 2161.3-2818.0 | 9.2% | 0.4% | 8.8% | 14.4% | 8.4% | 6.0% |
| 5 | 717 | 3453.5 | 2820.4-9451.8 | 9.1% | 1.1% | 7.9% | 13.8% | 9.5% | 4.3% |

## HousingQA state-filtered Binned Curve

Bins are within-dataset quintiles of question perplexity, pooled across the three model rows.

| Bin | N | Median PPL | PPL range | SCOPE retrieval win | Raw retrieval win | Net retrieval delta | SCOPE answer win | Raw answer win | Net answer delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4112 | 499.0 | 335.6-924.6 | 12.7% | 12.6% | 0.1% | 13.5% | 12.8% | 0.8% |
| 2 | 4112 | 1099.3 | 924.6-1250.1 | 11.5% | 18.2% | -6.8% | 11.8% | 14.2% | -2.4% |
| 3 | 4111 | 1434.8 | 1250.1-1689.4 | 11.5% | 16.9% | -5.4% | 10.8% | 14.1% | -3.4% |
| 4 | 4112 | 1961.7 | 1689.4-2271.3 | 9.5% | 22.6% | -13.1% | 9.6% | 11.5% | -1.9% |
| 5 | 4112 | 2951.5 | 2271.3-464677.9 | 9.4% | 17.7% | -8.3% | 6.8% | 11.3% | -4.5% |

## q200 Union Probe Supplement

These rows use the local q200 `or-gemma4-26b` raw+SCOPE union probe scratch outputs when present. They are diagnostic only and are not full-N results.

| Dataset | Arm | N | Accuracy | Hit@5 | Pearson union-vs-raw answer | Spearman union-vs-raw answer | Pearson union-vs-SCOPE answer | Spearman union-vs-SCOPE answer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | Union + CE-rerank | 200 | 83.5% | 4.0% | -0.101 | -0.119 | -0.077 | -0.084 |
| BarExamQA | Union + RRF | 200 | 87.0% | 5.5% | -0.011 | -0.022 | 0.002 | 0.004 |
| BarExamQA | Union + LLM-judge | 200 | 88.5% | 11.5% | -0.039 | -0.054 | -0.042 | -0.046 |
| HousingQA state-filtered | Union + CE-rerank | 200 | 65.0% | 38.0% | 0.030 | 0.025 | 0.012 | -0.010 |
| HousingQA state-filtered | Union + RRF | 200 | 60.5% | 45.5% | 0.018 | 0.017 | 0.003 | -0.017 |
| HousingQA state-filtered | Union + LLM-judge | 200 | 63.0% | 58.0% | -0.009 | -0.020 | -0.024 | -0.051 |

## Reading

- BarExamQA has a higher median question-corpus perplexity than HousingQA and is the dataset where SCOPE improves retrieval over raw on average: 9.5% pooled Hit@5 delta. The binned curve is not monotone, so the strong version of the per-query monotonicity hypothesis is not supported.
- HousingQA state-filtered has lower perplexity under its state-specific statute LMs, and SCOPE is not retrieval-positive overall: -6.7% pooled Hit@5 delta. This supports the strong-query/state-anchor caveat.
- Per-query perplexity is a dataset/regime separator more than a strong within-dataset predictor. Pooled Spearman correlations are -0.019 retrieval / 0.016 answer for BarExamQA and -0.066 retrieval / -0.033 answer for HousingQA.
- The answer-delta correlations are small; retrieval exposure moves more cleanly than downstream exact accuracy. Treat perplexity as a routing feature candidate, not a standalone policy.

## Sources

- `caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama8b_20260518_211000_barexam_local-snap-hyre-groq-llama8b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama8b_20260518_231747_barexam_local-snap-hyre-groq-llama8b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama70b_20260515_194919_barexam_local-snap-hyre-groq-llama70b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`
- `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`
- `logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl`
- `logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl`

q200 union probe scratch inputs:

- `/tmp/raw_scope_union_downstream_2026-05-25b_rows.jsonl`
- `/tmp/raw_scope_union_downstream_2026-05-25b_housing_rows.jsonl`

Unigram LM cache directory used for this run: `/tmp/perplexity_axis_lm_cache_2026-05-25`

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_perplexity_axis.py \
  --output docs/generated/perplexity_axis_2026-05-25.md
```
