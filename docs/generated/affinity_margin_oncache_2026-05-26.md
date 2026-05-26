# Affinity-Margin Mechanism Test - 2026-05-26

## Scope

Read-only results-lane analysis over existing BarExamQA and HousingQA state-filtered caches. No answer/model calls were made and no files under `paper/` were edited.

Gold margin definition: `M(x) = aff(x, best gold) - max_d aff(x, d)`, where `d` ranges over that condition's own retrieved top-10 non-gold cache entries. HousingQA multi-gold rows use the max over the gold set and exclude all gold ids from the distractor max.

- CE gold affinities come from the prior query-gap CE scoring cache when present; CE distractor affinities use the stored retrieval-cache cross-encoder scores.
- Cosine margins use the configured gte query embedder and stored Chroma document embeddings.
- Retrieval gain is `SCOPE Hit@5 - raw Hit@5`; Collins-Thompson RI is `(n_help - n_hurt) / N`.

## Verdicts

| Prediction | Verdict | Key numbers |
|---|---|---|
| P1 | **killed** | BarExam low/high M_raw net 8.2%/7.7%; Housing low/high 5.5%/-35.5% |
| P2 | **killed** | CE pooled full-margin rho=0.419, tau=0.336; gold-only rho=0.436; gain=-0.017 |
| P3 | **killed** | within-dataset crossover=no; joint max margin partial-R2=0.130, max confound partial-R2=0.004 |
| P4 | **supported** | failure AUC quality(logPPL+OOV)=0.574; margin(M_raw+CE(scope,gold))=0.913 |

## Collins-Thompson Robustness Index

| Dataset | Model | N | Raw Hit@5 | SCOPE Hit@5 | Net delta | Help | Hurt | RI | Answer delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | Groq Llama 8B | 1195 | 1.4% | 9.5% | 8.1% | 110 | 13 | 0.081 | 2.4% |
| BarExamQA | Gemma 4 26B | 1195 | 1.4% | 12.1% | 10.6% | 138 | 11 | 0.106 | 4.0% |
| BarExamQA | Groq Llama 70B | 1195 | 1.4% | 11.0% | 9.6% | 128 | 13 | 0.096 | 5.2% |
| BarExamQA | Pooled | 3585 | 1.4% | 10.9% | 9.5% | 376 | 37 | 0.095 | 3.9% |
| HousingQA state-filtered | Groq Llama 8B | 6853 | 36.9% | 29.6% | -7.4% | 711 | 1217 | -0.074 | -3.3% |
| HousingQA state-filtered | Gemma 4 26B | 6853 | 36.9% | 38.1% | 1.1% | 1023 | 946 | 0.011 | -1.1% |
| HousingQA state-filtered | Groq Llama 70B | 6853 | 36.9% | 23.1% | -13.8% | 507 | 1455 | -0.138 | -2.5% |
| HousingQA state-filtered | Pooled | 20559 | 36.9% | 30.2% | -6.7% | 2241 | 3618 | -0.067 | -2.3% |

## P2: Delta-Margin Correlation

Full margin is `deltaM = M_scope - M_raw`. Gold-only delta is `aff(scope,gold) - aff(raw,gold)`. P2 requires the full margin to correlate with retrieval gain and improve over gold-only affinity.

| Dataset | Model | Affinity | N | Full rho | Full tau | Gold-only rho | Gold-only tau | Rho gain | Tau gain |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | Groq Llama 8B | CE | 1195 | 0.369 | 0.299 | 0.343 | 0.279 | 0.026 | 0.021 |
| BarExamQA | Groq Llama 8B | Cosine | 1195 | 0.347 | 0.282 | 0.340 | 0.276 | 0.008 | 0.006 |
| BarExamQA | Gemma 4 26B | CE | 1195 | 0.412 | 0.334 | 0.354 | 0.287 | 0.058 | 0.047 |
| BarExamQA | Gemma 4 26B | Cosine | 1195 | 0.333 | 0.271 | 0.318 | 0.259 | 0.016 | 0.012 |
| BarExamQA | Groq Llama 70B | CE | 1195 | 0.381 | 0.309 | 0.327 | 0.265 | 0.054 | 0.044 |
| BarExamQA | Groq Llama 70B | Cosine | 1195 | 0.344 | 0.279 | 0.327 | 0.266 | 0.017 | 0.014 |
| BarExamQA | Pooled | CE | 3585 | 0.387 | 0.314 | 0.340 | 0.276 | 0.047 | 0.038 |
| BarExamQA | Pooled | Cosine | 3585 | 0.342 | 0.278 | 0.330 | 0.268 | 0.012 | 0.010 |
| HousingQA state-filtered | Groq Llama 8B | CE | 6853 | 0.466 | 0.376 | 0.444 | 0.358 | 0.023 | 0.019 |
| HousingQA state-filtered | Groq Llama 8B | Cosine | 6853 | 0.388 | 0.312 | 0.375 | 0.302 | 0.013 | 0.010 |
| HousingQA state-filtered | Gemma 4 26B | CE | 6853 | 0.501 | 0.406 | 0.504 | 0.410 | -0.004 | -0.005 |
| HousingQA state-filtered | Gemma 4 26B | Cosine | 6853 | 0.385 | 0.309 | 0.395 | 0.317 | -0.010 | -0.008 |
| HousingQA state-filtered | Groq Llama 70B | CE | 6853 | 0.467 | 0.377 | 0.440 | 0.354 | 0.028 | 0.023 |
| HousingQA state-filtered | Groq Llama 70B | Cosine | 6853 | 0.298 | 0.239 | 0.308 | 0.248 | -0.010 | -0.009 |
| HousingQA state-filtered | Pooled | CE | 20559 | 0.472 | 0.381 | 0.453 | 0.366 | 0.019 | 0.015 |
| HousingQA state-filtered | Pooled | Cosine | 20559 | 0.363 | 0.291 | 0.366 | 0.295 | -0.003 | -0.003 |
| Pooled | Pooled | CE | 24144 | 0.419 | 0.336 | 0.436 | 0.352 | -0.017 | -0.016 |
| Pooled | Pooled | Cosine | 24144 | 0.353 | 0.282 | 0.368 | 0.295 | -0.015 | -0.013 |

## P1: Raw-Margin Quintiles

Bins sort by CE `M_raw` within each dataset, pooled across model rows. A crossover means SCOPE helps more in low raw-margin bins and stops helping or hurts as raw margin rises.

### BarExamQA

| Bin | N | CE M_raw median | CE M_raw range | Raw Hit@5 | SCOPE Hit@5 | Net delta | Help | Hurt | RI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 717 | -6.180 | [-14.830, -4.750] | 0.0% | 8.2% | 8.2% | 59 | 0 | 0.082 |
| 2 | 717 | -3.951 | [-4.739, -3.319] | 0.4% | 12.3% | 11.9% | 88 | 3 | 0.119 |
| 3 | 717 | -2.867 | [-3.317, -2.455] | 0.4% | 9.8% | 9.3% | 69 | 2 | 0.093 |
| 4 | 717 | -2.052 | [-2.455, -1.502] | 0.8% | 11.0% | 10.2% | 77 | 4 | 0.102 |
| 5 | 717 | -0.915 | [-1.498, 4.340] | 5.4% | 13.1% | 7.7% | 83 | 28 | 0.077 |

### HousingQA state-filtered

| Bin | N | CE M_raw median | CE M_raw range | Raw Hit@5 | SCOPE Hit@5 | Net delta | Help | Hurt | RI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4112 | -9.173 | [-18.029, -6.775] | 0.3% | 5.8% | 5.5% | 237 | 10 | 0.055 |
| 2 | 4112 | -5.289 | [-6.775, -3.997] | 6.3% | 19.0% | 12.8% | 665 | 140 | 0.128 |
| 3 | 4111 | -2.968 | [-3.997, -2.118] | 24.6% | 29.9% | 5.3% | 798 | 579 | 0.053 |
| 4 | 4112 | -1.215 | [-2.118, -0.281] | 62.8% | 41.2% | -21.6% | 410 | 1299 | -0.216 |
| 5 | 4112 | 0.931 | [-0.281, 8.178] | 90.7% | 55.3% | -35.5% | 131 | 1590 | -0.355 |

### Pooled

| Bin | N | CE M_raw median | CE M_raw range | Raw Hit@5 | SCOPE Hit@5 | Net delta | Help | Hurt | RI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4829 | -8.772 | [-18.029, -6.453] | 0.4% | 6.3% | 5.9% | 300 | 17 | 0.059 |
| 2 | 4829 | -4.965 | [-6.453, -3.809] | 6.0% | 19.1% | 13.1% | 784 | 152 | 0.131 |
| 3 | 4828 | -2.938 | [-3.809, -2.206] | 18.9% | 24.8% | 5.8% | 816 | 534 | 0.058 |
| 4 | 4829 | -1.387 | [-2.206, -0.547] | 47.0% | 33.8% | -13.2% | 520 | 1157 | -0.132 |
| 5 | 4829 | 0.696 | [-0.547, 8.178] | 86.0% | 52.9% | -33.1% | 197 | 1795 | -0.331 |

## P3: Confound Checks

The primary check is whether the help-to-hurt crossover appears within BarExamQA and within HousingQA, not only between datasets. The secondary check is a standardized OLS regression on retrieval gain with partial-R2 deltas from dropping each feature.

| Regression | N | R2 | Standardized coefficients and partial-R2 |
|---|---:|---:|---|
| Joint + dataset id | 24144 | 0.205 | `ce_margin_raw` beta=-0.027, partial-R2=0.001; `ce_delta_margin` beta=0.444, partial-R2=0.130; `log_perplexity` beta=-0.010, partial-R2=0.000; `question_tokens` beta=0.099, partial-R2=0.002; `oov_rate` beta=0.054, partial-R2=0.002; `dataset_id` beta=-0.146, partial-R2=0.004 |
| BarExamQA only | 3585 | 0.164 | `ce_margin_raw` beta=0.187, partial-R2=0.025; `ce_delta_margin` beta=0.446, partial-R2=0.162; `log_perplexity` beta=-0.009, partial-R2=0.000; `question_tokens` beta=0.031, partial-R2=0.001; `oov_rate` beta=0.036, partial-R2=0.001 |
| HousingQA only | 20559 | 0.216 | `ce_margin_raw` beta=-0.007, partial-R2=0.000; `ce_delta_margin` beta=0.463, partial-R2=0.144; `log_perplexity` beta=-0.004, partial-R2=0.000; `question_tokens` beta=0.017, partial-R2=0.000; `oov_rate` beta=0.052, partial-R2=0.002 |

## P4: Failure Model

Failure is `1[CE deltaM < 0]`. The hallucination/surprise explanation would be plausible if OOV/log-perplexity explained these failures about as well as the margin features.

| Model | N | Failures | AUC | Log loss | Pseudo-R2 | Coefficients |
|---|---:|---:|---:|---:|---:|---|
| OOV + log-perplexity | 24144 | 12285 | 0.574 | 0.681 | 0.018 | `oov_rate`=-0.025; `log_perplexity`=0.356 |
| CE M_raw + CE(scope,gold) | 24144 | 12285 | 0.913 | 0.373 | 0.462 | `ce_margin_raw`=2.989; `ce_scope_gold`=-2.505 |
| Combined | 24144 | 12285 | 0.914 | 0.371 | 0.464 | `oov_rate`=-0.159; `log_perplexity`=0.157; `ce_margin_raw`=2.988; `ce_scope_gold`=-2.493 |

## Risk-Reward Reading

- Overall RI is -0.043: 2617 SCOPE-only retrieval hits versus 3655 raw-only hits over 24144 question-model rows.
- BarExamQA is favorable (9.5% net Hit@5, RI 0.095); HousingQA state-filtered is unfavorable (-6.7%, RI -0.067).
- The margin mechanism is useful only if the distractor term adds signal beyond gold affinity. In this run CE full-margin rho is 0.419 versus gold-only rho 0.436.
- Practical implication: a no-gold router should estimate raw-retrieval confidence and expansion risk before applying SCOPE broadly; raw state/jurisdiction anchors remain valuable on HousingQA.

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
- `caches/hyre/full/barexam_qfull_seed42_groq-llama8b_snap_hyre.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_groq-llama70b_snap_hyre.jsonl`
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
- `caches/hyre/full/housing_qfull_seed42_groq-llama8b_snap_hyre.jsonl`
- `caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/hyre/full/housing_qfull_seed42_groq-llama70b_snap_hyre.jsonl`
- CE gold score cache: `/tmp/scope_gap_mechanism_2026-05-25_points.jsonl`
- Perplexity LM cache: `/tmp/perplexity_axis_lm_cache_2026-05-25`

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_affinity_margin_oncache.py \
  --output docs/generated/affinity_margin_oncache_2026-05-26.md
```
