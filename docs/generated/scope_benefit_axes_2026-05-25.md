# SCOPE Benefit Axes - 2026-05-25

## Scope

This results-lane analysis reuses `scripts/analyze_perplexity_axis.py` and the same signed raw/SCOPE retrieval caches and detail logs. It tests whether SCOPE benefit is better explained by question length/specificity or raw-retrieval difficulty than by unigram corpus perplexity. No files under `paper/` were edited.

Axes tested:

- `question_tokens`: Question tokens (higher = longer / more specific).
- `log_perplexity`: Log perplexity (higher = less corpus-like).
- `raw_hit_at5`: Raw Hit@5 (higher = raw already retrieved gold).
- `raw_gold_rank_at10`: Raw gold rank@10 (higher = harder; 11 means not in raw top-10).

## Dataset-Level Axis Values

| Dataset | Questions | Median question tokens | Median perplexity | Mean log perplexity | Raw Hit@5 | Raw gold in top-10 |
|---|---:|---:|---:|---:|---:|---:|
| BarExamQA | 1195 | 193 | 1898.4 | 7.564 | 1.4% | 2.2% |
| HousingQA state-filtered | 6853 | 22 | 1434.8 | 7.320 | 36.9% | 48.1% |

## Correlations

Outcome signs are `SCOPE - raw`: retrieval delta is Hit@5 movement, answer delta is exact-answer correctness movement. For `raw_hit_at5`, a negative correlation means SCOPE helps more when raw retrieval misses.

| Dataset | Model | Axis | N | Pearson retrieval | Spearman retrieval | Pearson answer | Spearman answer |
|---|---|---|---:|---:|---:|---:|---:|
| BarExamQA | Groq Llama 8B | Question tokens | 1195 | 0.051 | 0.055 | 0.014 | 0.014 |
| BarExamQA | Groq Llama 8B | Log perplexity | 1195 | 0.014 | 0.017 | -0.003 | -0.000 |
| BarExamQA | Groq Llama 8B | Raw Hit@5 | 1195 | -0.327 | -0.306 | -0.031 | -0.030 |
| BarExamQA | Groq Llama 8B | Raw gold rank@10 | 1195 | 0.291 | 0.227 | 0.022 | -0.004 |
| BarExamQA | Gemma 4 26B | Question tokens | 1195 | -0.005 | -0.007 | -0.033 | -0.035 |
| BarExamQA | Gemma 4 26B | Log perplexity | 1195 | -0.045 | -0.043 | 0.020 | 0.015 |
| BarExamQA | Gemma 4 26B | Raw Hit@5 | 1195 | -0.269 | -0.247 | -0.030 | -0.030 |
| BarExamQA | Gemma 4 26B | Raw gold rank@10 | 1195 | 0.224 | 0.167 | 0.022 | 0.016 |
| BarExamQA | Groq Llama 70B | Question tokens | 1195 | 0.027 | 0.022 | -0.000 | -0.013 |
| BarExamQA | Groq Llama 70B | Log perplexity | 1195 | -0.031 | -0.027 | 0.037 | 0.038 |
| BarExamQA | Groq Llama 70B | Raw Hit@5 | 1195 | -0.314 | -0.290 | 0.002 | 0.002 |
| BarExamQA | Groq Llama 70B | Raw gold rank@10 | 1195 | 0.280 | 0.217 | 0.003 | 0.005 |
| BarExamQA | Pooled | Question tokens | 3585 | 0.024 | 0.022 | -0.004 | -0.009 |
| BarExamQA | Pooled | Log perplexity | 3585 | -0.021 | -0.019 | 0.016 | 0.016 |
| BarExamQA | Pooled | Raw Hit@5 | 3585 | -0.302 | -0.280 | -0.020 | -0.020 |
| BarExamQA | Pooled | Raw gold rank@10 | 3585 | 0.264 | 0.203 | 0.016 | 0.004 |
| HousingQA state-filtered | Groq Llama 8B | Question tokens | 6853 | -0.006 | -0.017 | 0.012 | 0.017 |
| HousingQA state-filtered | Groq Llama 8B | Log perplexity | 6853 | -0.036 | -0.051 | -0.049 | -0.049 |
| HousingQA state-filtered | Groq Llama 8B | Raw Hit@5 | 6853 | -0.593 | -0.600 | -0.026 | -0.026 |
| HousingQA state-filtered | Groq Llama 8B | Raw gold rank@10 | 6853 | 0.523 | 0.493 | 0.031 | 0.030 |
| HousingQA state-filtered | Gemma 4 26B | Question tokens | 6853 | -0.008 | -0.017 | 0.000 | -0.001 |
| HousingQA state-filtered | Gemma 4 26B | Log perplexity | 6853 | -0.049 | -0.060 | -0.052 | -0.030 |
| HousingQA state-filtered | Gemma 4 26B | Raw Hit@5 | 6853 | -0.550 | -0.549 | -0.052 | -0.051 |
| HousingQA state-filtered | Gemma 4 26B | Raw gold rank@10 | 6853 | 0.470 | 0.427 | 0.043 | 0.038 |
| HousingQA state-filtered | Groq Llama 70B | Question tokens | 6853 | -0.043 | -0.033 | -0.007 | -0.003 |
| HousingQA state-filtered | Groq Llama 70B | Log perplexity | 6853 | -0.054 | -0.090 | -0.038 | -0.016 |
| HousingQA state-filtered | Groq Llama 70B | Raw Hit@5 | 6853 | -0.646 | -0.660 | -0.056 | -0.055 |
| HousingQA state-filtered | Groq Llama 70B | Raw gold rank@10 | 6853 | 0.573 | 0.550 | 0.059 | 0.059 |
| HousingQA state-filtered | Pooled | Question tokens | 20559 | -0.019 | -0.022 | 0.002 | 0.005 |
| HousingQA state-filtered | Pooled | Log perplexity | 20559 | -0.046 | -0.066 | -0.046 | -0.033 |
| HousingQA state-filtered | Pooled | Raw Hit@5 | 20559 | -0.592 | -0.598 | -0.043 | -0.043 |
| HousingQA state-filtered | Pooled | Raw gold rank@10 | 20559 | 0.518 | 0.486 | 0.043 | 0.041 |

## Strongest Axis

| Dataset | Strongest pooled axis | Spearman retrieval | Pearson retrieval | Spearman answer | Binned curve axis | Mean retrieval delta | Mean answer delta |
|---|---|---:|---:|---:|---|---:|---:|
| BarExamQA | Raw Hit@5 | -0.280 | -0.302 | -0.020 | Raw gold rank@10 | 9.5% | 3.9% |
| HousingQA state-filtered | Raw Hit@5 | -0.598 | -0.592 | -0.043 | Raw gold rank@10 | -6.7% | -2.3% |

## BarExamQA Binned Curve

The strongest axis is `raw_hit_at5`, but it is binary; these quintiles use `raw_gold_rank_at10`, the rank-form of the same raw-difficulty signal. Direction: higher = harder; 11 means not in raw top-10.

| Bin | N | Axis median | Axis range | SCOPE retrieval win | Raw retrieval win | Net retrieval delta | SCOPE answer win | Raw answer win | Net answer delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 717 | 11.00 | 1.00-11.00 | 12.7% | 5.2% | 7.5% | 12.8% | 7.4% | 5.4% |
| 2 | 717 | 11.00 | 11.00-11.00 | 8.9% | 0.0% | 8.9% | 17.0% | 10.2% | 6.8% |
| 3 | 717 | 11.00 | 11.00-11.00 | 10.3% | 0.0% | 10.3% | 12.4% | 8.2% | 4.2% |
| 4 | 717 | 11.00 | 11.00-11.00 | 8.8% | 0.0% | 8.8% | 11.4% | 9.3% | 2.1% |
| 5 | 717 | 11.00 | 11.00-11.00 | 11.7% | 0.0% | 11.7% | 10.3% | 9.5% | 0.8% |

## HousingQA state-filtered Binned Curve

The strongest axis is `raw_hit_at5`, but it is binary; these quintiles use `raw_gold_rank_at10`, the rank-form of the same raw-difficulty signal. Direction: higher = harder; 11 means not in raw top-10.

| Bin | N | Axis median | Axis range | SCOPE retrieval win | Raw retrieval win | Net retrieval delta | SCOPE answer win | Raw answer win | Net answer delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4112 | 1.00 | 1.00-2.00 | 0.0% | 43.5% | -43.5% | 6.8% | 13.0% | -6.2% |
| 2 | 4112 | 4.00 | 2.00-7.00 | 5.5% | 44.5% | -39.0% | 9.5% | 12.7% | -3.2% |
| 3 | 4111 | 11.00 | 7.00-11.00 | 24.2% | 0.0% | 24.2% | 12.2% | 12.6% | -0.3% |
| 4 | 4112 | 11.00 | 11.00-11.00 | 10.5% | 0.0% | 10.5% | 11.4% | 13.1% | -1.7% |
| 5 | 4112 | 11.00 | 11.00-11.00 | 14.4% | 0.0% | 14.4% | 12.6% | 12.7% | -0.0% |

## Reading

- Question length strongly separates the datasets at the median level (193 BarExam tokens vs 22 Housing tokens), but within-dataset length has weak pooled Spearman correlation with retrieval delta: 0.022 on BarExamQA and -0.022 on HousingQA.
- Raw-retrieval difficulty is the strongest axis in both datasets. `raw_hit_at5` has pooled Spearman correlations of -0.280 on BarExamQA and -0.598 on HousingQA. This is partly mechanical because a positive SCOPE-minus-raw retrieval delta requires raw to miss, but it is still the clearest practical gating signal.
- Log-perplexity remains weaker: pooled Spearman retrieval correlations are -0.019 on BarExamQA and -0.066 on HousingQA. The previous null result holds.
- Answer-delta correlations are consistently smaller than retrieval-delta correlations. The best explanatory axis for downstream answer movement is therefore still indirect: first identify raw retrieval failures, then test whether SCOPE repairs them without introducing answer-context dilution.

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

Unigram LM cache directory reused for log-perplexity: `/tmp/perplexity_axis_lm_cache_2026-05-25`

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_scope_benefit_axes.py \
  --output docs/generated/scope_benefit_axes_2026-05-25.md
```
