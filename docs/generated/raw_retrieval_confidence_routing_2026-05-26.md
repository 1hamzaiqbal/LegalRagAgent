# Raw-Retrieval Confidence Routing (QPP) - 2026-05-26

## Scope

This results-lane analysis tests whether no-gold Query Performance Prediction (QPP) signals from the existing raw-question retrieval caches can decide when to apply SCOPE/Snap-HyRE. It uses BarExamQA and HousingQA state-filtered full-N caches across the three current model rows. No retrieval or answer calls were launched, and no files under `paper/` were edited.

Predictors:

- `NQC-CE top10`: standard NQC-style normalized score dispersion, transferred to MiniLM cross-encoder scores as `std(top10) / abs(mean(top10))`.
- `WIG-CE top5-vs-top10`: WIG-style top-set separation. The existing cache does not store corpus-wide CE background scores, so this is a cache-local top-5 minus top-10 background proxy.
- `SMV-CE top10`: score magnitude and variance fusion on the cached top-10 cross-encoder scores.
- Dense-native predictors: offline gte query-to-top-hit cosine plus top-5 document-embedding coherence from the already indexed Chroma embeddings.
- Prior axes retained for comparison: unigram log-perplexity and question token count.

Dense feature status: computed from existing Chroma embeddings plus offline local query embeddings.

## Outcome Baselines

| Dataset | Model | N | Raw Hit@5 | SCOPE Hit@5 | SCOPE-raw Hit@5 | Raw acc | SCOPE acc | SCOPE-raw acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | Groq Llama 8B | 1195 | 1.4% | 9.5% | 8.1% | 54.5% | 56.9% | 2.4% |
| BarExamQA | Gemma 4 26B | 1195 | 1.4% | 12.1% | 10.6% | 78.0% | 82.0% | 4.0% |
| BarExamQA | Groq Llama 70B | 1195 | 1.4% | 11.0% | 9.6% | 74.6% | 79.7% | 5.2% |
| BarExamQA | Pooled | 3585 | 1.4% | 10.9% | 9.5% | 69.0% | 72.9% | 3.9% |
| HousingQA state-filtered | Groq Llama 8B | 6853 | 36.9% | 29.6% | -7.4% | 62.3% | 59.0% | -3.3% |
| HousingQA state-filtered | Gemma 4 26B | 6853 | 36.9% | 38.1% | 1.1% | 66.1% | 65.1% | -1.1% |
| HousingQA state-filtered | Groq Llama 70B | 6853 | 36.9% | 23.1% | -13.8% | 62.1% | 59.6% | -2.5% |
| HousingQA state-filtered | Pooled | 20559 | 36.9% | 30.2% | -6.7% | 63.5% | 61.2% | -2.3% |
| All | Pooled | 24144 | 31.7% | 27.4% | -4.3% | 64.3% | 63.0% | -1.4% |

## Named QPP Reliability

The Datta-style reliability bar is Kendall tau `|tau| >= 0.5`. Negative signs mean higher raw-retrieval confidence predicts less SCOPE benefit; positive signs mean the transferred predictor is oriented the other way on this data. This table is the main transfer check: classic QPP predictors do not get assumed to work on dense/cross-encoder scores.

| Scope | Predictor | Kendall ret | Meets tau >= 0.5? | Spearman ret | Kendall ans | Spearman ans | Direction read |
|---|---|---:|---|---:|---:|---:|---|
| BarExamQA pooled | NQC-CE top10 | -0.005 | no | -0.006 | -0.011 | -0.014 | higher predictor -> less SCOPE benefit |
| BarExamQA pooled | WIG-CE top5-vs-top10 | -0.025 | no | -0.030 | -0.016 | -0.020 | higher predictor -> less SCOPE benefit |
| BarExamQA pooled | SMV-CE top10 | -0.017 | no | -0.021 | -0.013 | -0.016 | higher predictor -> less SCOPE benefit |
| BarExamQA pooled | Dense query-top1 cosine | -0.020 | no | -0.024 | -0.018 | -0.022 | higher predictor -> less SCOPE benefit |
| BarExamQA pooled | Dense top-5 coherence | -0.008 | no | -0.009 | -0.026 | -0.033 | higher predictor -> less SCOPE benefit |
| HousingQA pooled | NQC-CE top10 | -0.070 | no | -0.089 | -0.007 | -0.009 | higher predictor -> less SCOPE benefit |
| HousingQA pooled | WIG-CE top5-vs-top10 | -0.076 | no | -0.097 | 0.004 | 0.005 | higher predictor -> less SCOPE benefit |
| HousingQA pooled | SMV-CE top10 | -0.013 | no | -0.016 | 0.013 | 0.016 | higher predictor -> less SCOPE benefit |
| HousingQA pooled | Dense query-top1 cosine | -0.053 | no | -0.068 | 0.002 | 0.003 | higher predictor -> less SCOPE benefit |
| HousingQA pooled | Dense top-5 coherence | -0.043 | no | -0.054 | -0.016 | -0.021 | higher predictor -> less SCOPE benefit |
| All pooled | NQC-CE top10 | -0.082 | no | -0.104 | -0.015 | -0.019 | higher predictor -> less SCOPE benefit |
| All pooled | WIG-CE top5-vs-top10 | -0.109 | no | -0.139 | -0.016 | -0.020 | higher predictor -> less SCOPE benefit |
| All pooled | SMV-CE top10 | -0.053 | no | -0.067 | -0.006 | -0.008 | higher predictor -> less SCOPE benefit |
| All pooled | Dense query-top1 cosine | -0.009 | no | -0.012 | 0.015 | 0.019 | higher predictor -> less SCOPE benefit |
| All pooled | Dense top-5 coherence | -0.013 | no | -0.016 | -0.007 | -0.009 | higher predictor -> less SCOPE benefit |

## Dataset Separation

AUC is `P(Housing predictor > BarExam predictor)` using one raw-cache feature vector per question. Values near 0.5 mean weak dataset separation.

| Predictor | BarExam median | Housing median | Housing > BarExam AUC | Separation read |
|---|---:|---:|---:|---|
| NQC-CE top10 | 0.239 | 0.484 | 0.688 | weak |
| WIG-CE top5-vs-top10 | 0.390 | 1.061 | 0.915 | clear |
| SMV-CE top10 | 0.550 | 4.887 | 0.856 | clear |
| Top-1 CE score | -0.997 | -0.013 | 0.583 | weak |
| Mean top-5 CE | -1.734 | -1.662 | 0.492 | weak |
| CE spread top1-top5 | 1.077 | 2.643 | 0.839 | clear |
| Negative CE entropy top5 | -1.524 | -1.198 | 0.814 | clear |
| Dense query-top1 cosine | 0.738 | 0.684 | 0.169 | clear |
| Dense top-5 coherence | 0.739 | 0.709 | 0.290 | clear |
| Dense top-5 centroid norm | 0.889 | 0.876 | 0.290 | clear |
| Log perplexity | 7.549 | 7.269 | 0.348 | weak |
| Question tokens | 193.000 | 22.000 | 0.001 | clear |

## Context Against Gold-Needed Mechanism

The prior query-gold mechanism report (`docs/generated/scope_gap_mechanism_2026-05-25.md`) is not deployable because it uses gold passage text, but it gives a useful upper-bound comparison for whether a signal tracks SCOPE retrieval repair. In that report, CE delta `CE(scope,gold) - CE(raw,gold)` had Spearman retrieval correlations of 0.340 on BarExamQA, 0.453 on HousingQA, and 0.436 pooled; cosine delta had 0.330, 0.366, and 0.368. The best no-gold named QPP predictor here is materially weaker, so raw-cache QPP is a proxy for selective expansion, not the mechanism itself.

## Full Correlation Matrix

Outcomes are `SCOPE - raw`: retrieval delta is Hit@5 movement and answer delta is exact-answer correctness movement.

| Dataset | Model | Predictor | Family | N | Pearson ret | Spearman ret | Kendall ret | Pearson ans | Spearman ans | Kendall ans |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | Groq Llama 8B | NQC-CE top10 | NQC | 1195 | -0.003 | -0.024 | -0.020 | -0.025 | -0.014 | -0.012 |
| BarExamQA | Groq Llama 8B | WIG-CE top5-vs-top10 | WIG | 1195 | -0.060 | -0.044 | -0.036 | -0.021 | -0.037 | -0.029 |
| BarExamQA | Groq Llama 8B | SMV-CE top10 | SMV | 1195 | -0.067 | -0.020 | -0.016 | 0.003 | -0.044 | -0.034 |
| BarExamQA | Groq Llama 8B | Top-1 CE score | CE hand feature | 1195 | -0.043 | -0.046 | -0.037 | -0.001 | -0.007 | -0.006 |
| BarExamQA | Groq Llama 8B | Mean top-5 CE | CE hand feature | 1195 | -0.038 | -0.043 | -0.035 | 0.005 | 0.002 | 0.002 |
| BarExamQA | Groq Llama 8B | CE spread top1-top5 | CE hand feature | 1195 | -0.043 | -0.015 | -0.013 | -0.021 | -0.036 | -0.029 |
| BarExamQA | Groq Llama 8B | Negative CE entropy top5 | CE hand feature | 1195 | -0.041 | -0.006 | -0.005 | -0.020 | -0.037 | -0.029 |
| BarExamQA | Groq Llama 8B | Dense query-top1 cosine | Dense QPP | 1195 | -0.067 | -0.077 | -0.062 | -0.017 | -0.013 | -0.010 |
| BarExamQA | Groq Llama 8B | Dense top-5 coherence | Dense QPP | 1195 | -0.035 | -0.028 | -0.023 | -0.058 | -0.045 | -0.036 |
| BarExamQA | Groq Llama 8B | Dense top-5 centroid norm | Dense QPP | 1195 | -0.035 | -0.028 | -0.023 | -0.058 | -0.045 | -0.036 |
| BarExamQA | Groq Llama 8B | Log perplexity | Prior axis | 1195 | 0.014 | 0.017 | 0.014 | -0.003 | -0.000 | -0.000 |
| BarExamQA | Groq Llama 8B | Question tokens | Prior axis | 1195 | 0.051 | 0.055 | 0.045 | 0.014 | 0.014 | 0.011 |
| BarExamQA | Gemma 4 26B | NQC-CE top10 | NQC | 1195 | -0.006 | -0.020 | -0.016 | -0.029 | -0.019 | -0.015 |
| BarExamQA | Gemma 4 26B | WIG-CE top5-vs-top10 | WIG | 1195 | -0.012 | -0.036 | -0.029 | 0.018 | 0.011 | 0.009 |
| BarExamQA | Gemma 4 26B | SMV-CE top10 | SMV | 1195 | -0.010 | -0.022 | -0.018 | 0.079 | 0.043 | 0.034 |
| BarExamQA | Gemma 4 26B | Top-1 CE score | CE hand feature | 1195 | 0.015 | -0.002 | -0.002 | -0.036 | -0.027 | -0.022 |
| BarExamQA | Gemma 4 26B | Mean top-5 CE | CE hand feature | 1195 | 0.017 | 0.004 | 0.003 | -0.051 | -0.046 | -0.037 |
| BarExamQA | Gemma 4 26B | CE spread top1-top5 | CE hand feature | 1195 | -0.005 | -0.006 | -0.005 | 0.047 | 0.019 | 0.015 |
| BarExamQA | Gemma 4 26B | Negative CE entropy top5 | CE hand feature | 1195 | -0.007 | -0.006 | -0.005 | 0.048 | 0.019 | 0.015 |
| BarExamQA | Gemma 4 26B | Dense query-top1 cosine | Dense QPP | 1195 | 0.022 | 0.014 | 0.011 | -0.068 | -0.071 | -0.056 |
| BarExamQA | Gemma 4 26B | Dense top-5 coherence | Dense QPP | 1195 | 0.022 | 0.026 | 0.021 | -0.059 | -0.057 | -0.045 |
| BarExamQA | Gemma 4 26B | Dense top-5 centroid norm | Dense QPP | 1195 | 0.023 | 0.026 | 0.021 | -0.059 | -0.057 | -0.045 |
| BarExamQA | Gemma 4 26B | Log perplexity | Prior axis | 1195 | -0.045 | -0.043 | -0.035 | 0.020 | 0.015 | 0.012 |
| BarExamQA | Gemma 4 26B | Question tokens | Prior axis | 1195 | -0.005 | -0.007 | -0.006 | -0.033 | -0.035 | -0.028 |
| BarExamQA | Groq Llama 70B | NQC-CE top10 | NQC | 1195 | 0.020 | 0.024 | 0.019 | -0.006 | -0.009 | -0.007 |
| BarExamQA | Groq Llama 70B | WIG-CE top5-vs-top10 | WIG | 1195 | -0.005 | -0.012 | -0.009 | -0.045 | -0.028 | -0.023 |
| BarExamQA | Groq Llama 70B | SMV-CE top10 | SMV | 1195 | -0.016 | -0.023 | -0.019 | -0.032 | -0.036 | -0.029 |
| BarExamQA | Groq Llama 70B | Top-1 CE score | CE hand feature | 1195 | 0.017 | 0.015 | 0.012 | -0.021 | -0.013 | -0.010 |
| BarExamQA | Groq Llama 70B | Mean top-5 CE | CE hand feature | 1195 | 0.021 | 0.017 | 0.014 | -0.012 | -0.000 | -0.000 |
| BarExamQA | Groq Llama 70B | CE spread top1-top5 | CE hand feature | 1195 | -0.010 | 0.004 | 0.004 | -0.038 | -0.036 | -0.029 |
| BarExamQA | Groq Llama 70B | Negative CE entropy top5 | CE hand feature | 1195 | -0.022 | 0.007 | 0.006 | -0.040 | -0.042 | -0.034 |
| BarExamQA | Groq Llama 70B | Dense query-top1 cosine | Dense QPP | 1195 | -0.010 | -0.015 | -0.012 | 0.012 | 0.011 | 0.009 |
| BarExamQA | Groq Llama 70B | Dense top-5 coherence | Dense QPP | 1195 | -0.043 | -0.028 | -0.023 | -0.004 | 0.005 | 0.004 |
| BarExamQA | Groq Llama 70B | Dense top-5 centroid norm | Dense QPP | 1195 | -0.042 | -0.028 | -0.023 | -0.003 | 0.005 | 0.004 |
| BarExamQA | Groq Llama 70B | Log perplexity | Prior axis | 1195 | -0.031 | -0.027 | -0.022 | 0.037 | 0.038 | 0.030 |
| BarExamQA | Groq Llama 70B | Question tokens | Prior axis | 1195 | 0.027 | 0.022 | 0.018 | -0.000 | -0.013 | -0.010 |
| BarExamQA | Pooled | NQC-CE top10 | NQC | 3585 | 0.004 | -0.006 | -0.005 | -0.020 | -0.014 | -0.011 |
| BarExamQA | Pooled | WIG-CE top5-vs-top10 | WIG | 3585 | -0.025 | -0.030 | -0.025 | -0.017 | -0.020 | -0.016 |
| BarExamQA | Pooled | SMV-CE top10 | SMV | 3585 | -0.030 | -0.021 | -0.017 | 0.014 | -0.016 | -0.013 |
| BarExamQA | Pooled | Top-1 CE score | CE hand feature | 3585 | -0.003 | -0.010 | -0.008 | -0.017 | -0.015 | -0.012 |
| BarExamQA | Pooled | Mean top-5 CE | CE hand feature | 3585 | 0.001 | -0.006 | -0.005 | -0.016 | -0.013 | -0.010 |
| BarExamQA | Pooled | CE spread top1-top5 | CE hand feature | 3585 | -0.018 | -0.006 | -0.005 | -0.006 | -0.020 | -0.016 |
| BarExamQA | Pooled | Negative CE entropy top5 | CE hand feature | 3585 | -0.023 | -0.001 | -0.001 | -0.006 | -0.022 | -0.017 |
| BarExamQA | Pooled | Dense query-top1 cosine | Dense QPP | 3585 | -0.017 | -0.024 | -0.020 | -0.023 | -0.022 | -0.018 |
| BarExamQA | Pooled | Dense top-5 coherence | Dense QPP | 3585 | -0.018 | -0.009 | -0.008 | -0.041 | -0.033 | -0.026 |
| BarExamQA | Pooled | Dense top-5 centroid norm | Dense QPP | 3585 | -0.018 | -0.009 | -0.008 | -0.041 | -0.033 | -0.026 |
| BarExamQA | Pooled | Log perplexity | Prior axis | 3585 | -0.021 | -0.019 | -0.015 | 0.016 | 0.016 | 0.012 |
| BarExamQA | Pooled | Question tokens | Prior axis | 3585 | 0.024 | 0.022 | 0.018 | -0.004 | -0.009 | -0.007 |
| HousingQA state-filtered | Groq Llama 8B | NQC-CE top10 | NQC | 6853 | 0.002 | -0.082 | -0.065 | 0.000 | -0.019 | -0.015 |
| HousingQA state-filtered | Groq Llama 8B | WIG-CE top5-vs-top10 | WIG | 6853 | -0.108 | -0.115 | -0.090 | 0.004 | 0.001 | 0.001 |
| HousingQA state-filtered | Groq Llama 8B | SMV-CE top10 | SMV | 6853 | -0.029 | -0.035 | -0.028 | 0.012 | 0.023 | 0.018 |
| HousingQA state-filtered | Groq Llama 8B | Top-1 CE score | CE hand feature | 6853 | -0.056 | -0.056 | -0.044 | -0.008 | -0.008 | -0.006 |
| HousingQA state-filtered | Groq Llama 8B | Mean top-5 CE | CE hand feature | 6853 | -0.044 | -0.044 | -0.035 | -0.016 | -0.017 | -0.013 |
| HousingQA state-filtered | Groq Llama 8B | CE spread top1-top5 | CE hand feature | 6853 | -0.063 | -0.062 | -0.049 | 0.025 | 0.025 | 0.020 |
| HousingQA state-filtered | Groq Llama 8B | Negative CE entropy top5 | CE hand feature | 6853 | -0.035 | -0.042 | -0.033 | 0.021 | 0.024 | 0.019 |
| HousingQA state-filtered | Groq Llama 8B | Dense query-top1 cosine | Dense QPP | 6853 | -0.064 | -0.062 | -0.049 | 0.020 | 0.019 | 0.015 |
| HousingQA state-filtered | Groq Llama 8B | Dense top-5 coherence | Dense QPP | 6853 | -0.046 | -0.054 | -0.043 | -0.031 | -0.032 | -0.025 |
| HousingQA state-filtered | Groq Llama 8B | Dense top-5 centroid norm | Dense QPP | 6853 | -0.047 | -0.054 | -0.043 | -0.031 | -0.032 | -0.025 |
| HousingQA state-filtered | Groq Llama 8B | Log perplexity | Prior axis | 6853 | -0.036 | -0.051 | -0.040 | -0.049 | -0.049 | -0.039 |
| HousingQA state-filtered | Groq Llama 8B | Question tokens | Prior axis | 6853 | -0.006 | -0.017 | -0.014 | 0.012 | 0.017 | 0.014 |
| HousingQA state-filtered | Gemma 4 26B | NQC-CE top10 | NQC | 6853 | -0.023 | -0.087 | -0.068 | 0.000 | 0.011 | 0.009 |
| HousingQA state-filtered | Gemma 4 26B | WIG-CE top5-vs-top10 | WIG | 6853 | -0.089 | -0.091 | -0.072 | 0.035 | 0.037 | 0.030 |
| HousingQA state-filtered | Gemma 4 26B | SMV-CE top10 | SMV | 6853 | 0.002 | -0.009 | -0.007 | 0.020 | 0.023 | 0.018 |
| HousingQA state-filtered | Gemma 4 26B | Top-1 CE score | CE hand feature | 6853 | -0.084 | -0.086 | -0.067 | 0.000 | -0.006 | -0.005 |
| HousingQA state-filtered | Gemma 4 26B | Mean top-5 CE | CE hand feature | 6853 | -0.077 | -0.075 | -0.059 | 0.004 | -0.000 | -0.000 |
| HousingQA state-filtered | Gemma 4 26B | CE spread top1-top5 | CE hand feature | 6853 | -0.048 | -0.053 | -0.041 | -0.001 | -0.000 | -0.000 |
| HousingQA state-filtered | Gemma 4 26B | Negative CE entropy top5 | CE hand feature | 6853 | -0.027 | -0.036 | -0.028 | -0.008 | -0.008 | -0.006 |
| HousingQA state-filtered | Gemma 4 26B | Dense query-top1 cosine | Dense QPP | 6853 | -0.071 | -0.073 | -0.058 | 0.007 | 0.005 | 0.004 |
| HousingQA state-filtered | Gemma 4 26B | Dense top-5 coherence | Dense QPP | 6853 | -0.044 | -0.050 | -0.039 | -0.020 | -0.022 | -0.017 |
| HousingQA state-filtered | Gemma 4 26B | Dense top-5 centroid norm | Dense QPP | 6853 | -0.045 | -0.050 | -0.039 | -0.020 | -0.022 | -0.017 |
| HousingQA state-filtered | Gemma 4 26B | Log perplexity | Prior axis | 6853 | -0.049 | -0.060 | -0.047 | -0.052 | -0.030 | -0.024 |
| HousingQA state-filtered | Gemma 4 26B | Question tokens | Prior axis | 6853 | -0.008 | -0.017 | -0.014 | 0.000 | -0.001 | -0.001 |
| HousingQA state-filtered | Groq Llama 70B | NQC-CE top10 | NQC | 6853 | -0.018 | -0.102 | -0.081 | 0.001 | -0.016 | -0.012 |
| HousingQA state-filtered | Groq Llama 70B | WIG-CE top5-vs-top10 | WIG | 6853 | -0.087 | -0.086 | -0.068 | -0.023 | -0.021 | -0.016 |
| HousingQA state-filtered | Groq Llama 70B | SMV-CE top10 | SMV | 6853 | -0.000 | -0.003 | -0.002 | -0.002 | 0.002 | 0.002 |
| HousingQA state-filtered | Groq Llama 70B | Top-1 CE score | CE hand feature | 6853 | -0.091 | -0.091 | -0.072 | -0.019 | -0.021 | -0.016 |
| HousingQA state-filtered | Groq Llama 70B | Mean top-5 CE | CE hand feature | 6853 | -0.083 | -0.083 | -0.066 | -0.017 | -0.017 | -0.013 |
| HousingQA state-filtered | Groq Llama 70B | CE spread top1-top5 | CE hand feature | 6853 | -0.049 | -0.045 | -0.036 | -0.014 | -0.010 | -0.008 |
| HousingQA state-filtered | Groq Llama 70B | Negative CE entropy top5 | CE hand feature | 6853 | -0.032 | -0.033 | -0.026 | -0.003 | -0.004 | -0.003 |
| HousingQA state-filtered | Groq Llama 70B | Dense query-top1 cosine | Dense QPP | 6853 | -0.067 | -0.069 | -0.055 | -0.017 | -0.019 | -0.015 |
| HousingQA state-filtered | Groq Llama 70B | Dense top-5 coherence | Dense QPP | 6853 | -0.049 | -0.059 | -0.047 | -0.001 | -0.007 | -0.005 |
| HousingQA state-filtered | Groq Llama 70B | Dense top-5 centroid norm | Dense QPP | 6853 | -0.050 | -0.059 | -0.047 | -0.002 | -0.007 | -0.005 |
| HousingQA state-filtered | Groq Llama 70B | Log perplexity | Prior axis | 6853 | -0.054 | -0.090 | -0.072 | -0.038 | -0.016 | -0.013 |
| HousingQA state-filtered | Groq Llama 70B | Question tokens | Prior axis | 6853 | -0.043 | -0.033 | -0.027 | -0.007 | -0.003 | -0.002 |
| HousingQA state-filtered | Pooled | NQC-CE top10 | NQC | 20559 | -0.013 | -0.089 | -0.070 | 0.000 | -0.009 | -0.007 |
| HousingQA state-filtered | Pooled | WIG-CE top5-vs-top10 | WIG | 20559 | -0.094 | -0.097 | -0.076 | 0.004 | 0.005 | 0.004 |
| HousingQA state-filtered | Pooled | SMV-CE top10 | SMV | 20559 | -0.009 | -0.016 | -0.013 | 0.010 | 0.016 | 0.013 |
| HousingQA state-filtered | Pooled | Top-1 CE score | CE hand feature | 20559 | -0.077 | -0.076 | -0.060 | -0.009 | -0.011 | -0.009 |
| HousingQA state-filtered | Pooled | Mean top-5 CE | CE hand feature | 20559 | -0.067 | -0.066 | -0.052 | -0.010 | -0.012 | -0.009 |
| HousingQA state-filtered | Pooled | CE spread top1-top5 | CE hand feature | 20559 | -0.053 | -0.053 | -0.042 | 0.005 | 0.006 | 0.005 |
| HousingQA state-filtered | Pooled | Negative CE entropy top5 | CE hand feature | 20559 | -0.031 | -0.037 | -0.029 | 0.005 | 0.005 | 0.004 |
| HousingQA state-filtered | Pooled | Dense query-top1 cosine | Dense QPP | 20559 | -0.067 | -0.068 | -0.053 | 0.004 | 0.003 | 0.002 |
| HousingQA state-filtered | Pooled | Dense top-5 coherence | Dense QPP | 20559 | -0.046 | -0.054 | -0.043 | -0.018 | -0.021 | -0.016 |
| HousingQA state-filtered | Pooled | Dense top-5 centroid norm | Dense QPP | 20559 | -0.047 | -0.054 | -0.043 | -0.018 | -0.021 | -0.016 |
| HousingQA state-filtered | Pooled | Log perplexity | Prior axis | 20559 | -0.046 | -0.066 | -0.052 | -0.046 | -0.033 | -0.026 |
| HousingQA state-filtered | Pooled | Question tokens | Prior axis | 20559 | -0.019 | -0.022 | -0.018 | 0.002 | 0.005 | 0.004 |
| All | Pooled | NQC-CE top10 | NQC | 24144 | -0.013 | -0.104 | -0.082 | 0.000 | -0.019 | -0.015 |
| All | Pooled | WIG-CE top5-vs-top10 | WIG | 24144 | -0.132 | -0.139 | -0.109 | -0.020 | -0.020 | -0.016 |
| All | Pooled | SMV-CE top10 | SMV | 24144 | -0.039 | -0.067 | -0.053 | -0.003 | -0.008 | -0.006 |
| All | Pooled | Top-1 CE score | CE hand feature | 24144 | -0.079 | -0.082 | -0.065 | -0.014 | -0.016 | -0.013 |
| All | Pooled | Mean top-5 CE | CE hand feature | 24144 | -0.059 | -0.059 | -0.047 | -0.010 | -0.012 | -0.009 |
| All | Pooled | CE spread top1-top5 | CE hand feature | 24144 | -0.088 | -0.095 | -0.075 | -0.013 | -0.015 | -0.012 |
| All | Pooled | Negative CE entropy top5 | CE hand feature | 24144 | -0.064 | -0.079 | -0.062 | -0.011 | -0.015 | -0.012 |
| All | Pooled | Dense query-top1 cosine | Dense QPP | 24144 | -0.012 | -0.012 | -0.009 | 0.019 | 0.019 | 0.015 |
| All | Pooled | Dense top-5 coherence | Dense QPP | 24144 | -0.011 | -0.016 | -0.013 | -0.008 | -0.009 | -0.007 |
| All | Pooled | Dense top-5 centroid norm | Dense QPP | 24144 | -0.012 | -0.016 | -0.013 | -0.008 | -0.009 | -0.007 |
| All | Pooled | Log perplexity | Prior axis | 24144 | -0.032 | -0.037 | -0.029 | -0.036 | -0.018 | -0.014 |
| All | Pooled | Question tokens | Prior axis | 24144 | 0.100 | 0.055 | 0.044 | 0.040 | 0.032 | 0.026 |

## Strongest Named Predictor Curve

The strongest named QPP predictor by absolute pooled Kendall tau on retrieval delta is `WIG-CE top5-vs-top10` (Kendall -0.109, Spearman -0.139).

| Bin | N | Predictor median | Predictor range | SCOPE retrieval win | Raw retrieval win | Net retrieval delta | SCOPE answer win | Raw answer win | Net answer delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4829 | 0.404 | 0.024-0.604 | 12.1% | 5.3% | 6.8% | 11.6% | 10.7% | 0.9% |
| 2 | 4829 | 0.748 | 0.604-0.864 | 13.6% | 13.5% | 0.2% | 10.9% | 12.7% | -1.8% |
| 3 | 4828 | 0.977 | 0.864-1.105 | 11.2% | 17.5% | -6.2% | 11.0% | 13.0% | -2.0% |
| 4 | 4829 | 1.250 | 1.105-1.427 | 9.8% | 19.5% | -9.7% | 10.8% | 12.8% | -2.0% |
| 5 | 4829 | 1.695 | 1.427-3.756 | 7.4% | 20.0% | -12.6% | 9.9% | 11.8% | -2.0% |

## Selective SCOPE Routing Simulation

This in-sample diagnostic routes to SCOPE for the low-confidence side of the strongest predictor unless the learned sign is positive, in which case the threshold direction is inverted. It is a screening result, not a locked deployment threshold.

| Scope | Quantile | Threshold | SCOPE fraction | Routed Hit@5 | vs raw | vs SCOPE | Routed acc | vs raw | vs SCOPE | Captured SCOPE wins | Avoided SCOPE hurts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | 0.2 | 0.253 | 20.0% | 3.3% | 1.9% | -7.5% | 70.2% | 1.2% | -2.6% | 2.2% | 0.8% |
| BarExamQA | 0.4 | 0.343 | 40.0% | 6.0% | 4.6% | -4.9% | 70.9% | 1.9% | -2.0% | 4.9% | 0.7% |
| BarExamQA | 0.6 | 0.456 | 60.0% | 7.5% | 6.1% | -3.3% | 71.3% | 2.3% | -1.6% | 6.7% | 0.5% |
| BarExamQA | 0.8 | 0.630 | 80.0% | 8.9% | 7.4% | -2.0% | 72.3% | 3.3% | -0.6% | 8.2% | 0.3% |
| HousingQA | 0.2 | 0.749 | 20.0% | 37.2% | 0.3% | 7.0% | 62.9% | -0.6% | 1.7% | 2.8% | 15.1% |
| HousingQA | 0.4 | 0.954 | 40.0% | 36.5% | -0.4% | 6.3% | 62.5% | -1.0% | 1.3% | 5.4% | 11.8% |
| HousingQA | 0.6 | 1.185 | 60.0% | 34.9% | -2.1% | 4.6% | 62.2% | -1.4% | 0.9% | 7.6% | 8.0% |
| HousingQA | 0.8 | 1.487 | 80.0% | 32.8% | -4.2% | 2.5% | 61.7% | -1.8% | 0.4% | 9.5% | 4.0% |
| All | 0.2 | 0.604 | 20.0% | 33.0% | 1.4% | 5.7% | 64.5% | 0.2% | 1.6% | 2.4% | 14.1% |
| All | 0.4 | 0.864 | 40.0% | 33.1% | 1.4% | 5.7% | 64.1% | -0.2% | 1.2% | 5.2% | 11.4% |
| All | 0.6 | 1.105 | 60.0% | 31.8% | 0.2% | 4.5% | 63.8% | -0.6% | 0.8% | 7.4% | 7.9% |
| All | 0.8 | 1.427 | 80.0% | 29.9% | -1.8% | 2.5% | 63.4% | -1.0% | 0.4% | 9.4% | 4.0% |

## Reading

- The no-gold QPP transfer is useful as a weak routing diagnostic but does not clear the `|Kendall tau| >= 0.5` reliability bar. The best pooled named predictor is `WIG-CE top5-vs-top10` with Kendall -0.109; passing predictors: none.
- Per dataset, the strongest named retrieval-delta predictors are `WIG-CE top5-vs-top10` on BarExamQA (Kendall -0.025) and `WIG-CE top5-vs-top10` on HousingQA (Kendall -0.076). This supports the Faggioli-style caution: score-based QPP transfer has to be validated on the actual neural scores and does not behave like a guaranteed oracle.
- Answer-delta prediction is weaker than retrieval-delta prediction. The strongest pooled named answer predictor is `WIG-CE top5-vs-top10` with Kendall -0.016. This matches the previous mechanism reports: QPP can screen for retrieval repair opportunities, but answer conversion remains noisier.
- Selective query expansion is therefore viable as a conservative research direction, not yet as a standalone gate. The safest next step is to learn or calibrate thresholds on a held-out slice, then test whether routed SCOPE preserves BarExam retrieval gains while avoiding Housing dilution.

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

- Unigram LM cache directory for comparison axes: `/tmp/perplexity_axis_lm_cache_2026-05-25`
- Chroma collections read for dense QPP features: `legal_passages`, `housing_statutes`
- Gold-needed comparison source: `docs/generated/scope_gap_mechanism_2026-05-25.md`

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_raw_retrieval_confidence_routing.py \
  --output docs/generated/raw_retrieval_confidence_routing_2026-05-26.md
```
