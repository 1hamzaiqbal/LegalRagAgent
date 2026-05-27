# HotpotQA Distractor Weak-Query Verification

## Verdict

The q1000 distractor slice is clean but does **not** show the expected expansion help side. Raw-question retrieval is already near-saturated on standard Hit@5 and is strongest on the multi-hop metrics. Because SCOPE and HyDE both net-hurt full-support and bridge recall, I did not scale to full N=7405 and did not run the optional q500 answer EM.

## Source Files

| Role | Path |
|---|---|
| Raw retrieval | `caches/retrieval/full/hotpotqa_q1000_seed42_raw_question_k10.jsonl` |
| HyDE generation | `caches/generation/full/hotpotqa_q1000_seed42_or-gemma4-26b_rag_hyde.jsonl` |
| HyDE retrieval | `caches/retrieval/full/hotpotqa_q1000_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` |
| SCOPE / snap_hyre generation | `caches/generation/full/hotpotqa_q1000_seed42_or-gemma4-26b_snap_hyre.jsonl` |
| SCOPE / snap_hyre retrieval | `caches/retrieval/full/hotpotqa_q1000_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` |
| Dataset questions | `datasets/hotpotqa_distractor/questions.csv` |
| Per-question paragraphs | `datasets/hotpotqa_distractor/passages.csv` |

## Cache Health

| Cache | Rows | Duplicates | Errors | Missing passage | Parse bad | Answer artifact | Short retrieval | Format retry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw retrieval | 1000 | 0 | 0 | -- | -- | -- | 6 | -- |
| HyDE generation | 1000 | 0 | 0 | 0 | 0 | 0 | -- | 0 |
| HyDE retrieval | 1000 | 0 | 0 | -- | -- | -- | 6 | -- |
| SCOPE / snap_hyre generation | 1000 | 0 | 0 | 0 | 0 | 0 | -- | 0 |
| SCOPE / snap_hyre retrieval | 1000 | 0 | 0 | -- | -- | -- | 6 | -- |

Short retrieval lists are expected for the six validation rows whose original HotpotQA candidate set has fewer than ten paragraphs.

## Retrieval Metrics

| Method | Hit@5 | Full-support@2 | Full-support@5 | Bridge@2 | Bridge@5 |
|---|---:|---:|---:|---:|---:|
| Raw question | 99.4% | 44.0% | 70.6% | 44.0% | 70.6% |
| HyDE | 93.2% | 14.9% | 51.0% | 37.5% | 67.9% |
| SCOPE / snap_hyre | 98.0% | 26.5% | 58.1% | 39.4% | 63.2% |

Bridge paragraph = the gold paragraph with the lower raw-query CE score for that question.

## Expansion vs Raw

| Method | Metric | Delta | Help rows | Hurt rows | RI |
|---|---|---:|---:|---:|---:|
| HyDE | hit@5 | -6.2pp | 4 | 66 | -0.062 |
| HyDE | full@2 | -29.1pp | 58 | 349 | -0.291 |
| HyDE | full@5 | -19.6pp | 112 | 308 | -0.196 |
| HyDE | bridge@2 | -6.5pp | 184 | 249 | -0.065 |
| HyDE | bridge@5 | -2.7pp | 165 | 192 | -0.027 |
| SCOPE / snap_hyre | hit@5 | -1.4pp | 2 | 16 | -0.014 |
| SCOPE / snap_hyre | full@2 | -17.5pp | 74 | 249 | -0.175 |
| SCOPE / snap_hyre | full@5 | -12.5pp | 77 | 202 | -0.125 |
| SCOPE / snap_hyre | bridge@2 | -4.6pp | 129 | 175 | -0.046 |
| SCOPE / snap_hyre | bridge@5 | -7.4pp | 84 | 158 | -0.074 |

## Mechanism Correlations

| Method | Axis | Gain metric | N | Spearman | Kendall |
|---|---|---|---:|---:|---:|
| HyDE | delta margin | full@5 | 999 | 0.143 | 0.112 |
| HyDE | delta margin | bridge@5 | 999 | 0.161 | 0.127 |
| HyDE | delta margin | hit@5 | 999 | 0.135 | 0.111 |
| HyDE | gold-affinity delta | full@5 | 1000 | 0.005 | 0.004 |
| HyDE | gold-affinity delta | bridge@5 | 1000 | -0.001 | -0.000 |
| HyDE | gold-affinity delta | hit@5 | 1000 | 0.114 | 0.093 |
| HyDE | raw margin | full@5 | 999 | -0.010 | -0.007 |
| HyDE | raw margin | bridge@5 | 999 | -0.072 | -0.057 |
| HyDE | raw margin | hit@5 | 999 | -0.016 | -0.013 |
| HyDE | log perplexity | full@5 | 1000 | -0.115 | -0.091 |
| HyDE | log perplexity | bridge@5 | 1000 | -0.136 | -0.107 |
| HyDE | log perplexity | hit@5 | 1000 | -0.035 | -0.029 |
| HyDE | OOV rate | full@5 | 1000 | 0.041 | 0.039 |
| HyDE | OOV rate | bridge@5 | 1000 | 0.040 | 0.038 |
| HyDE | OOV rate | hit@5 | 1000 | -0.033 | -0.033 |
| SCOPE / snap_hyre | delta margin | full@5 | 999 | 0.183 | 0.146 |
| SCOPE / snap_hyre | delta margin | bridge@5 | 999 | 0.153 | 0.122 |
| SCOPE / snap_hyre | delta margin | hit@5 | 999 | 0.189 | 0.154 |
| SCOPE / snap_hyre | gold-affinity delta | full@5 | 1000 | 0.001 | 0.001 |
| SCOPE / snap_hyre | gold-affinity delta | bridge@5 | 1000 | -0.006 | -0.005 |
| SCOPE / snap_hyre | gold-affinity delta | hit@5 | 1000 | 0.168 | 0.137 |
| SCOPE / snap_hyre | raw margin | full@5 | 999 | 0.046 | 0.037 |
| SCOPE / snap_hyre | raw margin | bridge@5 | 999 | 0.010 | 0.008 |
| SCOPE / snap_hyre | raw margin | hit@5 | 999 | 0.033 | 0.027 |
| SCOPE / snap_hyre | log perplexity | full@5 | 1000 | -0.069 | -0.055 |
| SCOPE / snap_hyre | log perplexity | bridge@5 | 1000 | -0.086 | -0.069 |
| SCOPE / snap_hyre | log perplexity | hit@5 | 1000 | 0.035 | 0.028 |
| SCOPE / snap_hyre | OOV rate | full@5 | 1000 | 0.048 | 0.046 |
| SCOPE / snap_hyre | OOV rate | bridge@5 | 1000 | 0.044 | 0.042 |
| SCOPE / snap_hyre | OOV rate | hit@5 | 1000 | 0.024 | 0.024 |

Margin correlations have N=999 because one q1000 row has no non-gold candidate available for the distractor maximum; retrieval metrics use all 1000 rows.

## Raw-Margin Quintiles

| Method | Raw-margin bin | N | Raw full@5 | Method full@5 | Delta | Raw bridge@5 | Method bridge@5 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HyDE | Q1 | 200 | 50.0% | 41.5% | -8.5pp | 50.0% | 61.5% | +11.5pp |
| HyDE | Q2 | 200 | 75.0% | 45.5% | -29.5pp | 75.0% | 64.0% | -11.0pp |
| HyDE | Q3 | 199 | 74.9% | 48.2% | -26.6pp | 74.9% | 67.8% | -7.0pp |
| HyDE | Q4 | 200 | 70.0% | 56.0% | -14.0pp | 70.0% | 71.0% | +1.0pp |
| HyDE | Q5 | 200 | 83.0% | 63.5% | -19.5pp | 83.0% | 75.0% | -8.0pp |
| SCOPE / snap_hyre | Q1 | 200 | 50.0% | 42.0% | -8.0pp | 50.0% | 49.0% | -1.0pp |
| SCOPE / snap_hyre | Q2 | 200 | 75.0% | 51.5% | -23.5pp | 75.0% | 59.5% | -15.5pp |
| SCOPE / snap_hyre | Q3 | 199 | 74.9% | 59.3% | -15.6pp | 74.9% | 63.3% | -11.6pp |
| SCOPE / snap_hyre | Q4 | 200 | 70.0% | 63.5% | -6.5pp | 70.0% | 67.0% | -3.0pp |
| SCOPE / snap_hyre | Q5 | 200 | 83.0% | 74.0% | -9.0pp | 83.0% | 77.0% | -6.0pp |

## P4 Failure AUC

Target is `delta margin < 0`; geometry uses `{M_raw, CE(exp,gold)}`, surprise uses `{OOV, log perplexity}`.

| Method | Failure rate | Geometry AUC | Surprise AUC |
|---|---:|---:|---:|
| HyDE | 87.6% | 0.945 | 0.568 |
| SCOPE / snap_hyre | 52.6% | 0.718 | 0.514 |

## SCOPE vs HyDE

| Metric | SCOPE minus HyDE | SCOPE-help rows | HyDE-help rows |
|---|---:|---:|---:|
| hit@5 | +4.8pp | 2 | 4 |
| full@2 | +11.6pp | 74 | 58 |
| full@5 | +7.1pp | 77 | 112 |
| bridge@2 | +1.9pp | 129 | 184 |
| bridge@5 | -4.7pp | 84 | 165 |

## Reading

HotpotQA distractor q1000 is not weak enough in this within-question candidate form. The raw query sees all ten candidate paragraphs and the CE reranker usually places at least one gold paragraph in the top five; the harder full-support and bridge metrics are still best under raw retrieval. SCOPE is less damaging than HyDE on Hit@5 and full-support, but it does not cross into net-help.

The margin mechanism is only weakly visible: SCOPE delta margin has the largest positive correlation with gain, but it stays below a strong explanatory threshold and low raw-margin bins still do not become a positive SCOPE regime. P4 does rule out corpus-surprise as the main failure explanation here: OOV/log-perplexity are near chance, while geometry is much more predictive of negative margin movement. That geometry result is partly expected because the target is defined from the margin itself, so it should not be read as evidence that expansion helps in this setting. For the help-side benchmark, the next better target is a setting with a larger candidate pool or weaker literal query anchoring, such as full-wiki HotpotQA or MuSiQue.

## Decision

- Stop HotpotQA distractor at q1000 for this lane.
- Do not scale to full 7405 under the current per-question distractor retrieval setup.
- Do not run optional q500 answer EM because the retrieval premise is net-negative.
- Keep the q1000 caches as reusable artifacts for future prompt/selection comparisons.
