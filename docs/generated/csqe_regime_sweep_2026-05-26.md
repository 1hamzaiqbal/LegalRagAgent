# CSQE Regime Sweep - 2026-05-26

## Scope

This is a read-from-cache regime sweep for corpus-steered query expansion (CSQE). The only new arm is CSQE on BarExamQA and HousingQA state-filtered; raw, HyDE, and SCOPE use the existing signed retrieval caches. The BEIR rows are the strong-query Phase-A reference from `docs/generated/exemplar_scope_select_2026-05-26.md`.

HousingQA is interpreted as retrieval-only here: prior answer results showed answer conversion is the binding issue, so this table should not be read as downstream answer accuracy.

## Verdicts

| Hypothesis | Verdict | Key read |
|---|---|---|
| H-collapse | **supported** | BarExam CSQE 2.0% vs Raw 1.4%; HyDE 11.4%, SCOPE 12.1%. |
| H-scope-wins-weak | **supported** | On BarExam, best parametric expansion is 12.1% vs CSQE 2.0%. |
| H-csqe-strong | **killed** | Housing retrieval-only: CSQE 37.5% vs SCOPE 38.1%, Raw 36.9%. |
| Net crossover | **mixed** | BEIR pooled all-arm best=raw; expansion-arm best=CSQE. Raw 62.2%, CSQE 59.4%, SCOPE 49.8%, HyDE 30.8%. BarExam remains the single weak-query legal set here, so treat the weak-end read as provisional. |

## Regime Sweep Table

| Dataset | Regime | Arm | N | Hit@5 | Correct | RI vs raw | Help | Hurt | Mean CE gold-affinity delta vs raw |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | weak | Raw | 1195 | 1.4% | 17 | 0.000 | 0 | 0 | -- |
| BarExamQA | weak | HyDE | 1195 | 11.4% | 136 | 0.100 | 130 | 11 | 4.125 |
| BarExamQA | weak | SCOPE | 1195 | 12.1% | 144 | 0.106 | 138 | 11 | 3.885 |
| BarExamQA | weak | CSQE | 1195 | 2.0% | 24 | 0.006 | 15 | 8 | -0.526 |
| HousingQA state-filtered | intermediate | Raw | 6853 | 36.9% | 2532 | 0.000 | 0 | 0 | -- |
| HousingQA state-filtered | intermediate | HyDE | 6853 | 30.6% | 2099 | -0.063 | 864 | 1297 | 3.360 |
| HousingQA state-filtered | intermediate | SCOPE | 6853 | 38.1% | 2609 | 0.011 | 1023 | 946 | 2.990 |
| HousingQA state-filtered | intermediate | CSQE | 6853 | 37.5% | 2572 | 0.006 | 507 | 467 | 4.567 |
| SciFact | strong | Raw | 300 | 82.0% | 246 | 0.000 | 0 | 0 | -- |
| SciFact | strong | HyDE | 300 | 35.0% | 105 | -0.470 | 12 | 153 | -7.360 |
| SciFact | strong | SCOPE | 300 | 65.7% | 197 | -0.163 | 12 | 61 | -0.909 |
| SciFact | strong | CSQE | 300 | 78.3% | 235 | -0.037 | 8 | 19 | -0.837 |
| NFCorpus | strong | Raw | 323 | 69.3% | 224 | 0.000 | 0 | 0 | -- |
| NFCorpus | strong | HyDE | 323 | 33.4% | 108 | -0.359 | 6 | 122 | -5.005 |
| NFCorpus | strong | SCOPE | 323 | 65.0% | 210 | -0.043 | 20 | 34 | -0.919 |
| NFCorpus | strong | CSQE | 323 | 61.9% | 200 | -0.074 | 6 | 30 | 0.324 |
| FiQA | strong | Raw | 648 | 66.2% | 429 | 0.000 | 0 | 0 | -- |
| FiQA | strong | HyDE | 648 | 32.3% | 209 | -0.340 | 38 | 258 | -4.055 |
| FiQA | strong | SCOPE | 648 | 35.2% | 228 | -0.310 | 25 | 226 | -2.947 |
| FiQA | strong | CSQE | 648 | 63.9% | 414 | -0.023 | 23 | 38 | -1.351 |
| TREC-COVID | strong | Raw | 50 | 98.0% | 49 | 0.000 | 0 | 0 | -- |
| TREC-COVID | strong | HyDE | 50 | 70.0% | 35 | -0.280 | 1 | 15 | -7.662 |
| TREC-COVID | strong | SCOPE | 50 | 96.0% | 48 | -0.020 | 1 | 2 | -1.824 |
| TREC-COVID | strong | CSQE | 50 | 98.0% | 49 | 0.000 | 0 | 0 | -3.305 |
| SciDocs | strong | Raw | 989 | 49.3% | 488 | 0.000 | 0 | 0 | -- |
| SciDocs | strong | HyDE | 989 | 25.8% | 255 | -0.236 | 58 | 291 | -3.286 |
| SciDocs | strong | SCOPE | 989 | 47.2% | 467 | -0.021 | 84 | 105 | 1.298 |
| SciDocs | strong | CSQE | 989 | 47.9% | 474 | -0.014 | 56 | 70 | 0.240 |
| BEIR pooled reference | strong | Raw | 2310 | 62.2% | 1436 | 0.000 | 0 | 0 | -- |
| BEIR pooled reference | strong | HyDE | 2310 | 30.8% | 712 | -0.313 | 115 | 839 | -4.366 |
| BEIR pooled reference | strong | SCOPE | 2310 | 49.8% | 1150 | -0.124 | 142 | 428 | -0.557 |
| BEIR pooled reference | strong | CSQE | 2310 | 59.4% | 1372 | -0.028 | 93 | 157 | -0.411 |

## Reading

The weak-query BarExamQA control is the sharpest legal check: raw retrieval almost never exposes gold evidence, so CSQE has little useful real text to extract from the raw top-k. A CSQE gain there would have killed the collapse hypothesis.

HousingQA sits between regimes. CSQE can reuse top-ranked state-filtered statutory language, but it is still not a downstream answer claim. Treat it as evidence about retrieval exposure only.

The BEIR reference is stronger-query retrieval: raw is already competitive, and CSQE is the expansion-style arm that preserves most of that raw strength. That makes the aggregate crossover a retrieval-regime pattern rather than a universal CSQE win.

## Artifacts

- Per-row legal plus BEIR summary points: `docs/generated/csqe_regime_sweep_2026-05-26_points.jsonl`
- BEIR point source used for this run: `/tmp/exemplar_scope_select_2026-05-26_points.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/hyre/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_csqe_k10.jsonl`
- `caches/generation/full/barexam_qfull_seed42_or-gemma4-26b_csqe.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_csqe_k10.jsonl`
- `caches/generation/full/housing_qfull_seed42_statefilter_or-gemma4-26b_csqe.jsonl`
- `docs/generated/exemplar_scope_select_2026-05-26.md`
- `caches/retrieval/full/beir_scifact_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_csqe.jsonl`
