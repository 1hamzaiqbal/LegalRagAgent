# Exemplar-Grounded SCOPE Phase A - 2026-05-26

## Scope

Phase A tests a single exemplar-grounded SCOPE candidate against raw retrieval, HyDE, vanilla SCOPE, and deterministic CSQE on five BEIR strong-query sets. The only model calls were SCOPE-exemplar query-generation calls; no downstream answer cells were run, and no files under `paper/` were edited.

SciDocs note: every document in the local SciDocs corpus snapshot is a qrels positive for some eval query. To avoid exemplar leakage, the three selected medoid ids are treated as eval exclusions for SciDocs, removing 11/1000 rows from every Phase-A comparison.

## Verdicts

| Hypothesis | Verdict | Key read |
|---|---|---|
| H1 grounding cuts drift | **mixed** | SCOPE-exemplar pooled Hit@5 48.9% vs vanilla SCOPE 49.8%, HyDE 30.8%; mean CE gold delta -0.846 vs SCOPE -0.557 |
| H2 selection helps | **not run** | Phase B was gated on Phase A promise; Phase A did not beat vanilla SCOPE or CSQE pooled. |
| H3 net-positive on strong-query BEIR | **killed** | no arm had positive RI vs raw on any strong-query BEIR set |
| H4 snap-answer adds over CSQE | **killed** | SCOPE-exemplar pooled Hit@5 48.9%; CSQE pooled Hit@5 59.4%. |
| H5 weak-query intact | **not run** | Weak-query control belongs to Phase B and was not launched after the Phase A stop decision. |

Phase B decision: **stop for now**. The single-candidate exemplar arm does not beat vanilla SCOPE pooled and is far below CSQE pooled, so the selection arms are not justified under the pre-stated gate.

## Exemplar Guardrail

| Dataset | Source | Exemplar ids | Eval rows excluded |
|---|---|---|---:|
| SciFact | chroma | `4444861, 3052213, 581832` | 0 |
| NFCorpus | chroma | `MED-1034, MED-2235, MED-5007` | 0 |
| FiQA | chroma | `51311, 178061, 583912` | 0 |
| TREC-COVID | chroma | `k596omcy, 2xsjxjml, okqsvg8q` | 0 |
| SciDocs | chroma_with_eval_row_exclusion | `9e463eefadbcd336c69270a299666e4104d50159, 4017f984d1b4b8748a06da2739183782bbe9b46d, 1a090df137014acab572aa5dc23449b270db64b4` | 11 |

## Hit@5

| Dataset | Arm | N | Hit@5 | Correct | RI vs raw | Help vs raw | Hurt vs raw | Mean CE gold delta vs raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SciFact | Raw | 300 | 82.0% | 246 | 0.000 | 0 | 0 | -- |
| SciFact | HyDE | 300 | 35.0% | 105 | -0.470 | 12 | 153 | -7.360 |
| SciFact | SCOPE | 300 | 65.7% | 197 | -0.163 | 12 | 61 | -0.909 |
| SciFact | CSQE | 300 | 78.3% | 235 | -0.037 | 8 | 19 | -0.837 |
| SciFact | SCOPE-exemplar | 300 | 68.3% | 205 | -0.137 | 16 | 57 | -1.131 |
| NFCorpus | Raw | 323 | 69.3% | 224 | 0.000 | 0 | 0 | -- |
| NFCorpus | HyDE | 323 | 33.4% | 108 | -0.359 | 6 | 122 | -5.005 |
| NFCorpus | SCOPE | 323 | 65.0% | 210 | -0.043 | 20 | 34 | -0.919 |
| NFCorpus | CSQE | 323 | 61.9% | 200 | -0.074 | 6 | 30 | 0.324 |
| NFCorpus | SCOPE-exemplar | 323 | 63.8% | 206 | -0.056 | 14 | 32 | -0.706 |
| FiQA | Raw | 648 | 66.2% | 429 | 0.000 | 0 | 0 | -- |
| FiQA | HyDE | 648 | 32.3% | 209 | -0.340 | 38 | 258 | -4.055 |
| FiQA | SCOPE | 648 | 35.2% | 228 | -0.310 | 25 | 226 | -2.947 |
| FiQA | CSQE | 648 | 63.9% | 414 | -0.023 | 23 | 38 | -1.351 |
| FiQA | SCOPE-exemplar | 648 | 37.5% | 243 | -0.287 | 41 | 227 | -3.289 |
| TREC-COVID | Raw | 50 | 98.0% | 49 | 0.000 | 0 | 0 | -- |
| TREC-COVID | HyDE | 50 | 70.0% | 35 | -0.280 | 1 | 15 | -7.662 |
| TREC-COVID | SCOPE | 50 | 96.0% | 48 | -0.020 | 1 | 2 | -1.824 |
| TREC-COVID | CSQE | 50 | 98.0% | 49 | 0.000 | 0 | 0 | -3.305 |
| TREC-COVID | SCOPE-exemplar | 50 | 92.0% | 46 | -0.060 | 1 | 4 | -2.170 |
| SciDocs | Raw | 989 | 49.3% | 488 | 0.000 | 0 | 0 | -- |
| SciDocs | HyDE | 989 | 25.8% | 255 | -0.236 | 58 | 291 | -3.286 |
| SciDocs | SCOPE | 989 | 47.2% | 467 | -0.021 | 84 | 105 | 1.298 |
| SciDocs | CSQE | 989 | 47.9% | 474 | -0.014 | 56 | 70 | 0.240 |
| SciDocs | SCOPE-exemplar | 989 | 43.5% | 430 | -0.059 | 81 | 139 | 0.862 |
| Pooled | Raw | 2310 | 62.2% | 1436 | 0.000 | 0 | 0 | -- |
| Pooled | HyDE | 2310 | 30.8% | 712 | -0.313 | 115 | 839 | -4.366 |
| Pooled | SCOPE | 2310 | 49.8% | 1150 | -0.124 | 142 | 428 | -0.557 |
| Pooled | CSQE | 2310 | 59.4% | 1372 | -0.028 | 93 | 157 | -0.411 |
| Pooled | SCOPE-exemplar | 2310 | 48.9% | 1130 | -0.132 | 153 | 459 | -0.846 |

## RI Matrix

Each cell is Collins-Thompson `RI=(help-hurt)/N` for the row arm against the column baseline.

| Dataset | Arm | vs Raw | vs HyDE | vs SCOPE | vs CSQE |
|---|---|---:|---:|---:|---:|
| SciFact | HyDE | -0.470 | 0.000 | -0.307 | -0.433 |
| SciFact | SCOPE | -0.163 | 0.307 | 0.000 | -0.127 |
| SciFact | CSQE | -0.037 | 0.433 | 0.127 | 0.000 |
| SciFact | SCOPE-exemplar | -0.137 | 0.333 | 0.027 | -0.100 |
| NFCorpus | HyDE | -0.359 | 0.000 | -0.316 | -0.285 |
| NFCorpus | SCOPE | -0.043 | 0.316 | 0.000 | 0.031 |
| NFCorpus | CSQE | -0.074 | 0.285 | -0.031 | 0.000 |
| NFCorpus | SCOPE-exemplar | -0.056 | 0.303 | -0.012 | 0.019 |
| FiQA | HyDE | -0.340 | 0.000 | -0.029 | -0.316 |
| FiQA | SCOPE | -0.310 | 0.029 | 0.000 | -0.287 |
| FiQA | CSQE | -0.023 | 0.316 | 0.287 | 0.000 |
| FiQA | SCOPE-exemplar | -0.287 | 0.052 | 0.023 | -0.264 |
| TREC-COVID | HyDE | -0.280 | 0.000 | -0.260 | -0.280 |
| TREC-COVID | SCOPE | -0.020 | 0.260 | 0.000 | -0.020 |
| TREC-COVID | CSQE | 0.000 | 0.280 | 0.020 | 0.000 |
| TREC-COVID | SCOPE-exemplar | -0.060 | 0.220 | -0.040 | -0.060 |
| SciDocs | HyDE | -0.236 | 0.000 | -0.214 | -0.221 |
| SciDocs | SCOPE | -0.021 | 0.214 | 0.000 | -0.007 |
| SciDocs | CSQE | -0.014 | 0.221 | 0.007 | 0.000 |
| SciDocs | SCOPE-exemplar | -0.059 | 0.177 | -0.037 | -0.044 |
| Pooled | HyDE | -0.313 | 0.000 | -0.190 | -0.286 |
| Pooled | SCOPE | -0.124 | 0.190 | 0.000 | -0.096 |
| Pooled | CSQE | -0.028 | 0.286 | 0.096 | 0.000 |
| Pooled | SCOPE-exemplar | -0.132 | 0.181 | -0.009 | -0.105 |

## Key Contrasts

| Dataset | Arm | Baseline | N | Delta Hit@5 | 95% bootstrap CI | Arm-only | Baseline-only | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SciFact | HyDE | Raw | 300 | -47.0% | [-53.0%, -40.7%] | 12 | 153 | <0.001 |
| SciFact | SCOPE | Raw | 300 | -16.3% | [-21.7%, -11.0%] | 12 | 61 | <0.001 |
| SciFact | CSQE | Raw | 300 | -3.7% | [-7.0%, -0.7%] | 8 | 19 | 0.052 |
| SciFact | SCOPE-exemplar | Raw | 300 | -13.7% | [-19.0%, -8.3%] | 16 | 57 | <0.001 |
| SciFact | SCOPE-exemplar | HyDE | 300 | 33.3% | [26.7%, 40.0%] | 121 | 21 | <0.001 |
| SciFact | SCOPE-exemplar | SCOPE | 300 | 2.7% | [-1.3%, 7.3%] | 29 | 21 | 0.322 |
| SciFact | SCOPE-exemplar | CSQE | 300 | -10.0% | [-15.0%, -4.7%] | 18 | 48 | <0.001 |
| NFCorpus | HyDE | Raw | 323 | -35.9% | [-41.8%, -30.3%] | 6 | 122 | <0.001 |
| NFCorpus | SCOPE | Raw | 323 | -4.3% | [-9.0%, 0.3%] | 20 | 34 | 0.076 |
| NFCorpus | CSQE | Raw | 323 | -7.4% | [-11.1%, -4.0%] | 6 | 30 | <0.001 |
| NFCorpus | SCOPE-exemplar | Raw | 323 | -5.6% | [-9.9%, -1.5%] | 14 | 32 | 0.011 |
| NFCorpus | SCOPE-exemplar | HyDE | 323 | 30.3% | [24.5%, 36.2%] | 110 | 12 | <0.001 |
| NFCorpus | SCOPE-exemplar | SCOPE | 323 | -1.2% | [-5.3%, 2.8%] | 21 | 25 | 0.659 |
| NFCorpus | SCOPE-exemplar | CSQE | 323 | 1.9% | [-2.8%, 6.5%] | 31 | 25 | 0.504 |
| FiQA | HyDE | Raw | 648 | -34.0% | [-38.4%, -29.5%] | 38 | 258 | <0.001 |
| FiQA | SCOPE | Raw | 648 | -31.0% | [-35.2%, -27.0%] | 25 | 226 | <0.001 |
| FiQA | CSQE | Raw | 648 | -2.3% | [-4.6%, 0.0%] | 23 | 38 | 0.072 |
| FiQA | SCOPE-exemplar | Raw | 648 | -28.7% | [-33.3%, -24.2%] | 41 | 227 | <0.001 |
| FiQA | SCOPE-exemplar | HyDE | 648 | 5.2% | [1.4%, 9.1%] | 97 | 63 | 0.009 |
| FiQA | SCOPE-exemplar | SCOPE | 648 | 2.3% | [-0.8%, 5.4%] | 62 | 47 | 0.180 |
| FiQA | SCOPE-exemplar | CSQE | 648 | -26.4% | [-30.7%, -21.9%] | 42 | 213 | <0.001 |
| TREC-COVID | HyDE | Raw | 50 | -28.0% | [-42.0%, -14.0%] | 1 | 15 | <0.001 |
| TREC-COVID | SCOPE | Raw | 50 | -2.0% | [-10.0%, 4.0%] | 1 | 2 | 1.000 |
| TREC-COVID | CSQE | Raw | 50 | 0.0% | [0.0%, 0.0%] | 0 | 0 | 1.000 |
| TREC-COVID | SCOPE-exemplar | Raw | 50 | -6.0% | [-14.0%, 2.0%] | 1 | 4 | 0.375 |
| TREC-COVID | SCOPE-exemplar | HyDE | 50 | 22.0% | [8.0%, 36.0%] | 14 | 3 | 0.013 |
| TREC-COVID | SCOPE-exemplar | SCOPE | 50 | -4.0% | [-12.0%, 4.0%] | 1 | 3 | 0.625 |
| TREC-COVID | SCOPE-exemplar | CSQE | 50 | -6.0% | [-14.0%, 2.0%] | 1 | 4 | 0.375 |
| SciDocs | HyDE | Raw | 989 | -23.6% | [-27.1%, -20.3%] | 58 | 291 | <0.001 |
| SciDocs | SCOPE | Raw | 989 | -2.1% | [-4.9%, 0.4%] | 84 | 105 | 0.146 |
| SciDocs | CSQE | Raw | 989 | -1.4% | [-3.5%, 0.8%] | 56 | 70 | 0.247 |
| SciDocs | SCOPE-exemplar | Raw | 989 | -5.9% | [-8.8%, -2.9%] | 81 | 139 | <0.001 |
| SciDocs | SCOPE-exemplar | HyDE | 989 | 17.7% | [14.2%, 21.2%] | 246 | 71 | <0.001 |
| SciDocs | SCOPE-exemplar | SCOPE | 989 | -3.7% | [-6.3%, -1.2%] | 66 | 103 | 0.005 |
| SciDocs | SCOPE-exemplar | CSQE | 989 | -4.4% | [-7.6%, -1.5%] | 95 | 139 | 0.005 |
| Pooled | HyDE | Raw | 2310 | -31.3% | [-33.5%, -29.0%] | 115 | 839 | <0.001 |
| Pooled | SCOPE | Raw | 2310 | -12.4% | [-14.3%, -10.5%] | 142 | 428 | <0.001 |
| Pooled | CSQE | Raw | 2310 | -2.8% | [-4.1%, -1.4%] | 93 | 157 | <0.001 |
| Pooled | SCOPE-exemplar | Raw | 2310 | -13.2% | [-15.2%, -11.1%] | 153 | 459 | <0.001 |
| Pooled | SCOPE-exemplar | HyDE | 2310 | 18.1% | [15.9%, 20.3%] | 588 | 170 | <0.001 |
| Pooled | SCOPE-exemplar | SCOPE | 2310 | -0.9% | [-2.4%, 0.7%] | 179 | 199 | 0.328 |
| Pooled | SCOPE-exemplar | CSQE | 2310 | -10.5% | [-12.5%, -8.4%] | 187 | 429 | <0.001 |

## Reading

- CSQE is the strongest expansion-style arm in Phase A: 59.4% pooled Hit@5, only -2.8% behind raw.
- SCOPE-exemplar does not rescue strong-query BEIR: pooled Hit@5 is 48.9%, slightly below vanilla SCOPE at 49.8%.
- The exemplar arm does improve over HyDE by 18.1% pooled Hit@5, but that is not the relevant bar because vanilla SCOPE already does most of that recovery.
- The snap-answer component is not adding over corpus steering here. CSQE, which uses real raw top-k corpus snippets without a snap answer, is substantially stronger than SCOPE-exemplar on the pooled retrieval metric.
- The current evidence favors a selective/gated expansion story rather than more ungated generation. Phase B selection may still be interesting later, but Phase A does not justify spending more model calls under the stated gate.

## Sources

- `caches/exemplars/beir_orthogonal3_exemplars_2026-05-26.json`
- `docs/generated/beir_orthogonal3_exemplars_2026-05-26.md`
- `caches/retrieval/full/beir_scifact_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_csqe_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_csqe.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_exemplar_orthogonal3.jsonl`
- `/tmp/beir_phase1_verification_2026-05-26_points.jsonl` for reused raw/HyDE/SCOPE gold-affinity CE scores.

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CROSS_ENCODER_DEVICE=cuda \
uv run python scripts/analyze_exemplar_scope_select.py \
  --output docs/generated/exemplar_scope_select_2026-05-26.md
```
