# Speculative-RAG-Aligned Metrics Report - 2026-04-30

Generated from detail JSONL logs. Metrics are offline and do not call an LLM.

## Speculative-RAG Metric Mapping

| Speculative RAG metric family | What this report computes now | Gap / caveat |
|---|---|---|
| Answer quality | closed-set accuracy, MuSiQue EM/F1, and free-form gold-answer containment when aliases are logged | Containment is only an automatic proxy; legal open-ended rows still need judge/rubric scoring. |
| Efficiency | average, p50, and p95 latency; LLM calls; input/output token use | Local timings mix API latency and harness overhead, so compare only like-for-like runs. |
| Rationale/context compression | generated pseudo-context tokens versus retrieved evidence tokens | This approximates Speculative RAG rationale-vs-document compression; our logs do not yet separate verifier rationale from HyDE/snap artifacts. |
| Drafting | draft count and speculative-score row coverage | Current modes do not log answer drafts or verifier probabilities, so rhoDraft/rhoSelf-contain/rhoSelf-reflect are not computable yet. |
| Retrieval diagnostics | gold-hit rate, retrieval row rate, empty retrieval, evidence docs/tokens | CaseHOLD gold-hit instrumentation is known untrustworthy in current logs. |

## Run Matrix

| Label | Dataset | Mode | N | Acc | EM | F1 | Contains gold | Gold hit | Evid docs/q | Evid tok/q | Gen ctx tok/q | Gen/Evid | Calls/q | Lat avg/p95 | In tok/q | Out tok/q | Drafts/q | Spec score rows |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| musique_rag | musique | rag_simple | 200 | 27.5% | 27.5% | 36.9% | 35.0% | 84.0% | 5.0 | 501 | 0 | 0.00 | 1.00 | 0.63/2.00 | 833 | 84 | 0.0 | 0.0% |
| musique_rag_top1 | musique | rag_simple | 200 | 13.0% | 13.0% | 18.1% | 19.0% | 47.0% | 1.0 | 102 | 0 | 0.00 | 1.00 | 0.40/1.50 | 256 | 74 | 0.0 | 0.0% |
| musique_2call | musique | rag_snap_hyde_2call | 200 | 37.0% | 37.0% | 48.2% | 48.0% | 86.5% | 5.0 | 536 | 422 | 0.79 | 2.00 | 1.20/3.30 | 1135 | 249 | 0.0 | 0.0% |
| musique_mhd | musique | multi_hyde_diverse | 200 | 35.5% | 35.5% | 45.6% | 47.0% | 84.0% | 5.0 | 563 | 586 | 1.04 | 2.00 | 5.07/4.30 | 1033 | 417 | 0.0 | 0.0% |
| musique_iter | musique | iterative_planning_table | 200 | 36.0% | 36.0% | 45.4% | 53.0% | 92.0% | 6.0 | 12 | 86 | 7.22 | 6.76 | 3.17/4.60 | 2507 | 422 | 0.0 | 0.0% |
| barexam_rag_top5 | barexam | rag_simple | 200 | 82.5% | - | - | - | 2.5% | 5.0 | 470 | 0 | 0.00 | 1.00 | 14.81/32.31 | 989 | 620 | 0.0 | 0.0% |
| barexam_rag_top1 | barexam | rag_simple | 200 | 83.0% | - | - | - | 0.5% | 1.0 | 97 | 0 | 0.00 | 1.00 | 22.25/57.11 | 469 | 633 | 0.0 | 0.0% |
| barexam_2call | barexam | rag_snap_hyde_2call | 200 | 85.5% | - | - | - | 9.0% | 5.0 | 577 | 1340 | 2.32 | 2.00 | 58.32/170.51 | 1625 | 1221 | 0.0 | 0.0% |
| casehold_rag | casehold | rag_simple | 200 | 72.0% | - | - | - | 0.0% | 5.0 | 147 | 0 | 0.00 | 1.00 | 1.53/2.30 | 672 | 438 | 0.0 | 0.0% |
| casehold_rag_top1 | casehold | rag_simple | 200 | 70.5% | - | - | - | 0.0% | 1.0 | 31 | 0 | 0.00 | 1.00 | 1.59/2.30 | 521 | 436 | 0.0 | 0.0% |
| casehold_2call | casehold | rag_snap_hyde_2call | 200 | 69.5% | - | - | - | 0.0% | 5.0 | 151 | 710 | 4.69 | 2.00 | 2.59/3.50 | 1232 | 700 | 0.0 | 0.0% |
| scalr_rag | legalbench_scalr | rag_simple | 200 | 77.0% | - | - | - | 54.0% | 5.0 | 188 | 0 | 0.00 | 1.00 | 1.92/2.60 | 723 | 423 | 0.0 | 0.0% |
| scalr_rag_top1 | legalbench_scalr | rag_simple | 200 | 59.5% | - | - | - | 32.5% | 1.0 | 38 | 0 | 0.00 | 1.00 | 1.89/2.60 | 518 | 430 | 0.0 | 0.0% |
| scalr_rag_top10 | legalbench_scalr | rag_simple | 200 | 77.0% | - | - | - | 63.0% | 10.0 | 375 | 0 | 0.00 | 1.00 | 2.33/3.50 | 979 | 417 | 0.0 | 0.0% |
| scalr_2call | legalbench_scalr | rag_snap_hyde_2call | 200 | 75.0% | - | - | - | 55.0% | 5.0 | 187 | 1047 | 5.59 | 2.00 | 3.46/4.50 | 1337 | 848 | 0.0 | 0.0% |

## Log Provenance

| Label | Detail log | Hypothesis | Caveat |
|---|---|---|---|
| musique_rag | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl` | baseline multi-hop retrieved-context performance | N=200 diagnostic slice; some abstention-like predictions in prior audit. |
| musique_rag_top1 | `logs/eval_rag_simple_groq-llama70b_20260428_0011_detail.jsonl` | top-1 retrieval depth should collapse when multi-hop needs multiple passages | Top-1 proof is clean but has abstention-like predictions noted in audit docs. |
| musique_2call | `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0040_detail.jsonl` | answer-conditioned pseudo-document retrieval rescues query-formulation failures | Three parse fallbacks; N=200 diagnostic slice; full-corpus replicate not landed. |
| musique_mhd | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl` | diverse pseudo-document queries rescue a subset of multi-hop retrieval failures | N=200 paper-headline candidate, but Gemma 27B N=200 is null and full-corpus replicate is pending. |
| musique_iter | `logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl` | iterative evidence gathering rescues more retrieval misses but may introduce reasoning drift | McNemar p=0.0533; more gold evidence sometimes still harms baseline-correct rows. |
| barexam_rag_top5 | `logs/eval_rag_simple_or-gemma4-26b_20260428_0231_detail.jsonl` | legal MC baseline should be weakly sensitive to retrieval depth | Cluster summary row was reconstructed from landed detail log; use full-N BarExam claims from signoff for paper-grade statements. |
| barexam_rag_top1 | `logs/eval_rag_simple_or-gemma4-26b_20260428_0138_detail.jsonl` | top-1 retrieval should be near-flat if more retrieved documents are not the bottleneck | N=200 diagnostic; cite as depth-flatness probe, not final BarExam performance. |
| barexam_2call | `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260428_1435_detail.jsonl` | answer anchoring may help legal MC despite flat top-k sensitivity | N=200 +3.0pp is not significant; full-N rag_snap_hyde result is stronger but different mode. |
| casehold_rag | `logs/eval_rag_simple_groq-llama70b_20260428_0259_detail.jsonl` | holding-selection legal MC may be candidate-depth insensitive under current harness | Current gold_retrieved is 0/200 and cannot support retrieval-recall claims. |
| casehold_rag_top1 | `logs/eval_rag_simple_groq-llama70b_20260429_2318_detail.jsonl` | top-1 should remain near top-5 if CaseHOLD answer choice is not depth-limited | Gold-option retrieval mapping is missing; treat as answer-level flatness only. |
| casehold_2call | `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0309_detail.jsonl` | pseudo-doc query formation should not help if current harness is candidate-depth insensitive | Two parse fallbacks; gold_retrieved is 0/200 and not meaningful for retrieval-recall claims. |
| scalr_rag | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl` | SCALR should need a small candidate set but saturate before top-10 | SCALR logging was fixed after early smoke rows; use this detail log and later audits only. |
| scalr_rag_top1 | `logs/eval_rag_simple_groq-llama70b_20260429_2159_detail.jsonl` | top-1 should underperform if SCALR requires a small candidate set | Use with SCALR audit; top-1 supports depth sensitivity but not top-10 benefit. |
| scalr_rag_top10 | `logs/eval_rag_simple_groq-llama70b_20260430_0054_detail.jsonl` | top-10 should increase evidence/gold hits but not answer accuracy if top-5 already saturates | Top-10 raises gold-hit rate without net accuracy gain; useful as saturation evidence. |
| scalr_2call | `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_1520_detail.jsonl` | pseudo-doc query formulation should not help once SCALR top-5 candidate set is sufficient | Flat/negative versus top-5 rag; supports saturation rather than query-formulation limit. |

## Immediate Wiring Gaps

- Add explicit `answer_drafts` and `draft_rationales` arrays if we implement a Speculative-RAG arm.
- Store verifier logprob-derived scores only when the backend exposes token logprobs; otherwise log a separate `llm_verifier_vote` field and keep it labeled as a proxy.
- Split generated-context logging into `query_pseudo_context`, `reasoning_trace`, and `verifier_rationale` so compression is not overloaded.
- Repair CaseHOLD gold-option retrieval mapping before interpreting gold-hit or recall numbers.
