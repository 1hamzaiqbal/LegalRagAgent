# Choice-Aware Retrieval Probe

Status: probe-only. These are N=20 retrieval-exposure diagnostics for Gemma 4
26B, not promoted paper results. They were run to separate choice exposure,
generated legal-query style, and snap-conditioned reasoning before choosing
which variants deserve larger q50/q100 checks.

Follow-up: the q50 SCALR and CaseHOLD probe is recorded in
`docs/choice_aware_retrieval_q50_2026-05-14.md`.

## Methods

| Method | Retrieval query definition |
|---|---|
| `rag_simple` | Raw stem/context only. No answer choices in retrieval. |
| `rag_choice_simple` | Raw stem/context plus each unlabeled choice text as separate retrieval queries. |
| `rag_hyde_blind` | One generated legal-reference passage from stem/context only. |
| `rag_hyde_choice` | One generated legal-reference passage from stem/context plus unlabeled choices. |
| `snap_hyre` | One answer-first snap call plus one neutral HyRE passage; retrieval uses the passage only. |
| `snap_hyre_anchor` | Same exact `snap_hyre` generated passage plus raw/candidate anchor at retrieval time. |
| `multi_hyde_diverse` | Three diverse generated legal-reference passages plus raw/candidate anchor. |
| `snap_choice_hyre` | Snap predicts primary and strongest alternative, then emits predicted/alternative/neutral retrieval passages. |

The SCALR tuned pass used `rag_hyde` before the explicit split landed; in this
note, treat that row as `rag_hyde_choice` because the intermediate formatter
included unlabeled candidate holdings.

## Source Logs

| Dataset | Detail log | Summary |
|---|---|---|
| LegalBench-SCALR | `logs/choice_aware_retrieval_legalbench_scalr_or-gemma4-26b_q20_k10_tuned.jsonl` | `docs/generated/choice_aware_retrieval_legalbench_scalr_or-gemma4-26b_q20_tuned.md` |
| CaseHOLD | `logs/choice_aware_retrieval_casehold_or-gemma4-26b_q20_k10_explicit.jsonl` | `docs/generated/choice_aware_retrieval_casehold_or-gemma4-26b_q20_explicit.md` |
| BarExamQA | `logs/choice_aware_retrieval_barexam_or-gemma4-26b_q20_k10_combined.jsonl` | `docs/generated/choice_aware_retrieval_barexam_or-gemma4-26b_q20_combined.md` |

The first BarExam `snap_choice_hyre` attempt stopped on an OpenRouter upstream
401 from the DekaLLM connector. `NO_SILENT_FALLBACK=1` blocked the row instead
of rerouting. The completed `snap_choice_hyre` BarExam rows were rerun with
`OPENROUTER_PROVIDER_IGNORE=dekallm`; OpenRouter fallback remained disabled.

Qrel alignment audit passed on all three q20 samples:

| Dataset | Collection | Gold-id existence |
|---|---|---:|
| BarExamQA | `legal_passages` | 20/20 |
| CaseHOLD | `casehold_holdings` | 20/20 |
| LegalBench-SCALR | `legalbench_scalr_holdings` | 20/20 |

## Health

| Dataset | Rows | Errors | Parse failures | Answer-artifact rows | Empty retrieval rows |
|---|---:|---:|---:|---:|---:|
| LegalBench-SCALR | 140 | 0 | 0 | 0 | 0 |
| CaseHOLD | 160 | 0 | 0 | 0 | 0 |
| BarExamQA combined | 160 | 0 | 0 | 0 | 0 |

## Hit@k Summary

| Dataset | Method | Hit@1 | Hit@5 | MRR@5 | Hit@10 |
|---|---|---:|---:|---:|---:|
| SCALR | `rag_simple` | 0.35 | 0.65 | 0.4625 | 0.70 |
| SCALR | `rag_choice_simple` | 0.30 | 0.60 | 0.4100 | 0.65 |
| SCALR | `rag_hyde_choice` | 0.65 | 0.80 | 0.7100 | 0.80 |
| SCALR | `snap_hyre` | 0.60 | 0.70 | 0.6200 | 0.90 |
| SCALR | `multi_hyde_diverse` | 0.60 | 0.75 | 0.6600 | 0.80 |
| SCALR | `snap_choice_hyre` | 0.50 | 0.70 | 0.5833 | 0.75 |
| CaseHOLD | `rag_simple` | 0.10 | 0.20 | 0.1250 | 0.20 |
| CaseHOLD | `rag_choice_simple` | 0.00 | 0.20 | 0.0517 | 0.20 |
| CaseHOLD | `rag_hyde_blind` | 0.10 | 0.30 | 0.1767 | 0.30 |
| CaseHOLD | `rag_hyde_choice` | 0.40 | 0.55 | 0.4667 | 0.70 |
| CaseHOLD | `snap_hyre` | 0.35 | 0.55 | 0.4375 | 0.65 |
| CaseHOLD | `snap_hyre_anchor` | 0.35 | 0.55 | 0.4292 | 0.65 |
| CaseHOLD | `multi_hyde_diverse` | 0.30 | 0.55 | 0.3892 | 0.65 |
| CaseHOLD | `snap_choice_hyre` | 0.35 | 0.60 | 0.4267 | 0.60 |
| BarExamQA | `rag_simple` | 0.00 | 0.00 | 0.0000 | 0.00 |
| BarExamQA | `rag_choice_simple` | 0.00 | 0.00 | 0.0000 | 0.00 |
| BarExamQA | `rag_hyde_blind` | 0.00 | 0.05 | 0.0100 | 0.05 |
| BarExamQA | `rag_hyde_choice` | 0.05 | 0.05 | 0.0500 | 0.05 |
| BarExamQA | `snap_hyre` | 0.05 | 0.10 | 0.0750 | 0.10 |
| BarExamQA | `snap_hyre_anchor` | 0.05 | 0.10 | 0.0750 | 0.10 |
| BarExamQA | `multi_hyde_diverse` | 0.00 | 0.00 | 0.0000 | 0.00 |
| BarExamQA | `snap_choice_hyre` | 0.00 | 0.10 | 0.0500 | 0.10 |

## Interpretation

- Choice text by itself is not a safe retrieval shortcut. `rag_choice_simple`
  did not beat `rag_simple` on any of these q20 slices and hurt SCALR.
- Generated legal-reference query style is the main retrieval lift. On CaseHOLD,
  `rag_hyde_choice` rose from raw 0.20 Hit@5 to 0.55; on SCALR the analogous
  choice-aware HyDE row reached 0.80.
- Snap-HyRE remains interesting because it has distinct row-level wins and the
  best SCALR Hit@10, but it did not dominate `rag_hyde_choice` at Hit@5.
- `snap_choice_hyre` is promising as an ablation, not yet a canonical method:
  it led CaseHOLD Hit@5 at 0.60 and tied BarExam Snap-HyRE at 0.10, but lagged
  SCALR and costs a more complicated prompt.
- `snap_hyre_anchor` did not improve Hit@5 on CaseHOLD or BarExam when using
  the same generated snap passage. Do not promote it unless larger top-k checks
  show a consistent MRR/Hit@10 advantage.
- `multi_hyde_diverse` is a strong holding-task ablation but failed on BarExam
  in this slice; it should stay in analysis unless it generalizes at q50/q100.

## Recommended Next Step

Completed on 2026-05-14. The q50 retrieval-only follow-up covered CaseHOLD and
SCALR with:

`rag_simple`, `rag_hyde_blind`, `rag_hyde_choice`, `snap_hyre`,
`multi_hyde_diverse`, and `snap_choice_hyre`.

Do not add all variants to the downstream comprehensive answer grid yet. The
q50 follow-up keeps the fixed ladder recommendation small unless a downstream
q20/q50 answer slice justifies adding one extra ablation row.
