# Adaptive HyRE CaseHOLD Option Reranker N=50 - 2026-05-10

## Question

Can per-candidate retrieval bundles improve CaseHOLD beyond a candidate-first
final prompt?

The prior `adaptive_snap_hyre_candidate_verifier` probe improved CaseHOLD
relative to the cached frontier, but did not beat the rag_simple N=50 slice.
This probe tests whether the missing ingredient is option-conditioned retrieval
rather than a longer final-answer prompt.

## Method

Mode: `adaptive_snap_hyre_option_reranker`

For CaseHOLD only:

- replay the cached snap/HyRE object;
- retrieve a small general evidence bundle from HyRE plus the formatted task;
- retrieve one evidence item per displayed candidate holding;
- ask one final LLM call to compare the candidate evidence bundles and select
  the holding best supported by the citing context.

This is still a one-final-call cached replay path. It spends extra retrieval
work, not extra LLM calls.

## Cluster Evidence

Job: `67400`

Detail log:
`/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_reranker_or-gemma4-26b_20260510_1201_casehold_casehold-option-reranker-cached-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_option_reranker_detail.jsonl`

Summary:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_casehold_option_reranker_cached_n50.md`

Health:

- rows: 50
- average LLM calls: 1.00
- empty retrieval: 0
- missing predictions: 0
- parse failures: 0
- audit: PASS

## Result

| Method | N | Accuracy | Gold Retrieved | Calls | Health |
|---|---:|---:|---:|---:|---|
| `adaptive_snap_hyre_frontier` | 50 | 70.0% | 14/50 | 1.00 | PASS |
| `adaptive_snap_hyre_stability` | 50 | 72.0% | 16/50 | 4.12 | PASS |
| `adaptive_snap_hyre_candidate_verifier` | 50 | 74.0% | 14/50 | 1.00 | PASS |
| `adaptive_snap_hyre_option_reranker` | 50 | 74.0% | 20/50 | 1.00 | PASS |
| `rag_simple` | 50 slice | 76.0% | source-paired slice | 1.00 | control |

Paired comparisons from the postprocess summary:

| Comparison | Delta | b/c | p |
|---|---:|---:|---:|
| frontier -> option reranker | +4.0pp | 4/2 | 0.6875 |
| rag_snap_hyde_2call -> option reranker | +8.0pp | 5/1 | 0.2188 |
| rag_simple -> option reranker | -2.0pp | 4/5 | 1 |

## Interpretation

The option reranker improves gold retrieval from 14/50 to 20/50 relative to the
candidate verifier, but accuracy stays at 74.0%. That means the per-candidate
retrieval bundle is doing something real, but the final answer conversion still
does not reliably exploit the extra retrieved gold.

Do not scale this exact method to N=200 yet. The next CaseHOLD step should be a
more explicit selector over the candidate evidence bundles, or a non-generative
reranking score that chooses among candidates before the final answer prompt.

## Decision

- Keep the option-reranker retrieval structure as useful instrumentation.
- Do not treat this exact final prompt as the CaseHOLD solution.
- Do not run the SCALR version; the candidate-first verifier already failed
  there and produced pathological long generations.

