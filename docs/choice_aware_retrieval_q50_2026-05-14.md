# Choice-Aware Retrieval q50 Probe - 2026-05-14

Status: probe-only. These are q50 retrieval-exposure diagnostics for
`or-gemma4-26b` on the two holding-style datasets where choice awareness was
most plausible after the q20 pass. They are not paper-facing result claims yet.

## Source Logs

| Dataset | Combined detail log | Combined summary |
|---|---|---|
| LegalBench-SCALR | `logs/choice_aware_retrieval_legalbench_scalr_or-gemma4-26b_q50_k10_combined.jsonl` | `docs/generated/choice_aware_retrieval_legalbench_scalr_or-gemma4-26b_q50_combined.md` |
| CaseHOLD | `logs/choice_aware_retrieval_casehold_or-gemma4-26b_q50_k10_combined.jsonl` | `docs/generated/choice_aware_retrieval_casehold_or-gemma4-26b_q50_combined.md` |

Chunked source logs are preserved next to the combined files. SCALR needed a
Snap-HyRE continuation after an upstream OpenRouter rate-limit error. CaseHOLD
needed a `multi_hyde_diverse` continuation after one malformed generated block.
Both failures stopped under `NO_SILENT_FALLBACK=1`; no failed rows were silently
accepted into the combined logs.

OpenRouter route policy for these runs: model id fixed to
`google/gemma-4-26b-a4b-it`; `OPENROUTER_PROVIDER_IGNORE=dekallm,deepinfra`;
OpenRouter model fallback disabled. Same-model provider routing is acceptable
for future retries if it is explicit and logged; the prohibited case is any
silent change of model id, method, cache, or prompt.

## Health Gates

| Dataset | Rows | Expected rows | Errors | Parse failures | Fallback rows | Answer-artifact rows | Empty retrieval rows | Think-tag rows | Qrel alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LegalBench-SCALR | 300 | 6 modes x 50 | 0 | 0 | 0 | 0 | 0 | 0 | 50/50 unique gold ids found |
| CaseHOLD | 300 | 6 modes x 50 | 0 | 0 | 0 | 0 | 0 | 0 | 50/50 unique gold ids found |

The audited modes were:

`rag_simple`, `rag_hyde_blind`, `rag_hyde_choice`, `snap_hyre`,
`multi_hyde_diverse`, and `snap_choice_hyre`.

## Hit@k Summary

| Dataset | Method | Hit@1 | Hit@5 | MRR@5 | Hit@10 |
|---|---|---:|---:|---:|---:|
| SCALR | `rag_simple` | 0.26 | 0.56 | 0.3713 | 0.64 |
| SCALR | `rag_hyde_blind` | 0.38 | 0.60 | 0.4700 | 0.66 |
| SCALR | `rag_hyde_choice` | 0.60 | 0.74 | 0.6557 | 0.76 |
| SCALR | `snap_hyre` | 0.64 | 0.76 | 0.6857 | 0.80 |
| SCALR | `multi_hyde_diverse` | 0.62 | 0.66 | 0.6340 | 0.72 |
| SCALR | `snap_choice_hyre` | 0.56 | 0.68 | 0.6167 | 0.74 |
| CaseHOLD | `rag_simple` | 0.10 | 0.24 | 0.1420 | 0.28 |
| CaseHOLD | `rag_hyde_blind` | 0.16 | 0.28 | 0.2023 | 0.34 |
| CaseHOLD | `rag_hyde_choice` | 0.48 | 0.66 | 0.5550 | 0.72 |
| CaseHOLD | `snap_hyre` | 0.38 | 0.58 | 0.4550 | 0.64 |
| CaseHOLD | `multi_hyde_diverse` | 0.44 | 0.58 | 0.5017 | 0.72 |
| CaseHOLD | `snap_choice_hyre` | 0.20 | 0.46 | 0.2897 | 0.56 |

## Row-Level Disagreement

SCALR:

- `snap_hyre` beat `rag_simple` at Hit@5 on 15 rows and lost on 5.
- `snap_hyre` beat `rag_hyde_choice` at Hit@5 on 2 rows and lost on 1.
- `snap_choice_hyre` beat `snap_hyre` at Hit@5 on 1 row and lost on 5.
- `rag_hyde_choice` beat `rag_hyde_blind` at Hit@5 on 9 rows and lost on 2.

CaseHOLD:

- `snap_hyre` beat `rag_simple` at Hit@5 on 19 rows and lost on 2.
- `snap_hyre` beat `rag_hyde_choice` at Hit@5 on 4 rows and lost on 8.
- `multi_hyde_diverse` beat `rag_hyde_choice` at Hit@5 on 6 rows and lost on 10.
- `snap_choice_hyre` beat `snap_hyre` at Hit@5 on 6 rows and lost on 12.
- `rag_hyde_choice` beat `rag_hyde_blind` at Hit@5 on 20 rows and lost on 1.

## Interpretation

- Snap-HyRE is a real retrieval lift over raw RAG on both q50 slices:
  SCALR Hit@5 0.76 vs 0.56, CaseHOLD Hit@5 0.58 vs 0.24.
- SCALR is the cleanest retrieval-supportive Snap-HyRE result in this probe:
  Snap-HyRE leads all checked methods at Hit@1, Hit@5, MRR@5, and Hit@10.
- CaseHOLD is more choice-sensitive than snap-sensitive. `rag_hyde_choice`
  leads Hit@5 and MRR@5, while Snap-HyRE still beats raw and blind HyDE.
- Choice text helps when the generated retrieval query is allowed to reason over
  candidate holdings, but raw choice exposure alone was not promoted from q20.
- `snap_choice_hyre` should not be promoted as a canonical method from this
  evidence. It lagged Snap-HyRE on both q50 slices.
- `multi_hyde_diverse` is useful as an analysis row. It ties Snap-HyRE on
  CaseHOLD Hit@5 and reaches CaseHOLD Hit@10 0.72, but it underperforms on
  SCALR and had one transient parse failure that strict mode correctly blocked.

## Recommendation

For comprehensive downstream answer runs, keep the main ladder small:

`llm_only`, `rag_simple`, `rag_hyde`, `snap_hyre`, plus golden controls.

For retrieval-only analysis, keep the q50/q100 split between
`rag_hyde_blind` and `rag_hyde_choice` where the dataset has answer choices.
Use this to explain that choice-aware generated retrieval can be a strong
holding-task baseline, while Snap-HyRE remains the fixed method under test.

Do not add `snap_choice_hyre` to the comprehensive three-model answer grid
unless a separate downstream q20/q50 answer slice shows an accuracy gain that
justifies the extra method complexity.
