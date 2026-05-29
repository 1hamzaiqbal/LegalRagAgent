# CaseHOLD raw+SCOPE Pool Test - 2026-05-28

This test pools existing CaseHOLD raw-question top-10 and Llama-70B SCOPE top-10 retrieval caches, deduplicates by document id, and reranks the union with `cross-encoder/ms-marco-MiniLM-L-6-v2` to top-5. No LLM generation or answer calls were run, and no files under `paper/` were edited.

## Verdict

- **H-pool-intermediate-weak: killed.** CaseHOLD raw+SCOPE pool collapses below 30% Hit@5, so pooling fails even in the intermediate-weak regime.
- Regime read: **binary**. BarExam remains the extreme-weak failure point, CaseHOLD tests the intermediate-weak band, and Housing/BEIR show pool gains once raw retrieval has enough useful candidates for CE reranking.

## Regime Gradient

| Regime | N | Raw Hit@5 | SCOPE Hit@5 | raw+SCOPE pool Hit@5 | Pool hits | RI vs raw | Help/Hurt vs raw | RI vs SCOPE | Help/Hurt vs SCOPE | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BarExamQA | 1192 | 1.4% | 12.0% | 3.9% | 47 | 0.025 | 35/5 | -0.081 | 23/119 | Gemma-26B SCOPE/pool |
| CaseHOLD | 3600 | 17.9% | 45.0% | 19.2% | 692 | 0.013 | 95/49 | -0.258 | 213/1140 | Llama-70B SCOPE; no new generation |
| HousingQA state-filtered | 6832 | 36.8% | 38.0% | 41.1% | 2809 | 0.043 | 556/259 | 0.031 | 961/748 | Gemma-26B SCOPE/pool |
| BEIR pooled | 2310 | 62.2% | 49.8% | 65.9% | 1522 | 0.037 | 131/45 | 0.161 | 472/100 | Gemma-26B SCOPE/pool |

## Reading

- CaseHOLD raw retrieval is 17.9%, far above BarExam's 1.4% but far below Housing and BEIR. This is the intended intermediate-weak point.
- The CaseHOLD pool reaches 19.2% Hit@5. It helps 95 rows over raw and hurts 49 raw-hit rows, for RI=0.013.
- Relative to SCOPE, the pool helps 213 rows and hurts 1140, giving RI=-0.258. This is the direct preservation check.
- Generator mismatch caveat: CaseHOLD SCOPE is the existing `groq-llama70b` cache, while the BarExam, Housing, and BEIR pool rows in the comparison table use `or-gemma4-26b`. The pooling/reranking mechanism is generator-agnostic, but this is not a strict generator-controlled comparison.

## Sources

- CaseHOLD raw cache: `caches/retrieval/full/casehold_qfull_seed42_raw_question_k10.jsonl`
- CaseHOLD SCOPE cache: `caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`
- CaseHOLD pool cache: `caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_raw_scope_pool_k5.jsonl`
- CaseHOLD row-level points: `docs/generated/casehold_pool_test_2026-05-28_points.jsonl`
- Prior BarExam/Housing/BEIR points: `docs/generated/3scope_raw_pool_2026-05-28_points.jsonl`
