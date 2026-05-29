# MuSiQue Cross-Domain Regime Test - 2026-05-28

This is a retrieval-only MuSiQue validation run over the per-question candidate paragraphs. Retrieval uses `Alibaba-NLP/gte-large-en-v1.5` dense scoring inside each question's candidate set followed by `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking. No answer calls and no `paper/` edits were made.

## Verdict

- **H-cross-domain-help-side: mixed/killed.** HyDE/SCOPE improve bridge@5 by +14.6pp/+16.1pp, but reduce full-support@5 by -2.2pp/-3.1pp versus raw.
- **H-csqe-collapse-cross-domain: killed.** CSQE is not raw-like on the multi-hop metrics: full-support@5 changes by +22.6pp and bridge@5 by +33.4pp versus raw.
- **H-regime-placement: mixed/killed.** MuSiQue raw Hit@5 is 97.4%, placing it in the moderate-to-saturated raw regime; pool was expected to help preserve raw candidates; pool Hit@5 is 97.4% and SCOPE Hit@5 is 97.9%.
- Optional q500 answer EM was not run in this phase; the task gate was retrieval-regime evidence first.

## Source Files

| Role | Path |
|---|---|
| Dataset questions | `datasets/musique/questions.csv` |
| Per-question paragraphs | `datasets/musique/passages.csv` |
| Raw question retrieval | `caches/retrieval/full/musique_qfull_seed42_raw_question_k10.jsonl` |
| HyDE generation | `caches/generation/full/musique_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` |
| HyDE retrieval | `caches/retrieval/full/musique_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` |
| SCOPE / snap_hyre generation | `caches/generation/full/musique_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl` |
| SCOPE / snap_hyre retrieval | `caches/retrieval/full/musique_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` |
| CSQE generation | `caches/generation/full/musique_qfull_seed42_or-gemma4-26b_csqe.jsonl` |
| CSQE retrieval | `caches/retrieval/full/musique_qfull_seed42_or-gemma4-26b_csqe_k10.jsonl` |
| raw∪SCOPE pool retrieval | `caches/retrieval/full/musique_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl` |
| Row-level points | `docs/generated/musique_cross_domain_regime_2026-05-28_points.jsonl` |

## Cache Health

| Cache | Rows | Duplicates | Errors | Missing passage | Parse bad | Answer artifact | Short retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw question retrieval | 2417 | 0 | 0 | -- | -- | -- | 0 |
| HyDE generation | 2417 | 0 | 0 | 0 | 0 | 0 | -- |
| HyDE retrieval | 2417 | 0 | 0 | -- | -- | -- | 0 |
| SCOPE / snap_hyre generation | 2417 | 0 | 0 | 0 | 0 | 0 | -- |
| SCOPE / snap_hyre retrieval | 2417 | 0 | 0 | -- | -- | -- | 0 |
| CSQE generation | 2417 | 0 | 0 | 0 | 0 | 0 | -- |
| CSQE retrieval | 2417 | 0 | 0 | -- | -- | -- | 0 |
| raw∪SCOPE pool retrieval | 2417 | 0 | 0 | -- | -- | -- | 0 |

## Retrieval Metrics

Bridge paragraph = the gold paragraph with the lowest raw-query CE score within the question's gold support set. Full-support requires every gold paragraph for the question to be present in the top-k.

| Method | Hit@5 | Full-support@2 | Full-support@5 | Bridge@2 | Bridge@5 | Mean gold-affinity delta |
|---|---:|---:|---:|---:|---:|---:|
| Raw question | 97.4% | 11.6% | 30.5% | 11.6% | 30.5% | 0.000 |
| HyDE | 98.0% | 9.3% | 28.2% | 23.1% | 45.0% | 1.885 |
| SCOPE / snap_hyre | 97.9% | 9.6% | 27.4% | 25.2% | 46.6% | 1.468 |
| CSQE | 98.1% | 24.6% | 53.0% | 34.5% | 63.8% | 5.794 |
| raw∪SCOPE pool | 97.4% | 11.6% | 30.5% | 11.6% | 30.5% | -0.001 |

## Expansion vs Raw

| Method | Metric | Delta | Help rows | Hurt rows | RI |
|---|---|---:|---:|---:|---:|
| HyDE | hit@5 | +0.5pp | 47 | 34 | 0.005 |
| HyDE | full@2 | -2.3pp | 119 | 175 | -0.023 |
| HyDE | full@5 | -2.2pp | 297 | 351 | -0.022 |
| HyDE | bridge@2 | +11.5pp | 401 | 124 | 0.115 |
| HyDE | bridge@5 | +14.6pp | 586 | 234 | 0.146 |
| SCOPE / snap_hyre | hit@5 | +0.5pp | 52 | 41 | 0.005 |
| SCOPE / snap_hyre | full@2 | -2.0pp | 125 | 174 | -0.020 |
| SCOPE / snap_hyre | full@5 | -3.1pp | 290 | 364 | -0.031 |
| SCOPE / snap_hyre | bridge@2 | +13.6pp | 449 | 120 | 0.136 |
| SCOPE / snap_hyre | bridge@5 | +16.1pp | 618 | 228 | 0.161 |
| CSQE | hit@5 | +0.7pp | 30 | 14 | 0.007 |
| CSQE | full@2 | +12.9pp | 357 | 44 | 0.129 |
| CSQE | full@5 | +22.6pp | 670 | 124 | 0.226 |
| CSQE | bridge@2 | +22.8pp | 583 | 31 | 0.228 |
| CSQE | bridge@5 | +33.4pp | 875 | 68 | 0.334 |
| raw∪SCOPE pool | hit@5 | +0.0pp | 0 | 0 | 0.000 |
| raw∪SCOPE pool | full@2 | +0.0pp | 0 | 0 | 0.000 |
| raw∪SCOPE pool | full@5 | +0.0pp | 0 | 0 | 0.000 |
| raw∪SCOPE pool | bridge@2 | +0.0pp | 0 | 0 | 0.000 |
| raw∪SCOPE pool | bridge@5 | +0.0pp | 0 | 0 | 0.000 |

## Regime Gradient Context

| Dataset/regime | Raw Hit@5 | SCOPE Hit@5 | raw-SCOPE pool Hit@5 | Note |
|---|---:|---:|---:|---|
| BarExamQA | 1.4% | 12.0% | 3.9% | extreme weak legal query; pool fails versus SCOPE |
| CaseHOLD | 17.9% | 45.0% | 19.2% | intermediate weak legal query; pool fails versus SCOPE |
| MuSiQue | 97.4% | 97.9% | 97.4% | current cross-domain weak-query test |
| HousingQA state-filtered | 36.8% | 38.0% | 41.1% | stronger raw state anchors; pool helps |
| BEIR pooled | 62.2% | 49.8% | 65.9% | strong raw queries; pool preserves raw candidates |

## Recommendation

- Do not spend q500 answer budget on the standard MuSiQue SCOPE/HyDE setup unless a later prompt or selector improves full-support without losing the bridge-recall gain.
- Use the row-level points file to inspect where expansion helps bridge recall versus where it drops a required support paragraph.
