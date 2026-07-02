---
title: Raw∪SCOPE Pooling — regime-dependent fusion
type: result
tags: [pooling, fusion, regime, retrieval]
created: 2026-07-02
updated: 2026-07-02
date: 2026-05-25/28
verdict: win on strong/mid regimes; killed on weak
evidence: docs/generated/3scope_raw_pool_2026-05-28.md
---

# Raw∪SCOPE pooling and the no-single-recipe result

**Setup.** Pool unique docs from raw-query top-10 and SCOPE top-10 (optionally
3 independent SCOPE samples), CE-rerank to top-5. Full-N on 5 BEIR + BarExamQA
+ HousingQA(state-filtered); arms: raw, HyDE, SCOPE, [[csqe]], raw∪SCOPE,
3SCOPE+raw.

**Numbers** (Hit@5):
- **Strong/mid regimes — pooling wins**: BEIR pooled raw 62.2 → raw∪SCOPE
  **65.9** (+3.7pp, help/hurt 131/45); positive on all 5 sets; Housing
  state-filtered 36.8 → **41.1** (+4.3pp). Beats CSQE everywhere on BEIR.
- **Weak regime — pooling destroys the gain**: BarExamQA SCOPE-alone **12.0**
  vs raw∪SCOPE 3.9 vs CSQE 2.0 (raw 1.4). The CE reranker buries
  generated-query candidates when plausible-looking raw candidates exist
  (CE-buries-gold finding, [docs/generated/raw_scope_pooling_ce_separability_2026-05-25.md](../../docs/generated/raw_scope_pooling_ce_separability_2026-05-25.md)).
- **Diversity adds nothing**: 3SCOPE+raw ≈ raw∪SCOPE everywhere (65.0 vs
  65.9 BEIR pooled) — killed.
- Mid-regime (raw 20–30% band, credibility E): pooling gives small strict
  wins (SciDocs Hit@1 22.2→23.2; Housing Hit@2 23.9→25.9).

**Verdict.** No single recipe covers both regimes → [[regime-routing]] is
forced, not optional. Also corrects an earlier over-call that pooling was a
universal fix.

**CSQE crossover** (same table + [docs/generated/casehold_csqe_collapse_2026-05-28.md](../../docs/generated/casehold_csqe_collapse_2026-05-28.md)):
corpus-steered expansion is the robust choice on strong-query corpora (BEIR
−2.8pp only) but **collapses on weak-query legal** (BarExam 2.0%) because
initial retrieval surfaces nothing real to steer with — generative expansion's
niche *is* the weak-query regime, mechanistically confirmed.

## Links
[[regime-routing]] · [[weak-vs-strong-query-regime]] · [[csqe]] ·
[[query-drift]] · [[scope]] ·
[pool report](../../docs/generated/3scope_raw_pool_2026-05-28.md) ·
[CE separability](../../docs/generated/raw_scope_pooling_ce_separability_2026-05-25.md)
