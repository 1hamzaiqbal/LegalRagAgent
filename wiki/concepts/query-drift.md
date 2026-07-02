---
title: Query Drift
type: concept
tags: [expansion-failure, retrieval, prf]
created: 2026-07-02
updated: 2026-07-02
status: draft
---

# Query drift

**Definition.** The classic pseudo-relevance-feedback failure mode: an
expanded/rewritten query moves *away* from the user's information need, so
expansion breaks retrieval that the raw query already handled. In the
generative-expansion era: the LLM's hypothetical document imposes a wrong or
over-specific framing, and the embedding follows it away from gold.

**Why it matters to us.** It is the established name (criticism from the
grounding doc: stop calling it a discovery) for what SCOPE does on strong
queries — HousingQA answers −3pp (C10 in [[icml-ai4law-2026-rejection]]),
strong-query BEIR Hit@5 HyDE −31% / SCOPE −12%. Macro version owned by
Weller'24 ([[weller-drift]]); classical selective-expansion response owned by
Amati'04/Cronen-Townsend'04.

**Our additions**:
1. *Per-query localization*: drift severity is predicted by the geometric
   affinity margin, not by hallucination/factuality
   ([[geometry-vs-factuality]]).
2. *Method-variance finding*: SCOPE's snap-conditioning drifts far less than
   HyDE on strong queries (−12% vs −31% — the one surviving snap benefit,
   a robustness claim). Mechanistic hypothesis: committing to a legal frame
   first constrains the pseudo-doc's topic support.
3. *Mitigation*: [[regime-routing]] (don't expand strong queries; pool
   raw∪SCOPE candidates instead — recovers +2–3pp).

**Open questions.** Is SCOPE's drift-robustness itself significance-tested and
replicated beyond BEIR Phase 1? (needs verification pass); can drift be
detected *post-hoc, pre-answer* from retrieval-set geometry cheaply enough to
fall back to the raw query per-item?

## Links
[[weller-drift]] · [[weak-vs-strong-query-regime]] · [[regime-routing]] ·
[[geometry-vs-factuality]] · [[qpp]] · [[scope]]
