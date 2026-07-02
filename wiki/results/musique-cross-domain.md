---
title: MuSiQue Cross-Domain — mechanism confirms; pool structure caveat
type: result
tags: [musique, multi-hop, cross-domain]
created: 2026-07-02
updated: 2026-07-02
date: 2026-05-28/29
verdict: win (mechanism) with structural caveat
evidence: docs/generated/musique_cross_domain_regime_2026-05-28.md
---

# MuSiQue cross-domain check

**Setup.** MuSiQue multi-hop QA, per-question ~20-paragraph pool (not a global
corpus); raw vs SCOPE vs HyDE vs CSQE generation caches; bridge-evidence
recall.

**Findings.**
- SCOPE/HyDE **bridge-recall +15–16pp** over raw — clean cross-domain
  confirmation that generated queries surface the vocabulary-distant hop.
- CSQE "helps dramatically" here — but only because the per-question pool
  *guarantees* real relevant text to steer with; the global-corpus collapse
  premise ([[pooling-regime]] CSQE finding) doesn't apply. **Not** a
  refutation of the weak-query CSQE collapse.
- Structural lesson: per-question-pool benchmarks are a *different object*
  from global-corpus retrieval; conclusions don't transfer either way.
  Global-corpus weak-query evidence remains legal(+medical)-only — a
  Limitations item, and the reason discovery for more weak-query global
  corpora matters.

## Links
[[weak-vs-strong-query-regime]] · [[csqe]] · [[scope]] ·
[report](../../docs/generated/musique_cross_domain_regime_2026-05-28.md)
