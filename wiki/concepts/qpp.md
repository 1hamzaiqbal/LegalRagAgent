---
title: Query Performance Prediction (QPP)
type: concept
tags: [qpp, routing, retrieval, ir]
created: 2026-07-02
updated: 2026-07-02
status: draft
---

# Query Performance Prediction (QPP)

**Definition.** Estimating how well a retrieval system will do on a query
*without relevance labels*. Pre-retrieval predictors (clarity, IDF stats) and
post-retrieval predictors over the score distribution — canonical:
WIG (Zhou & Croft '07), NQC (Shtok '09), SMV (Tao & Wu '14). Modern:
dense/neural QPP is notoriously harder ([[faggioli-qpp]]).

**Why it matters to us.** Everything we called "no-gold retrieval confidence"
(top-1 CE score, score spread/entropy, bi-encoder sim) *is* unsupervised
post-retrieval QPP — the [RELATED_WORK_GROUNDING](../../paper/submission/RELATED_WORK_GROUNDING.md)
repositioning. Any routing/selection claim must use NQC/WIG/SMV as baselines
and report against the field's reliability bars ([[datta-qpp-reliability]]:
selective processing needs τ ≥ ~0.5; observed ceilings ≈ 0.37).

**Status in our work — a principled negative.** Per-query QPP routing of
SCOPE-on/off failed: best predictor WIG-CE at τ ≈ −0.11, far under the bar;
consistent with Faggioli's dense-QPP pessimism and Datta's ceilings. What
survives is *regime-level* routing ([[regime-routing]]) — dataset/slice-level
signals separate the regimes even though per-query signals can't. Near-twins
to engage: Emami'26 ([[emami-qpp-variant]], QPP-selects-among-query-variants)
and Tian'25 ([[tian-right-track]], QPP of generated queries → answer quality).
Both also document the QPP↔answer-quality disconnect — the field's version of
our [[answer-conversion-gap]].

**Open questions.** Is there any *generation-aware* predictor (using the
pseudo-doc itself, or gen-vs-retrieval agreement) that clears τ 0.37? Emami'26
explicitly punts this — it's an open, ownable slot, but Datta's ceilings say
don't bet the paper on it.

## Links
[[faggioli-qpp]] · [[datta-qpp-reliability]] · [[emami-qpp-variant]] ·
[[tian-right-track]] · [[regime-routing]] · [[query-drift]] ·
[[answer-conversion-gap]] ·
[qpp routing analysis](../../docs/generated/raw_retrieval_confidence_routing_2026-05-26.md)
