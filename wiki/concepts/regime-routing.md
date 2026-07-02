---
title: Regime Routing (when to expand)
type: concept
tags: [routing, expansion, selective-qe, recipe]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# Regime routing — the operational recipe

**Definition.** Decide *whether/how* to use generative query expansion by
regime rather than per-query: vanilla [[scope]] on weak-query regimes; raw∪SCOPE
candidate pooling (union before rerank) on strong/intermediate regimes; never
pure expansion on strong queries.

**Evidence.**
- Pooling raw∪SCOPE: +2–3pp Hit@5 on all 5 strong/intermediate BEIR sets and
  Housing, but *destroys* the weak-query gain (BarExam pool 3.9% vs SCOPE
  12.0% — the CE reranker buries generated-query candidates when raw
  candidates look plausible). 3-candidate diversity adds nothing over
  raw∪SCOPE. → No single recipe covers both regimes; routing is forced.
- Per-query routing is dead: QPP τ ≈ 0.11 vs the ≥0.5 reliability bar
  ([[qpp]], [[datta-qpp-reliability]]) — a *principled negative* consistent
  with Datta'25 ceilings and Faggioli dense-QPP pessimism.
- Regime-level (dataset/slice) routing works: it avoids ~14% Housing dilution
  while keeping BarExam wins (in-sample; held-out validation still owed).
- Prior art to engage: selective query expansion is ~20 years old (Amati'04);
  the modern LLM instance is Emami'26 variant-selection
  ([[emami-qpp-variant]]); Adaptive-RAG/Mallen route retrieve-vs-not, not
  expand-vs-not ([[adaptive-rag-mallen]]).

**Why it matters.** It converts the HousingQA embarrassment (C10) into a
predicted, avoidable failure with a recipe — the practical payoff of the
[[weak-vs-strong-query-regime]] theory. As a *contribution* it must be framed
as recipe + negative-result-on-per-query, not as a novel router (taken).

**Open.** Held-out regime-routing validation; a pre-retrieval regime detector
(corpus-conditioned, few-shot probe set?) cheap enough for deployment; where
MASLegalBench/Legal-Link-EU sit on the axis (anchored questions = strong).

**SUPERSEDED as primary recipe (2026-07-02, [[thesis-v2]] P3 confirmed both
regimes)**: the pooling-kills-weak result that forced routing was an
*ms-marco-CE artifact*. With a trained judge as selector, always-pool wins on
BOTH ends: weak ([[judge-pilot-v0-results]] 20.6% vs SCOPE-alone 12.0%,
p=3.4e-06) and strong ([[judge-pilot-housing]] 55.0% vs CE-pool 38.2%,
p=2.5e-23; vs SCOPE-alone p=8.5e-12). This page remains the recipe for
**judge-less deployments** (a general-domain CE does force the regime split)
and as the historical record of why the routing question arose.

## Links
[[weak-vs-strong-query-regime]] · [[qpp]] · [[query-drift]] ·
[[emami-qpp-variant]] · [[adaptive-rag-mallen]] · [[scope]] · [[beir-phase1]]
