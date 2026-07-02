---
title: Affinity-Margin Mechanism (legal, pre-registered)
type: result
tags: [mechanism, margin, legal, preregistered]
created: 2026-07-02
updated: 2026-07-02
date: 2026-05-25/26
verdict: win (P2/P4 supported; P1/P3 floor-artifact caveat)
evidence: docs/generated/scope_gap_mechanism_2026-05-25.md, docs/generated/affinity_margin_oncache_2026-05-26.md
---

# The affinity-margin mechanism on the legal caches

**Claim.** Per-query SCOPE retrieval benefit is governed by how far the
generated passage moves query–gold CE/embedding affinity, relative to
distractors: $margin = \operatorname{CE}(q,gold) - \max_d \operatorname{CE}(q,d)$.

**Findings** (pre-registered P1–P4 on the legal caches):
- Gold-affinity movement ~ retrieval gain: pooled Spearman ≈ **0.44**,
  monotone; backfires when generation imposes a wrong frame
  ([[query-drift]]).
- **Gold-affinity movement dominates confounds**: partial-R² 0.13 vs ≤0.004
  for length/format/domain covariates.
- Distractor-margin didn't beat gold-affinity-only *pooled* (P1/P3 "killed"),
  but that was a BarExam floor artifact — within Housing the crossover held;
  the margin variant stays re-openable (per the experimental-discipline rule:
  one quirky dataset ≠ falsified).
- P4 (failures are geometric, not knowledge-gap): AUC 0.91 vs 0.57 on the
  weak hallucination proxy — later strengthened with real judges in
  [[factuality-falsification]].

**Lineage.** This is the analysis that turned SCOPE from a method claim into a
mechanism claim (grounding doc's "recommended anchor"), then replicated on
BEIR ([[beir-phase1]]) and across retrievers ([[three-retriever-generality]]),
and got its falsification arm hardened by the credibility battery.
Perplexity/OOV had already been ruled out
([docs/generated/perplexity_axis_2026-05-25.md](../../docs/generated/perplexity_axis_2026-05-25.md): per-query corr ≈ 0; only a
weak dataset-level regime separator — used once for the MedQA pre-screen).

## Links
[[geometry-vs-factuality]] · [[weak-vs-strong-query-regime]] ·
[[beir-phase1]] · [[three-retriever-generality]] ·
[[factuality-falsification]] ·
[gap mechanism](../../docs/generated/scope_gap_mechanism_2026-05-25.md) ·
[margin on-cache](../../docs/generated/affinity_margin_oncache_2026-05-26.md) ·
[perplexity axis](../../docs/generated/perplexity_axis_2026-05-25.md)
