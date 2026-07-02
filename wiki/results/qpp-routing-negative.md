---
title: QPP Routing — a principled negative
type: result
tags: [qpp, routing, negative-result]
created: 2026-07-02
updated: 2026-07-02
date: 2026-05-26/29
verdict: negative (useful)
evidence: docs/generated/raw_retrieval_confidence_routing_2026-05-26.md, docs/generated/credibility_D_ood_predictor_2026-05-29.md
---

# Per-query QPP routing of expansion: closed as a principled negative

**Question.** Can a label-free QPP signal on the *raw* query decide per-query
whether to invoke SCOPE?

**Answer: no, and we can say why.**
- In-distribution: no predictor clears the τ≥0.5 reliability bar
  ([[datta-qpp-reliability]]); best WIG-CE τ≈−0.11. Regime-level routing
  still avoids ~14% of Housing dilution while keeping BarExam wins
  (in-sample).
- Out-of-distribution (credibility D): learned QPP router held-out-generator
  Kendall τ = **0.090**, held-out-dataset τ = **0.052**; label calibration up
  to 1000 examples doesn't approach the bar.
- Consistent with the field: Faggioli'23 dense-QPP pessimism
  ([[faggioli-qpp]]), Datta'25 τ ceilings ≈0.37, Emami'26's weak
  QPP↔answer link ([[emami-qpp-variant]]).

**What survives**: [[regime-routing]] at dataset/slice level (weak → expand;
strong → raw∪SCOPE pool), which needs held-out validation but does not depend
on per-query prediction.

**Framing rule** (from [RELATED_WORK_GROUNDING](../../paper/submission/RELATED_WORK_GROUNDING.md)):
present this as "why no cheap per-query router works here" with NQC/WIG/SMV
baselines — a contribution to the selective-expansion literature's negative
side, never as a novel router.

## Links
[[qpp]] · [[regime-routing]] · [[datta-qpp-reliability]] ·
[[emami-qpp-variant]] · [[tian-right-track]] ·
[routing analysis](../../docs/generated/raw_retrieval_confidence_routing_2026-05-26.md) ·
[OOD predictor report](../../docs/generated/credibility_D_ood_predictor_2026-05-29.md)
