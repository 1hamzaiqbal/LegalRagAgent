---
title: Factuality Falsification — geometry beats hallucination as failure predictor
type: result
tags: [factuality, falsification, judge, mechanism]
created: 2026-07-02
updated: 2026-07-02
date: 2026-05-28 (q200) / 2026-05-31 (full-N gpt-4o)
verdict: win (survived escalating credibility battery)
evidence: docs/generated/factuality_falsification_2026-05-28.md, docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31.md
---

# Factuality falsification — the C4 rebuttal

**Question.** When SCOPE's generated passage hurts retrieval, is the cause
hallucinated content (the reviewer-C4 / CSQE story) or embedding geometry?

**Phase A (q200 × 5 datasets, Gemma judge).** Predicting realized retrieval
hurt, pooled AUC: OOV/logPPL 0.514 · gold-grounded factuality 0.581 ·
raw-top3 factuality 0.529 · **geometry 0.791** · factuality+geometry 0.792
(**marginal lift +0.001**). High-factuality rows still fail geometrically
(8.5% hurt rate; within-stratum ρ(ΔM, retrieval delta)=0.389).

**Credibility battery (2026-05-29/31)** — the strawman-proofing:
- **Independent judge**: full-N gpt-4o re-judge (BarExamQA + 4 BEIR sets,
  $21.96): factuality AUC **0.548**, geometry 0.823, joint 0.826, marginal
  **+0.003**; judge-judge IRR Spearman 0.681, κ@0.5 0.614. Survival gates
  (factuality <0.65, marginal <+0.03) met.
- Replicates on BEIR at full N ([[beir-phase1]]: geometry 0.944 vs 0.520 on
  ΔM<0).

**Honest nuance** (from the full AUC table, keep in any paper):
- BarExamQA is the *least* clean dataset for the story: raw-top3 factuality
  alone hits AUC 0.757 there, and gold-factuality predicts ΔM<0 at 0.728 —
  on the weak-query legal end, factuality and geometry are correlated. The
  falsification is strongest pooled and on BEIR.
- Single independent judge ≠ two-judge closeout: Claude judge still pending;
  gpt-4o wave excluded HousingQA and truncated SciDocs (budget cap).

**Why it matters.** Direct empirical answer to C4 in
[[icml-ai4law-2026-rejection]] ("pseudo-docs inherit LLM fabrication"): the
pseudo-document doesn't need to be *true*, it needs to be *well-placed*; and
fixing expansion failures means steering geometry (exemplars, corpus
anchoring), not fact-checking. This is the paper-grade anchor result of
[[geometry-vs-factuality]].

## Links
[[geometry-vs-factuality]] · [[beir-phase1]] · [[csqe]] · [[scope]] ·
[[icml-ai4law-2026-rejection]] ·
[phase A](../../docs/generated/factuality_falsification_2026-05-28.md) ·
[gpt-4o wave](../../docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31.md) ·
[battery summary](../../docs/generated/credibility_comprehensive_summary_2026-05-29.md)
