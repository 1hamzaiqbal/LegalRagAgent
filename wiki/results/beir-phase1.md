---
title: BEIR Phase 1 — mechanism replication + expansion is net-negative on strong queries
type: result
tags: [beir, mechanism, query-drift, generalization]
created: 2026-07-02
updated: 2026-07-02
date: 2026-05-26
verdict: win (mechanism) + negative (ungated expansion)
evidence: docs/generated/beir_phase1_verification_2026-05-26.md
---

# BEIR Phase 1 (5 datasets, full-N, Gemma 4 26B generator)

**Setup.** SciFact / NFCorpus / FiQA / TREC-COVID / SciDocs, full query sets
(N=50–1000), raw vs HyDE vs SCOPE generation caches, gte-large + MiniLM CE,
Hit@5/@10, Collins-Thompson RI, per-query CE gold-affinity deltas and raw
margin $M_{raw} = CE(q,gold) - \max_d CE(q,distractor)$.

**Numbers** (pooled N=2321):
- Ungated expansion is **net-negative on this strong-query slice**: raw Hit@5
  62.0% → HyDE 30.7% (**−31.3pp**, RI −0.313) vs SCOPE 49.7% (**−12.2pp**,
  RI −0.122). SCOPE is the far more drift-robust expansion on every dataset
  (e.g. TREC-COVID −2pp vs −28pp; NFCorpus −4.3 vs −35.9).
- **Mechanism replicates 5/5**: per-query retrieval gain ~ CE gold-affinity
  movement; pooled ρ = 0.501 (HyDE), 0.426 (SCOPE).
- **Falsification replicates**: predicting geometric failure (ΔM<0), geometry
  AUC 0.944/0.909 vs OOV/log-perplexity 0.520/0.509; predicting realized
  retrieval hurt, geometry 0.798/0.743 vs 0.490/0.598.
- **Regime crossover within-dataset**: lowest raw-margin quintile is the only
  bin where expansion *helps* (e.g. SciDocs SCOPE bins 1–3 positive, 4–5
  negative); sign-crossover in 4/5 datasets — the [[weak-vs-strong-query-regime]]
  law holds *within* corpora, not just across them (the confound-breaking
  design the deep-read demanded).

**Verdict.** The mechanism ([[geometry-vs-factuality]]) and the regime law are
domain-general; ungated expansion is not a deployable policy on strong-query
corpora — motivates [[regime-routing]]. Also the origin of the surviving
snap-conditioning claim: SCOPE ≫ HyDE *robustness* under drift (a variance
claim, not a mean-lift claim — relevant to C7 in
[[icml-ai4law-2026-rejection]]).

**Caveats.** Single generator (Gemma 26B) for this phase — model breadth added
in phase 1b (Qwen/Mistral/DeepSeek, `docs/generated/beir_phase1b_model_breadth_2026-05-26.md`);
TREC-COVID margins mostly undefined (dense qrels); Hit@5 on BEIR ≠ nDCG
conventions (we score exposure, not graded ranking).

## Links
[[weak-vs-strong-query-regime]] · [[query-drift]] · [[geometry-vs-factuality]]
· [[regime-routing]] · [[scope]] · [[hyde]] ·
[verification report](../../docs/generated/beir_phase1_verification_2026-05-26.md) ·
[model breadth](../../docs/generated/beir_phase1b_model_breadth_2026-05-26.md)
