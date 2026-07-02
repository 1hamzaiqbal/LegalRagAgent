---
title: Judge Pilot v0 (Tinker-trained legal relevance judge)
type: method
tags: [judge, tinker, path-c, reranking, expert-judgment]
created: 2026-07-02
updated: 2026-07-02
status: maintained
code: scripts/judge_pilot/
impl_status: prototype
---

# Judge pilot v0 — trained legal relevance judge vs the cross-encoder

**Hypothesis.** The answer-conversion wall is partly a *selector* problem: the
general-domain ms-marco CE demonstrably buries gold passages it already holds
(gold median rank 4–5 in pooled candidates; raw∪SCOPE pool Hit@5 3.9% vs
SCOPE-alone 12.0% on BarExamQA — [[pooling-regime]]). A judge *trained on
legal relevance* should recover buried pool headroom that similarity scoring
cannot. This is [[expert-judgment-replication]] at v0 scale: supervision is
free (gold ids from the benchmark = a lawyer-annotated relevance label), no
new human labels needed.

**Setup** (`scripts/judge_pilot/`):
- **Task**: (fact pattern, question, passage) → "Does this passage state the
  controlling legal rule? Yes/No".
- **Data**: BarExamQA qa.csv (1,195 q); positives = gold passages; hard
  negatives = actual retrieved non-gold from the signed raw + SCOPE caches
  (the exact distractors the CE faced). Question-level split 700/95/400
  (seed 42): train 3,500 pairs, dev 475.
- **Eval**: rerank the *identical* raw∪SCOPE 20-candidate pools from the
  signed pool cache (399 held-out pools, gold-in-pool recall ceiling 22.8%);
  score = logP(" Yes") − logP(" No"); compare Hit@5/MRR@5 vs the CE's recorded
  top-5 on the same pools, plus raw-top5 / SCOPE-top5 references and a
  **zero-shot (untrained) same-model judge** — the prompted-vs-trained
  contrast that is the [[thinking-machines-expert-judgment]] thesis.
- **Training**: Tinker LoRA (rank 32) on `Qwen/Qwen3.5-9B`, cross-entropy on
  the single answer token (" Yes"=7179 / " No"=2233), lr 1e-4, batch 128,
  3 epochs (~85 steps). Passage text hydrated from the EIT cluster corpus
  (`barexam_qa_train.csv` + the 857K-passage Chroma) because the Mac lane has
  no local index.

**Interpretation guide.**
- Judge ≫ CE on gold-in-pool conversion → selector confirmed as a bottleneck;
  motivates scaling to Housing + the union recipe (queue #8) and real lawyer
  labels (Path C proper).
- Trained ≫ zero-shot → the "judgment needs labels, not prompts" thesis
  transfers to legal relevance at academic scale.
- Judge ≈ CE → headroom is not selector-limited; the wall moves to the reader
  ([[answer-conversion-gap]]), and Path C pivots to answer-quality judging.

**Cost**: Tinker credits (~$150 available); train ≈ 5.6M token-passes,
eval ≈ 30M prefill tokens across ~32K logprob calls (2 per candidate × 2 arms).

## Links
[[expert-judgment-replication]] · [[answer-conversion-gap]] ·
[[pooling-regime]] · [[thinking-machines-expert-judgment]] ·
[[direction-2026-07]] (queue #7) · results: [[judge-pilot-v0-results]]
