---
title: Expert-Judgment Replication (Thinking Machines pattern → legal)
type: concept
tags: [direction, judgment, evaluation, rl, legal-reasoning]
created: 2026-07-02
updated: 2026-07-02
status: draft
---

# Replicating expert judgment — candidate new direction

**The pattern** (from Thinking Machines' financial-tasks post,
[[thinking-machines-expert-judgment]] — details pending ingestion): collect
expert judgments on domain tasks, train/tune models (or judges/reward models)
to *replicate* that judgment, evaluate agreement with held-out experts rather
than against brittle exact-match labels.

**Why it's interesting for us** (advisor/user flagged it 2026-07-02):
1. **It attacks the [[answer-conversion-gap]] at the metric layer.** Our whole
   pipeline is scored by exact-match answer lines; legal quality (correct
   authority, sound application, jurisdictional fit) is invisible to it. A
   lawyer-judgment-replicating grader would let retrieval quality show up in
   scores the way it shows up in practice.
2. **It answers criticism C1 head-on** ([[icml-ai4law-2026-rejection]]):
   "practitioners' actual analytical processes" — modeling what lawyers judge
   as good research/answers is engaging practice, not decorating an IR paper
   with legal vocabulary.
3. **It composes with what we own**: signed full-N logs across 6 benchmarks ×
   3 models × 7 methods are a ready-made pool of (question, evidence, answer)
   triples to judge; the geometry machinery gives candidate *features* for
   judgment models.

**Candidate instantiations** (to pressure-test at the meeting):
- **Legal relevance judge**: replicate lawyer judgments of *passage relevance*
  for fact-pattern queries (BarExamQA-style). Direct impact: a learned
  reranker objective aligned with legal relevance instead of ms-marco CE
  similarity — plausibly the biggest lever on our weak-query Hit@5 wall
  (gold missed 9/10 times even after 8× lift).
- **Answer-quality judge**: replicate expert grading of RAG answers (issue
  spotting, rule accuracy, citation support) → evaluate methods on judged
  quality, where retrieval's value may finally be visible.
- **Snap-answer verifier**: expert-style judgment of when parametric intuition
  is trustworthy — a learned, legally-grounded router (contrast with failed
  generic [[qpp]] routing).

**Open questions.** Where does expert signal come from (annotators? bar-exam
model answers? judicial opinions as supervision?); how much does the pattern
depend on TM-scale expert data; is there a public legal-judgment dataset to
bootstrap (discovery sweep B is looking).

## Links
[[thinking-machines-expert-judgment]] · [[answer-conversion-gap]] ·
[[icml-ai4law-2026-rejection]] · [[direction-2026-07]] · [[legal-rag-benchmarks]]
