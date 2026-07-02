# Skill: retrieval-effort allocation

You are deciding how a reader model should handle a question: answer from
its own knowledge, or answer with retrieved evidence — and if retrieving,
which retrieval strategy. Apply these measured rules.

## Rule 1 — the reader's parametric deficit decides whether evidence pays
Estimate whether the reader would answer this task correctly WITHOUT
evidence (its no-retrieval accuracy on this kind of question).
- If the reader is strong on the task (would score roughly above 60-65%
  without evidence), retrieved evidence usually does NOT help and often
  hurts: passages that do not contain the decisive text act as distractors
  (measured: −3.8 points when evidence misses gold for a strong reader on
  bar-exam MC; raw retrieval cost a strong reader −2.4 points on medical MC).
  Prefer NO RETRIEVAL.
- If the reader is weak on the task (well below ~60% without evidence),
  evidence pays even when imperfect (measured: +7.3 points from
  gold-missing evidence for a weak reader on bar-exam MC; +12 points on
  statutory yes/no where neighboring provisions carry answer value).
  Prefer RETRIEVAL.

## Rule 2 — expected evidence quality gates the decision at the margin
Evidence helps in proportion to the chance the decisive passage is actually
in the top-k. If the retrieval pipeline's hit rate on this task is far below
the break-even point implied by Rule 1's gain/cost ratio (example: gain
+2.4 / cost −3.8 → break-even ≈ 61% hit rate), retrieval is negative in
expectation for a strong reader even with the best available selector.

## Rule 3 — task structure changes the sign of non-gold evidence
- Multiple-choice with self-contained facts (bar exam, medical boards):
  non-gold passages are distractors; be conservative about retrieving.
- Entailment against a specific corpus (does statute X permit Y in state
  Z): even non-gold same-jurisdiction passages carry value; be liberal
  about retrieving, and prefer jurisdiction/metadata-filtered retrieval.

## Rule 4 — cost accounting
Retrieval roughly doubles to quadruples input tokens per question
(measured: 0.25→2.6-3.7k tokens on statutory questions, 0.7→1.5-1.7k on
MC). Under a token budget, spend retrieval only where Rules 1-3 predict a
positive expected gain; otherwise answer directly.

## Decision procedure
1. Classify the task type (MC-parametric vs corpus-entailment) and the
   reader's strength on it.
2. Strong reader + MC + low pipeline hit-rate → answer from knowledge.
3. Weak reader, or corpus-entailment task → retrieve; choose the strategy
   with the best selector available (trained judge > generated-query
   retrieval > cross-encoder > raw question, where measured).
4. When uncertain, estimate both expected values with Rules 1-4 and pick
   the larger; state the chosen action decisively.
