---
title: Helpfulness Benchmark (Idea 3) — measure whether retrieval helped the reader, not whether it hit gold
type: direction
tags: [benchmark, evaluation, metric, cost-per-task, dormant]
created: 2026-07-02
date: 2026-07-02
status: DORMANT — written to be picked up; no active work
---

# Idea 3 (dormant): the evidence-helpfulness benchmark / metric paper

**Status: DORMANT.** Recorded from the 2026-07-02 mentor meeting
([[08-meeting-notes]]) so it can be resumed without re-derivation. Flagged
in the meeting as the *lowest-cost* paper candidate because most of the
required experiments already exist as signed rows.

## The pitch (one paragraph)
RAG benchmarks score retrieval with rank metrics (Hit@k, MRR, nDCG, F1/AUC
over qrels) and report answer accuracy separately. But the decision-relevant
question in the agentic era is neither: it is **whether the retrieved
context helped this reader solve this task, at what cost**. We have repeated
demonstrations that the two diverge in both directions — gold retrieved but
useless (BarExam/70B: 8× exposure → +0.6pp answers), and non-gold context
that carries real answer value (Housing: gold-absent evidence **+12.0pp**).
Propose the metric family, show the divergences on existing data, and give
the community a cheap protocol: one llm_only anchor + paired evidence arms.

## Proposed metric family (v0 sketch)
1. **EHE (Evidence Helpfulness Effect)**: per-question paired Δcorrectness
   of an evidence arm vs the same reader's llm_only, reported with
   gold-present / gold-absent stratification. This is the row-level primitive
   ([[judge-answer-conversion]] already computes it).
2. **Harm-adjusted retrieval score**: Hit@k is silent about the (1−Hit@k)
   mass; weight exposure by the measured gain/cost pair — the break-even
   model as a *metric* (BarExam/70B: +2.4pp gain, −3.8pp cost → break-even
   61% Hit@5). A retrieval system's score = expected answer delta, not rank.
3. **Cost-per-solved-task**: marginal solves per token/call relative to
   llm_only (agentic framing from the meeting). All inputs already logged
   per row by the harness (input/output tokens, llm_calls, latency).
4. Reader-conditioning: every metric is reported *per reader*, since
   helpfulness inverts with parametric deficit (the 70B/8B 2×2).

Convergent construct: SKILL0's per-skill on-policy helpfulness Δ_k
([[skill0]]) — same idea at the skill level; cite as independent
arrival at "helpful-to-this-policy ≠ relevant."

## Evidence already in hand (the paper's empirical core, no new compute)
| Divergence | Numbers | Source |
|---|---|---|
| Gold retrieved, no help | BarExam/70B golden vs llm_only **+0.5pp ns**; Gemma-26B golden **−2.3pp** (gold *hurts*) | CLAUDE.md signed rows, [[01-scope-submission]] |
| Gold helps enormously elsewhere | CaseHOLD/70B **+25.7pp**, SCALR **+19.1pp**, Housing **+22.5pp** | CLAUDE.md signed rows |
| Neighbor dilution with gold present | CaseHOLD 97.5→79.4 (**−18.1pp**, p=1e-187); SCALR −10.5pp; one inversion BarExam/Gemma **+2.1pp** | CLAUDE.md signed rows |
| Non-gold context has value / harm by regime | gold-absent evidence: BarExam/70B **−3.8pp**, Housing/70B **+12.0pp**, BarExam/8B **+7.3pp** | [[judge-answer-conversion]] |
| Selector gains that do/don't convert | judge 5.4× exposure → 0 answer gain (BarExam/70B) vs 1:1 conversion (Housing) | [[judge-answer-conversion]], [[judge-pilot-housing]] |
| Reader inversion | the 70B/8B × BarExam/Housing 2×2, crossover ≈ llm_only 60% | [[judge-answer-conversion]] §2×2 |
| Exposure metrics themselves | full Hit@k/MRR ladders across arms | [[snap-vs-hyde-ledger]], docs/generated qrels files |

## What a submission still needs (the dormant queue)
1. Formalize the metrics + estimators (paired tests, CIs; the break-even
   estimator's variance).
2. Re-analysis pass over existing detail logs to produce the full metric
   tables (pure computation — see [[offline-bandit-v0]] for the loader that
   already joins paired arms by `idx`).
3. **1–2 non-legal/general replications** (e.g. NQ or HotpotQA full-corpus
   with a gold-injection arm) — the only new compute; needed so the paper
   isn't read as legal-only.
4. Related-work check (queued, unread): utility-based IR evaluation
   tradition; RAGAS/ARES-style LLM-judged RAG metrics; LegalBench-RAG
   span-level eval ([[legal-rag-benchmarks-src]]); [[power-noise-lostmiddle]]
   (distractor harm); LRAGE. Position: we measure *reader-conditional task
   utility with a causal paired design*, not judged relevance.
5. Decide framing: analysis/resource paper (IR eval track, SIGIR
   resource/perspective; or ARR) vs a section inside the mechanism paper.
   Risk to answer: "isn't this just an ablation?" — response: the paired
   llm_only-anchored protocol + break-even estimator is the contribution,
   demonstrated across 6+ reader×task cells with sign flips.

## Pick-up checklist (in order, ~first week)
- [ ] Read the four related-work families above (half a day).
- [ ] Write the metric definitions doc + estimator choices.
- [ ] Run the re-analysis over `logs/eval_*_20260702_*` + historical golden
      rows; produce master tables.
- [ ] Cost columns from existing logs; cost-per-solved-task table.
- [ ] Pick + run one general-domain cell (NQ/HotpotQA, ~$10-20 Groq).
- [ ] Draft 4-page core; decide venue.

## Links
[[08-meeting-notes]] · [[judge-answer-conversion]] · [[thesis-v2]] ·
[[skill0]] · [[offline-bandit-v0]] · [[direction-2026-07]]
