---
title: Skill-Distillation Bridge — internalize a big model's agentic retrieval skills into a small one
type: concept
tags: [direction, distillation, skills, judge, agentic, tinker, eit]
created: 2026-07-02
date: 2026-07-02
status: gated active track — plumbing validated; novelty narrowed; E2 pending
---

# The bridge: SKILL0 × expert-judgment distillation × our judge program

> **2026-07-17 state:** E0 and E1 are complete and EIT job 93802 passed the
> full teacher→student OPD smoke. [[sdar]] and [[skill1]] are now read. The
> direction survives only in a narrower form: reader-conditioned legal
> retrieval control distilled cross-scale. Bare OPD is a plumbing/collapse
> baseline; safe E3 requires task reward plus gap-gated dense supervision and
> is blocked on the E2 teacher skill-gap A/B.

**One sentence**: [[skill0]] shows in-context skills can be *internalized
into the same model's* weights via a helpfulness-driven curriculum; the
Thinking Machines line ([[thinking-machines-expert-judgment]]) and our own
judge battery show *small* models trained on outcome labels beat prompted
frontier models at judgment tasks; the proposed direction is the cross of
the two — **internalize a big model's (or skill-augmented policy's) agentic
retrieval skills into a small model**, i.e. skill-curriculum *distillation*
rather than skill-curriculum *self-internalization*.

## Why we're positioned to do this
1. **The pattern already worked once, at $0.** Trained 9B judge 20.6% vs
   prompted 235B 15.3% on identical pools ([[judge-capacity-dial]]) — expert
   judgment compressed into a small model beats scale. Mixed labels
   generalize across legal tasks with zero tax ([[judge-mixed-legal]]). The
   free EIT lane reproduces Tinker exactly ([[judge-pilot-v0-results]]
   §free-infrastructure).
2. **We know what makes context helpful** — the three dials. SKILL0's
   curriculum needs a per-skill on-policy helpfulness signal Δ_k; our
   conversion decomposition *is* that signal for retrieval evidence
   (gold-present/gold-absent effects, break-even model —
   [[judge-answer-conversion]]). We can define skill/evidence withdrawal
   schedules on a measurement we've already validated.
3. **The target skills are the meeting's Ideas 1+2**: search-effort control
   (when to retrieve, how many, parallel vs sequential, cost-aware) and
   conflict arbitration (doc-vs-doc, doc-vs-prior, verify-vs-refetch).
   These are reasoning-heavy, big-model-favored behaviors — exactly what
   you'd want to distill down ([[08-meeting-notes]]).

## Novelty after the required reads

- [[sdar]] already combines agentic task RL with privileged-skill on-policy
  self-distillation and shows why negative-gap gating is necessary.
- [[skill1]] already co-evolves skill search, selection, use, and distillation
  in one policy.
- Therefore neither “skill internalization” nor “self-distilled agentic RL” is
  our contribution. The remaining wedge is **cross-scale compression of a
  reader-conditioned retrieval-control skill**, measured on legal task success
  and cost, with evidence utility/harm as the supervision object.
- This wedge is still provisional: it earns a method claim only if the skill
  improves the teacher at E2 and distillation improves a pre-registered
  accuracy–cost frontier at E3.

## Technical paths (by teacher access)
| Teacher | Method | Notes |
|---|---|---|
| Open weights (Qwen-235B, Llama-405B, or a skill-augmented mid-size) | **Task reward + gap-gated OPD** — student samples, teacher top-k logprobs provide a small dense auxiliary signal | Primary safe design after SDAR; bare OPD remains a diagnostic only |
| Closed (GPT/Claude) | Sequence-level KD on teacher traces (SFT); GKD-style: student samples, teacher *scores/ranks* (reward, not logits); rejection-sampling SFT | Meeting action item: **survey how people distill from closed models** — OpenAI exposes top-20 logprobs, Anthropic none, so token-level OPD is out |
| No teacher, skills-as-teacher | SKILL0's own recipe at small scale: give the small model the skill *files* + curriculum-withdraw | Baseline arm — tests whether a teacher model is even needed vs just the skill text |

That third row is the cheapest scientifically interesting comparison:
**skill-context curriculum vs big-teacher distillation vs both** on the same
small student — does the teacher add anything beyond the skill description?
(SKILL0's own finding that models "follow but don't acquire" skills without
RL suggests the curriculum matters more than the teacher; our SciDocs result
warns the supervision signal's *semantics* dominate everything —
[[judge-pilot-scidocs]].)

## A v0 that fits our infrastructure (sketch, not committed)
Student: Qwen3.5-9B (or 3B) on the free EIT lane. Task: **agentic retrieval
effort control on BarExamQA/Housing** — the policy decides
retrieve-or-not / k / re-query, actions we can execute against our cached
retrieval stack; reward = answer correctness − λ·(retrieval + token cost)
(the meeting's cost-per-task metric, Idea 3). Teacher/skill arm per the
table above. Eval: cost-vs-accuracy frontier against fixed-k baselines
(k=0 = llm_only, k=5 = our signed rows — both already measured). The
three-dial numbers predict where effort should concentrate; the question is
whether a small policy *learns* that allocation.

Honest blockers before a scientific E3: (1) E2 teacher skill-gap A/B; (2)
gap-gated/task-reward implementation and tests; (3) closed-teacher
distillation survey; (4) SKILL0's stack is verl+vLLM multi-GPU RL — heavier
than our PEFT lane; a bandit-style single-turn version (retrieve-or-not +
k choice) may be the right first rung rather than full multi-turn RL.

## Rung 1 — EXECUTED same day ([[offline-bandit-v0]])

The single-turn bandit was built as pure offline replay of the paired
2026-07-02 arms (5 reader×task cells, zero new LLM calls;
`scripts/bandit/offline_bandit_v0.py`). Verdict: **instructive negative** —
no cheap external policy (question features + trained-judge scores, logistic
or 1-D gate) beats the best fixed arm in any cell, while the per-question
oracle sits 8–24pp above every fixed arm. The allocation headroom is real
(arms are strongly complementary per-question) but unreachable from external
features — extending [[qpp-routing-negative]] to answer-level allocation.
**This is rung 2's motivation**: the allocation signal must live in the
model's own state, i.e. internalize the policy (SKILL0's bet) rather than
route around a frozen one. Rung 2 = small model *emits* the
retrieve/k decision in its own forward pass, trained on the free EIT lane
(supervised on oracle actions first — cheaper than RL — then RL if the
supervised ceiling binds).

## Links
[[skill0]] · [[sdar]] · [[skill1]] · [[thinking-machines-expert-judgment]] ·
[[expert-judgment-replication]] · [[judge-capacity-dial]] ·
[[judge-mixed-legal]] · [[judge-answer-conversion]] · [[08-meeting-notes]] ·
[[direction-2026-07]]
