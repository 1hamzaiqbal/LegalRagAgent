---
title: OPD and Retrieval-Skill Distillation Track
type: hub
tags: [active-track, opd, distillation, agentic-rl]
created: 2026-07-17
updated: 2026-07-17
status: gated
---

# OPD and retrieval-skill distillation

## Current state

The implementation path works end to end: job 93802 served a Qwen3-8B
teacher, trained a Qwen3-1.7B student for three finite on-policy steps, and
wrote checkpoints. This validates plumbing only.

The literature gate is now closed: [[sdar]] and [[skill1]] were read. They
invalidate broad novelty language and make bare OPD an unsafe primary arm.

## Gates

1. **E2 teacher skill gap:** skill-augmented teacher must outperform the same
   teacher without the skill on a held-out allocation/control task.
2. **Objective safety:** real E3 uses task reward plus gap-gated OPD. Bare OPD
   remains a collapse diagnostic; loss finiteness is not success.
3. **Outcome relevance:** evaluation is task accuracy and cost, including the
   BarExam/70B `llm_only` decision—not imitation loss alone.

## E3 arms if E2 passes

1. outcome-label/task-RL baseline;
2. task RL + gap-gated OPD from a skill-augmented teacher;
3. bare OPD diagnostic;
4. student with skill in context but no training;
5. trace KD for the closed-teacher comparison.

## Candidate task extension

[[effort-conditioned-resource-allocation]] could provide a richer E2/E3 task
than the current single-turn allocation labels: a privileged teacher chooses
among thinking, search, verification, and stopping under an explicit cost
condition, and a smaller student is evaluated on the same cost/accuracy
frontier. This does not relax the gates. Direct cost-conditioned task RL is the
primary baseline; task RL plus gap-gated OPD must beat it; and bare OPD remains
a collapse diagnostic.

## Candidate primary framing after the 2026-07-17 audit

[[compute-elasticity-distillation]] is the cleaner domain-general framing, but
the July follow-up search narrowed it again. INTENT, MOC, ClawTrace, CoRL, and
OPID occupy dynamic tool prices, unseen-preference control, cost-aware skill
distillation, budget-conditioned routing, and skill-conditioned OPD. The
primary object is now whether cross-scale transfer preserves the teacher's
paired same-task action response across seen and unseen prices. Teacher-only
skills are a factorial condition, not the novelty claim. Start with one fixed
token cap, one variable Python-tool price, and Reasoning Gym tasks. Legal
retrieval becomes a later stress test, not a benchmark dependency. The full
handoff is [[compute_elasticity_handoff_2026-07-17/README]].

The literature audit adds three non-negotiable diagnostics:

1. [[rethinking-opd]]: verify thinking-pattern compatibility and genuinely new
   teacher signal;
2. [[reward-gated-opd]]: include reward-gated supervision rather than claim
   generic gating;
3. [[rethinking-privileged-opd]]: measure the whole budget curve and audit
   verification/backtracking, because privileged OPD can improve short-budget
   scores while destroying long-budget gains.

## Adjacent candidate: skill lifecycle, not another OPD architecture

[[skill-lifecycle-research-snapshot-2026-07-17]] audits the proposed
`SkillOpt → SKILL0 → OPD` chain. The literal method claim is closed:

- [[opcd]] already performs context-conditioned, on-policy reverse-KL from a
  larger teacher to a smaller context-free student, including optimized system
  prompts;
- [[promptkd]] already adapts teacher-side soft context using the student's
  distribution specifically to produce student-friendly generative KD;
- [[skill-sd]] and [[seed-self-evolving-opd]] already combine teacher-only or
  hindsight skills, OPD, and task reward with no skill at deployment;
- [[skillc]] already turns paired skill/no-skill behavior into direct
  internalization credit; and
- [[latent-skill]] plus [[skill-zero-five]] already make context-versus-weights
  placement and modular weight skills explicit.

[[skillgen-verified]], [[masa]], and [[skilllens]] now add an important
correction: the contextual ranking of fixed skills across readers is already a
direct empirical object. [[skillmaster]] also supplies an aggregate
retrieval-withdrawal control after skill-guided training. The remaining
question is narrower: for a named target student, does its own contextual
ordering over exact skill artifacts predict their no-context acquisition
ordering after reset-from-base, matched training? If studied, retain three
quantities separately: source-context utility, frozen-target-context utility,
and target no-context utility after matched training.

The broad executor-versus-teacher mismatch is not novel. [[lgtm-student-level-kd]]
already uses student validation influence to train a better teacher,
[[personalized-teacher-selection]] routes prompts using student likelihood,
and [[distillation-traps-guards]] directly changes downstream distillability
while preserving teacher task utility. Those signals are predictors/baselines;
the remaining object is the ranking and selection regret of fixed,
human-readable procedural artifacts across reader and placement.

The final ACL 2026 check tightens this again: [[smartad]] performs
student-NLL selection of successful tool-agent trajectories and weights
action/final spans during SFT, while [[informative-alignment-rsr]] predicts
post-training performance across 11 teachers and five students. Any later
training phase must compare against both; neither estimates the student's
immediate causal payoff from an external action.

The self-distillation cluster in
[[self-distillation-cluster-update-2026-07-17]] adds three more mandatory
baselines and a capability gate. [[opsd-self-distilled-reasoner]] is the
verified-solution, frozen-teacher, on-policy baseline;
[[sdft-continual-learning]] is the same-model demonstration/skill
internalization and forgetting baseline; and [[sdpo-rich-feedback]] is the
rich-feedback self-teaching baseline. Before allocating training compute,
measure whether privileged context improves the target reader's task-level
teacher behavior. A nonzero KL or a stronger source model is not sufficient.

For any later training comparison:

1. include direct target SFT/RL and the applicable unconditional
   OPSD/SDFT/SDPO arm;
2. cross or pre-register frozen versus EMA/synchronized teachers and any
   outcome-reward mixture, because the papers disagree by regime;
3. stratify by target capability and preserve weak-reader failures rather
   than averaging them away; and
4. record SDFT's method-custody discrepancy explicitly: the paper says reverse
   KL, while the official repository says all headline runs used forward KL
   on student/on-policy prefixes.

These methods test acquisition after a privileged view. They do not replace
the no-training forced-action panel, and SDPO's feedback-conditioned token
log-ratio must not be presented as causal external-action value.

Direct OPCD from `teacher + skill`, PromptKD, and SEED are mandatory
baselines. A
teacher-first-internalization stage is justified only if it beats direct
OPCD; the large teacher is justified only if it beats direct student
internalization and matched task RL. Frozen versus synchronized teachers is a
factor to cross, not a settled design choice: OPCD and continual fact writing
favor a frozen teacher in their regimes, while Skill-SD and SEED obtain value
from synchronization during joint RL.

No skill-lifecycle experiment was launched in this audit.
[[research-question-recommendation-2026-07-17]] ranks the forced-action-value
measurement pilot ahead of this more expensive placement study.

## Kill rule

If the teacher skill gap is absent or the policy cannot improve a pre-
registered cost/accuracy frontier, stop this allocation task and move the
distillation method to a task with observable skill headroom.

## Links

[[opd-skill0-design]] · [[skill-distillation-bridge]] · [[skill0]] ·
[[sdar]] · [[skill1]] · [[alloc-internalization-rung2]] ·
[[compute-elasticity-distillation]] ·
[[compute_elasticity_handoff_2026-07-17/README]] ·
[[skill-lifecycle-research-snapshot-2026-07-17]] · [[opcd]] ·
[[promptkd]] · [[seed-self-evolving-opd]] · [[skillgen-verified]] ·
[[skillmaster]] · [[research-question-recommendation-2026-07-17]] ·
[[self-distillation-cluster-update-2026-07-17]]
