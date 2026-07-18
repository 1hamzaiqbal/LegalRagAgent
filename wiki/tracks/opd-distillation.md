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

The literature gate now includes [[sdar]], [[skill1]], and the audited
[[self-distillation-cluster-update-2026-07-17]]. Broad context-to-weights
self-distillation is occupied: [[opsd-self-distilled-reasoner]],
[[sdft-continual-learning]], and [[sdpo-rich-feedback]] cover verified
solutions, demonstrations, rich feedback, and interaction history. Bare OPD
remains an unsafe diagnostic rather than a primary arm. The isolated
[[opd-math-source-transfer]] branch now implements and CPU-tests grouped
verifiable task reward, its matched `task_rl` baseline, and the gap-gated
score-function OPD auxiliary. No task-performance claim exists until EIT
quality gates and held-out evaluation pass.

## Gates

1. **E2 teacher skill gap:** skill-augmented teacher must outperform the same
   teacher without the skill on a held-out allocation/control task.
2. **Objective safety:** real E3 uses task reward plus gap-gated OPD. Bare OPD
   remains a collapse diagnostic; loss finiteness is not success.
3. **Outcome relevance:** evaluation is task accuracy and cost, including the
   BarExam/70B `llm_only` decision—not imitation loss alone.
4. **Capability gate:** privileged context must improve the named target's
   task-level teacher behavior before self-distillation; nonzero KL is not
   sufficient.

## E3 arms if E2 passes

1. outcome-label/task-RL baseline;
2. task RL + gap-gated OPD from a skill-augmented teacher;
3. bare OPD diagnostic;
4. student with skill in context but no training;
5. trace KD for the closed-teacher comparison.
6. applicable unconditional OPSD/SDFT/SDPO baseline from the same checkpoint,
   with the implemented KL direction and teacher update recorded exactly.

Keep acting utility, privileged-view teacher quality, and post-withdrawal
acquisition utility separate. SDPO's feedback-conditioned log-ratio is a
training signal, not causal external-action value. SDFT's paper says reverse
KL, while its repository states that the headline runs used forward KL on
student/on-policy prefixes.

## Candidate task extension

[[effort-conditioned-resource-allocation]] could provide a richer E2/E3 task
than the current single-turn allocation labels: a privileged teacher chooses
among thinking, search, verification, and stopping under an explicit cost
condition, and a smaller student is evaluated on the same cost/accuracy
frontier. This does not relax the gates. Direct cost-conditioned task RL is the
primary baseline; task RL plus gap-gated OPD must beat it; and bare OPD remains
a collapse diagnostic.

The isolated [[opd-math-source-transfer]] child track tests a narrower
prerequisite: whether the teacher-training source changes distillation value
after controlling teacher task quality and exact item overlap. It is a method
characterization surface, not evidence that retrieval skills were internalized.

## Kill rule

If the teacher skill gap is absent or the policy cannot improve a pre-
registered cost/accuracy frontier, stop this allocation task and move the
distillation method to a task with observable skill headroom.

## Links

[[opd-skill0-design]] · [[skill-distillation-bridge]] · [[skill0]] ·
[[sdar]] · [[skill1]] · [[alloc-internalization-rung2]] ·
[[self-distillation-cluster-update-2026-07-17]] · [[opd-math-source-transfer]] ·
[[ema-policy-gradient]] · [[verl-opd-trainer]]
