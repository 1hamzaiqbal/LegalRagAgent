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

## Kill rule

If the teacher skill gap is absent or the policy cannot improve a pre-
registered cost/accuracy frontier, stop this allocation task and move the
distillation method to a task with observable skill headroom.

## Links

[[opd-skill0-design]] · [[skill-distillation-bridge]] · [[skill0]] ·
[[sdar]] · [[skill1]] · [[alloc-internalization-rung2]]
