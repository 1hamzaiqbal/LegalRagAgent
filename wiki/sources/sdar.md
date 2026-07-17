---
title: SDAR — Self-Distilled Agentic Reinforcement Learning
type: source
tags: [agentic-rl, self-distillation, opd, gap-gating, skills]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.15155
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.15155.pdf
authors: Lu et al.
year: 2026
code: https://github.com/ZJU-REAL/SDAR
---

# SDAR: Self-Distilled Agentic Reinforcement Learning

## TL;DR

SDAR combines task-reward GRPO with token-level on-policy self-distillation
from a privileged, skill-augmented teacher branch. Its most important result
for us is negative: standalone OPSD nearly collapses on Search-QA, and naive
GRPO+OPSD can degrade sharply—especially for the 1.7B policy. The paper’s
gap-gated objective downweights tokens where the privileged teacher assigns
lower probability than the student.

## Method and evidence

- Student and teacher are branches of the same policy; only the teacher gets
  privileged skill context.
- The per-token gap is `teacher_logp - student_logp`. A sigmoid gate with
  temperature controls whether the dense teacher signal is trusted.
- The reported configuration uses a small distillation weight (`lambda=0.01`)
  and gap-gating coefficient `beta=5`, alongside task RL rather than instead of
  it.
- The combined method improves ALFWorld, Search-QA, and WebShop and requires no
  skills at inference.

## Consequence for our work

The three-step EIT smoke proves our teacher-logprob/student-update plumbing,
not the scientific safety of the bare objective. E3 must not scale the current
ungated OPD recipe as-is. The minimum credible next arm is task reward plus
gap-gated OPD, compared against task reward alone, bare OPD, and skill-context
baselines. E2 must first show that skill context makes the teacher better on
our allocation task.

## Differentiation

Cross-scale legal retrieval distillation is still a different setting, but
“self-distilled skill internalization” is not novel. Our defensible object is
reader-conditioned retrieval control and evidence utility, with a smaller
student and explicit cost/accuracy outcomes.

## Links

[[skill0]] · [[skill1]] · [[opd-skill0-design]] ·
[[skill-distillation-bridge]] · [[three-dial]]
