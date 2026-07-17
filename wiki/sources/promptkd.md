---
title: PromptKD — Student-Friendly Knowledge via Prompt Tuning
type: source
tags: [knowledge-distillation, prompt-tuning, student-friendly, adaptive-teaching]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2402.12842
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2402.12842.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/PromptKD
authors: Kim et al.
year: 2024
---

# PromptKD

## TL;DR

PromptKD alternates between a student and seven learned soft-prompt tokens
prepended to a frozen teacher. Student-generated responses guide the prompt to
make the teacher's distribution more student-like; the prompted teacher then
distills back into the student. It directly establishes student-conditioned
teacher-context optimization and prompt-to-weight transfer for generative
language models.

This is a high-severity conceptual collision with any claim that optimizing
teacher context for student absorbability is new. It does not compare
execution-optimized, human-readable procedural skills with their contextual
or post-withdrawal utility across readers.

## Method

For each student-generated pseudo-target, PromptKD:

1. minimizes forward KL from the prompted teacher to the current student to
   update the soft prompt;
2. initially regularizes the prompted teacher toward the unprompted teacher,
   with the regularization coefficient decaying to zero; and
3. minimizes reverse KL from the student to the newly prompted teacher.

The teacher's weights remain frozen. Prompt and student co-adapt, so the method
does not evaluate a fixed independently optimized artifact.

## Evidence

- The paper trains on Dolly and evaluates Dolly plus SelfInst, Vicuna, S-NI,
  and UnNI across GPT-2, OPT, and Llama teacher/student pairs.
- For GPT-2 XL → Large, the mean of five reported ROUGE-L cells is 25.08 for
  PromptKD, 24.18 for MiniLLM, 22.62 for GKD, and 23.34 for the teacher.
- PromptKD is best or tied in 13 of 15 GPT-2 size-by-dataset cells, but not
  every cell. For Llama-13B → 7B, it beats MiniLLM in four of five cells and
  trails by 0.1 on S-NI.
- On OPT-13B → 6.7B, reported allocated memory/training time are 43.62 GB and
  26.97 hours, versus 68.91 GB and 85.71 hours for MiniLLM. GKD is slightly
  cheaper at 41.99 GB and 25.37 hours.
- Table 4 is especially relevant: the student-friendly prompt changes teacher
  validation ROUGE-L from `29.695 to 26.893` for GPT-2 XL, `31.603 to 31.933`
  for OPT-13B, and `35.116 to 35.168` for Llama-13B. Student adaptation can
  therefore reduce the teacher's own measured execution quality in at least
  one capacity-gap regime.

## Novelty boundary

PromptKD already owns:

- adapting teacher context using the named student's distribution;
- making teacher signals easier for a student to absorb;
- alternating prompt and student updates on student-generated sequences; and
- removing the teacher prompt at student deployment.

It does not rank multiple natural-language skills by source execution,
frozen-target context use, and post-withdrawal learning; compare context,
permanent weights, and modular adapters; test skill portability across
families; or measure task reward, costly actions, and target-student regret.

The defensible [[skill-lifecycle-research-snapshot-2026-07-17]] question is
therefore not “can teacher context be optimized for a student?” It is whether
an **independently execution-optimized, human-readable procedural artifact**
preserves its ordering when used by another reader or as a withdrawn training
scaffold. PromptKD is a mandatory student-friendly-teacher baseline.

## Limits

The tasks are single-turn instruction following rather than procedural agent
work. Evaluation is largely ROUGE-L plus a limited GPT-4 preference study.
The prompt is continuous and model-bound rather than inspectable/portable
natural language. There is no direct task reward, action trace, context-token
deployment comparison, fixed-artifact ranking study, or sequential retention
test.

## Code custody

- ACL paper: https://aclanthology.org/2024.findings-emnlp.364/.
- Official project: https://promptkd.github.io/.
- Official repository: https://github.com/gmkim-ai/PromptKD.
- Persistent EIT checkout pinned at
  `383182f4cb66e005b789ed1c8585a68492f300c0` on 2026-07-17.
- PDF SHA-256:
  `931685d79bda73d003159d63f4160223e707ae15810a9f7c9750ab463cd64bed`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[opcd]] · [[skill-sd]] ·
[[token-teachability]] · [[continual-facts-in-weights]]
