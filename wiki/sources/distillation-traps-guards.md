---
title: Distillation Traps and Guards — Controlling LLM Distillability
type: source
tags: [knowledge-distillation, distillability, calibration, teacher-student-gap]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.18963
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.18963.pdf
authors: Zhan et al.
year: 2026
---

# Distillation Traps and Guards

## TL;DR

This paper explicitly separates a teacher's task utility from its downstream
distillability. It reinforcement-fine-tunes teacher weights to make them more
or less distillable while anchoring task performance, producing large changes
in student outcomes without corresponding teacher-performance changes.

It is the strongest collision with the broad “execution ability is not
teaching ability” framing. It does not select or transport fixed textual skill
artifacts or compare their runtime and post-withdrawal utility.

## Method

Teacher RFT combines:

1. task reward;
2. a KL anchor to the original teacher; and
3. a cross-tokenizer sequence log-probability-ratio reward against a smaller
   calibration model.

Changing one coefficient steers the teacher toward distillable or deliberately
undistillable behavior. The teacher/student pairs include Gemma-3-12B with a
Gemma-3-1B calibration proxy and Gemma-3-4B downstream student, and Qwen3-8B
with Qwen3-0.6B proxy and Qwen3-1.7B student.

## Evidence

- Gemma BM4 downstream student score moves from `0.413` to `0.523` with the
  distillable teacher and to `0.165` with the undistillable teacher.
- Qwen BM4 moves from `0.526` to `0.615` or `0.215`.
- Qwen MMLU-Pro moves from `0.297` to `0.546` or `0.123`.
- Undistillable-teacher task utility remains close or slightly higher: Gemma
  BM4 `0.639 to 0.649` and Qwen BM4 `0.625 to 0.651`.
- Reported teacher calibration costs 141.5 H100 GPU-hours; one on-policy
  GKD-RKL student run costs 196.8 GPU-hours.

One paper-wide characterization deserves caution: the authors say
distillable calibration consistently lowers token reverse KL, but the Gemma
rows increase on BM4 (`0.117 to 0.129`), BM5 (`0.121 to 0.133`), and MMLU-Pro
(`0.321 to 0.331`). Wrong-trace preference does decrease consistently.

## Novelty boundary

Distillability is already a named, controllable teacher property. A new skill
study cannot claim the generic task-utility/teaching-utility mismatch, a first
distillability metric, or teacher calibration as its contribution. The
remaining object is **artifact-level utility transport**: whether the same
inspectable, reusable procedural skill preserves its ordering across readers
and when moved from runtime context to withdrawn training context, and what
selection regret results from choosing it by source execution.

## Limits and custody

The intervention changes teacher weights rather than an artifact. Calibration
uses a proxy model rather than the exact downstream student. The paper does
not rank skill candidates, compare context/weights/adapters, or study agent
actions, cost, or lifecycle updates. No official code repository is linked by
the paper, ACL record, or arXiv page as of 2026-07-17.

- PDF SHA-256:
  `aa7f053181e916caa8813706b54a94992a9a4ea1786bd1bfbf1da1074b493c30`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[promptkd]] ·
[[personalized-teacher-selection]] · [[lgtm-student-level-kd]] ·
[[token-teachability]]
