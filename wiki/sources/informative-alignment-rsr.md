---
title: Informative Alignment — Rank-Surprisal Ratio for Student-Specific Teaching
type: source
tags: [distillation, teachability, trajectory-selection, teacher-selection, metric]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://aclanthology.org/2026.acl-long.1950/
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/acl_2026.acl-long.1950.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/RankSurprisalRatio
authors: Yang et al.
year: 2026
---

# Which reasoning trajectories teach students better? — RSR

## TL;DR

This paper is a major boundary for any “best teacher depends on the student”
or cheap teachability-metric claim. It evaluates reasoning trajectories from
11 teachers on five target students and proposes Rank–Surprisal Ratio (RSR), a
student-conditioned balance of token rank and surprisal. RSR's dataset-level
score has mean absolute Spearman correlation 0.86 with post-training reasoning
performance and supports both trajectory and teacher selection.

RSR predicts **training utility**. It does not measure the named student's
immediate counterfactual payoff from an external action, so it does not close
the forced-action transport question. For the secondary skill study, RSR is a
mandatory acquisition-utility predictor, but the paper never compares one
fixed procedural artifact as runtime context and as a withdrawn training
scaffold.

## Method

For teacher trajectory `x` and target student, RSR combines average token-wise
rank with average negative log-likelihood. Its preferred trajectories are
surprising enough to contain learning signal while their tokens remain
relatively high-ranked under the student. The implementation clips token ranks
and computes both sample- and dataset-level values.

The main panel contains 11 teacher models and five student models—55
teacher/student pairings—with three independently generated teacher datasets
per teacher. The students span Qwen 2.5/3 and Llama 3.1 checkpoints. The study
then trains each student and measures downstream math-reasoning performance.

## Evidence

- Dataset-level mean absolute Spearman correlation with post-training
  performance is 0.86 for RSR versus 0.23 for teacher performance.
- For Qwen2.5-7B, QwQ-32B trajectories teach to 52.0 while DeepSeek-R1 reaches
  47.3; the strongest teacher is not automatically the best teacher.
- For Qwen3-4B, Qwen3-4B-Thinking teaches to 61.9 while DeepSeek-R1 reaches
  55.8.
- Teacher selection from only 200 sampled trajectories reaches 48.3 versus
  48.7 for the paper's post-training oracle.
- RSR-selected 5,000-example trajectory sets are best among the reported
  selection methods for all five students in the main selection table.

## Exact claim boundary

RSR occupies student-specific teacher/trajectory suitability, post-training
gain prediction, and cheap selection among candidate teaching traces. Student
likelihood alone, teacher performance, and generic “teachability” cannot be
claimed as new.

It does not contain matched forced internal/external arms, a canonical payload
crossed over readers, action price/cost, immediate reader-specific action
value, or target regret from following a teacher's forced-outcome oracle. It
therefore separates rather than collapses the three objects:

1. immediate acting utility;
2. compatibility/absorbability of a teacher trace;
3. realized post-training utility.

In any later training experiment, RSR must be compared with target likelihood,
[[token-teachability]], [[smartad]], validation influence, and direct measured
training gain.

## Code and data custody

- Audited ACL PDF:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/acl_2026.acl-long.1950.pdf`.
- PDF SHA-256:
  `2dbfbd1e65b74360e3557dea652f782ad633f15f609f8721e072c9b774c11b12`.
- arXiv version: https://arxiv.org/abs/2601.14249.
- Official repository: https://github.com/UmeanNever/RankSurprisalRatio.
- EIT checkout pinned at
  `59a7c4cdbbb7c26b93f91472da5d79498e29b5c0` on 2026-07-17; MIT license.
- Official dataset: https://huggingface.co/datasets/Umean/RSR_data, observed at
  commit `8f24ed4aa8a69173b04833e5adba7e8b9413e933` on 2026-07-17. The full
  Hugging Face payload is linked and pinned here but not mirrored in EIT.

The repository contains the cleaned RSR computation code, not the complete
training/evaluation harness. The separate dataset release contains the 33
teacher trajectory datasets and five student-selected releases described by
the project page.

## Links

[[research-question-recommendation-2026-07-17]] ·
[[skill-lifecycle-research-snapshot-2026-07-17]] · [[smartad]] ·
[[personalized-teacher-selection]] · [[distillation-traps-guards]]
