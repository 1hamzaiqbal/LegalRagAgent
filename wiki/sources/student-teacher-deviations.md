---
title: "On Student-Teacher Deviations in Distillation: Does It Pay to Disobey?"
type: source
tags: [knowledge-distillation, student-teacher-deviation, implicit-bias, novelty-boundary]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2301.12923
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2301.12923.pdf
code: no official code located as of 2026-07-17
authors: Vaishnavh Nagarajan, Aditya Krishna Menon, Srinadh Bhojanapalli, Hossein Mobahi, Sanjiv Kumar
year: 2023
venue: NeurIPS 2023
---

# On Student-Teacher Deviations in Distillation

## TL;DR

This paper already owns the exact “does it pay to disobey?” title and the
broad observation that a student can systematically deviate from—and
outperform—its teacher. It studies ordinary predictive knowledge distillation,
not agent actions. Its main mechanism is confidence and implicit-bias
exaggeration under gradient descent. We must cite it and avoid the slogan, but
it does not estimate model-specific external-action value, action price, or
teacher-to-student policy regret.

## Exact question

Why can a distilled classifier fail to match its teacher's probability
distribution yet generalize better? Are deviations arbitrary optimization
failure, or systematic consequences of training dynamics?

The empirical diagnostic compares the teacher and student's probabilities on
the teacher's predicted class after a logit transform. Low-confidence
underfitting or high-confidence overfitting produces a fitted slope larger
than one and is described as confidence exaggeration.

The theoretical core is a linear gradient-flow result. If teacher learning
along data eigendirections is filtered by `A(t) = I - exp(-tXX^T)`, then the
distilled student receives the composed filter
`A(student_time) A(teacher_time)`. At matched progress on a high-eigenvalue
direction, the student has progressed less on a low-eigenvalue direction. In
that controlled setting, distillation exaggerates gradient descent's
preference for top eigendirections.

## Evidence

The study spans more than 25 self- and cross-architecture settings:

- CIFAR-10/100, noisy CIFAR-100, Tiny-ImageNet, and ImageNet;
- ResNet-20/56, ResNet-18/50, and MobileNet-v2;
- MNLI, IMDB, QQP, and AGNews;
- RoBERTa Small, Medium, and Base.

Key reported patterns:

- all 18 image settings show low-confidence underfitting on test data, while
  13/18 show it on training data;
- language results are less uniform: only 7/12 reported test slopes exceed
  one, and several cross-architecture cases weaken the clean story;
- CIFAR-100 ResNet-56 self-distillation improves `72.52% → 74.55%`;
- noisy CIFAR-100 improves `69.8% → 72.7%`;
- ImageNet and language results are mixed, so systematic deviation is not a
  guarantee of better generalization;
- switching between KD and one-hot loss changes training dynamics differently
  depending on whether the teacher interpolates the training labels.

## Limitations

The paper itself notes that its pointwise diagnostic examines only the
teacher's top-1 class, its theorem is limited to linear gradient flow, and it
does not exhaustively characterize when the bias improves generalization.
Additional cautions:

- most accuracy rows are single runs without uncertainty;
- the fitted slope is an imperfect summary and sometimes misses visually
  apparent language effects;
- cross-architecture language settings are precisely where the exaggeration
  narrative is least consistent;
- controlled examples support a mechanism, but do not establish it as a
  universal causal explanation.

## Bearing on our work

We cannot claim that students beneficially deviate from teachers, that exact
teacher matching is always desirable, or that systematic “disobedience” is a
new observation. We also should not use the paper's title as our title.

The useful transfer is methodological:

1. plot itemwise teacher versus student **action advantage**, not only action
   agreement;
2. separate ordinal transport (rank) from cardinal transport (scale and
   threshold);
3. partition the four regimes: both benefit, student-only benefit,
   teacher-only benefit, neither benefits;
4. include same-capacity/self-distillation controls to separate
   optimization-induced boundary drift from genuine capability mismatch;
5. test whether an affine, monotone, or isotonic map recovers student values
   from teacher values using a small calibration set.

The remaining claim in
[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] is
about executable external actions and target-student counterfactual outcomes,
not predictive probability deviation.

## Code custody

No official repository was linked from the paper, arXiv record, NeurIPS
proceedings page, or a targeted title/author search on 2026-07-17. Record this
as **no official code located**, not an incomplete download.

## Raw source

EIT PDF `papers/arxiv_2301.12923.pdf`.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] ·
[[action-value-transport-reading-packet-2026-07-17]] · [[rethinking-opd]]
