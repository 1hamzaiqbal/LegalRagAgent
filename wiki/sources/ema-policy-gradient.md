---
title: EMA Policy Gradient - EMA Anchor and Top-k KL
type: source
tags: [opd, kl, policy-gradient, top-k, ema, distillation]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2602.04417v1
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2602.04417.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/ema-pg
authors: Zhang and Ba
year: 2026
---

# EMA Policy Gradient

## TL;DR

EMA-PG makes the estimator distinction our OPD code must state precisely. For
a student-sampled token, K1 is an unbiased reverse-KL **value** estimate, but
naive direct autodiff through K1 averages to zero. Our detached
teacher-minus-student log-probability gap multiplied by the student
log-probability is instead the corresponding score-function gradient,
equivalent to the paper's K4/r-trick only when on-policy, ungated, and
unclipped.

The paper's Top-k estimator computes exact dense KL on the distribution's
high-mass head and uses a sampled tail correction. It is a principled later
variance ablation, not a new contribution for us. Its EMA anchor changes the
reference into a lagged student and therefore does not replace a separately
trained teacher in the source-transfer question.

## K1 value versus K4 gradient

At a fixed prefix, with student/current policy `p`, teacher/reference `q`, and
token `Y ~ p`:

\[
K1(Y)=\log p(Y)-\log q(Y),
\qquad E[K1]=D_{KL}(p\|q).
\]

Holding the sampled token fixed, however,

\[
E[\nabla K1(Y)]=E[\nabla\log p(Y)]=0.
\]

K4 restores the score-function term with
`r=p/stopgrad(p)` and `K4=r stopgrad(K1)`. It has unbiased values and
gradients in the paper's on-policy, unclipped setting. Our surrogate has the
same expected gradient. Positive-gap gating, advantage clipping, or clipped
stale-policy ratios deliberately give up that strict unbiasedness.

Anchors: Table 1 and Algorithms 1-2 on PDF p. 2; Section 5.2, pp. 5-6;
Appendix E.1-E.5, pp. 24-26. A finite sampled K1 mean may be negative even
though exact KL is nonnegative.

## Top-k KL

For reverse KL, Top-k uses the student's top-k indices. It exactly sums the KL
head and adds a K4-style correction only when the sampled token is outside the
head. At `k=0` it reduces to sampled K4; at vocabulary size it becomes exact
KL. Intermediate `k` retains `O(k)` logit state while preserving unbiased
values and gradients if the tail importance ratio is not clipped.

The official implementation normally uses `k=32`. Top-k reduces retained
logit/transfer memory, not necessarily the full vocabulary-projection compute.
For forward KL, top-k indices instead come from the teacher/reference and the
tail uses the paper's K5 estimator.

Our existing HTTP protocol returns only the teacher probability of the sampled
student token. A real Top-k arm additionally needs student top-k IDs, teacher
scores on arbitrary head IDs, both normalizers, and the sampled-tail score.
Calling the current path Top-k would therefore be incorrect.

## EMA anchor and results boundary

EMA-PG updates a reference copy as
`theta_ema <- eta theta_ema + (1-eta) theta`. Under the paper's local Fisher
approximation, stability requires `alpha beta lambda_max < 1 + eta`. This is a
lagged-policy regularizer, not an independently skilled teacher.

The paper reports agentic-search average reward `0.312` for GRPO, `0.348` with
EMA alone, and `0.416` with EMA plus Top-k reverse KL. It reports OlympiadBench
`50.8% -> 53.9%` and math average Pass@1 `50.3% -> 52.8%`. The released math
script at the pinned commit uses K3/`low_var_kl` plus EMA rather than Top-k, so
the math table is EMA evidence; the explicit Top-k ablations are primarily on
search. These are paper results, not LegalRagAgent results.

## Consequence and differentiation

1. Keep the current `k=0` pipeline, but name the value and gradient separately.
2. Compare `k=0`, `k=16`, and `k=32` only after the task-reward pipeline works.
3. Do not add EMA to the first source matrix; it changes the teacher identity.
4. Do not claim Top-k OPD as novel. The paper explicitly points to knowledge
   distillation, and its README describes on-policy tail-corrected and offline
   truncated variants.

EMA-PG clarifies rather than displaces our question. It studies estimator and
anchor stability; we hold a potentially mismatched trained teacher fixed and
ask when its signal helps or harms a particular student. Teacher usefulness,
not the existence of a KL estimator, must carry the contribution.

## Version and custody

- Audited paper: arXiv v1, submitted 4 February 2026.
- PDF SHA-256:
  `7d00df62c956e13d90ff2158c844dc9176ec243ff55807a81f75069b76bc4328`.
- Official MIT-licensed repository: https://github.com/LunjunZhang/ema-pg.
- EIT checkout pinned at
  `f9b977632ad948ce935432a517c22291bfaba562` on 2026-07-17.

## Links

[[opd-math-source-transfer]] · [[verl-opd-trainer]] · [[sdar]] ·
[[opd-distillation]] · [[opsd-self-distilled-reasoner]]
