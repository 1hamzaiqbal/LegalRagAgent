---
title: "From Entropy to Epiplexity"
type: source
tags: [epiplexity, data-value, mdl, prequential, requential, opd]
created: 2026-07-24
updated: 2026-07-24
status: primary paper and official code archived on EIT; integration boundary recorded
---

# From Entropy to Epiplexity

Primary source: [arXiv 2601.03220](https://arxiv.org/abs/2601.03220)  
Official code: [shikaiqiu/epiplexity](https://github.com/shikaiqiu/epiplexity)

## Durable primary-source custody

- PDF: `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/2601.03220.pdf`
- PDF SHA-256:
  `8c67d250b0507f341bf3bd91961c28ebde33290c8548f4af48d0e5683699488c`
- code: `/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/epiplexity`
- code commit: `3aa12a1be6a413fe9eaa41374a6a46a4a0d4e100`

## What the paper actually gives us

Epiplexity is structural information extractable by a computationally bounded
observer. It is observer- and compute-dependent. The same data can therefore
have different extractable structure for different students or budgets, which
is directly compatible with our reader-conditioned framing.

The official code exposes two practical estimates:

1. `K_auc`: a prequential approximation, computed from the area of a learning
   curve above its terminal loss;
2. `K_req`: a requential code length, computed as cumulative full-distribution
   teacher-to-student KL while a student learns from teacher-generated tokens.

The paper recommends prequential coding for inexpensive, rough rankings and
requential coding for a more faithful estimate. Neither quantity alone
guarantees improvement on a particular downstream target: high extractable
structure can be irrelevant to the target.

## Boundary for our OPD work

We must not rename an arbitrary OPD loss "epiplexity."

- The pinned upstream OPSD positive control uses full-vocabulary
  `KL(teacher || student)` at `beta=0`, but on student-generated continuations,
  with a per-token divergence clip. Its logged loss is therefore an
  **OPSD divergence signal**, not a requential code length.
- A requential-compatible estimate requires teacher-generated tokens,
  full-distribution `KL(teacher || student)`, explicit token weighting, no
  silent clipping, and cumulative bits under a fixed observer/budget.
- A prequential estimate requires a declared teacher-forced likelihood curve
  and terminal likelihood on the same source distribution. Reward curves or
  task accuracy are not substitutes.
- Cross-benchmark value remains a separate outcome. The defensible empirical
  question is whether these source/observer diagnostics predict marginal
  student improvement per unit cost, not whether they are identical to value.

## Research opportunity

For source dataset `d`, student `s`, teacher `t`, target `b`, and budget `c`,
measure a source-characterization vector before training and the matched
causal outcome afterward:

`V(d,s,t,b,c) = improvement(OPD) - improvement(matched control)`.

The characterization should include exact requential and prequential
quantities where admissible, plus student support, task relevance,
teacher-student disagreement, output length, and compute. The initial
hypothesis is not monotonic "more KL is better": very low divergence may
contain little new signal, while very high divergence may be inaccessible to
the bounded student. An intermediate learnable-structure regime, modulated by
target alignment, is the more plausible pattern to test.

## Links

[[opd-data-value-design-2026-07-24]] · [[opd-math-source-transfer]] ·
[[opsd-self-distilled-reasoner]]
