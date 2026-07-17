---
title: AdaSkill — When Is Skill Distillation Beneficial?
type: source
tags: [skills, metrics, multi-agent, utility, cost]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.01608
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.01608.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/AdaSkill
authors: Xu et al.
year: 2026
---

# AdaSkill

## TL;DR

AdaSkill's “distillation” converts a multi-agent system into external textual
skills and tools, not model weights. It shows that the same skill can help or
hurt depending on the evaluation metric and proposes Metric Freedom as a
predictor. Utility must therefore index or freeze the metric.

It ranks metrics, not alternative artifacts, and has no context-to-weight arm.

## Metric and evidence

Metric Freedom is `F = 1 - r`, where `r` is a Spearman Mantel correlation
between pairwise behavioral distance and score distance. The operating point
uses six questions by six approach-seeded base runs and costs $6.12.

- Across four tasks, 11 datasets, and six metrics, Output-Freedom versus
  headroom-normalized lift has `r=-0.85`, `p=3.1e-5`; Trace-Freedom has
  `r=-0.77`, `p=4.9e-4`.
- A GPT-5.1 replication gives `r=-0.71`, `p=.001` and `r=-0.79`,
  `p=.0001`.
- On causal estimation, the same distilled system changes Textbook MSA from
  .615 to .897 (+28.2 points) but Real MRE from .183 to .159 (-2.4 points).
- Reported execution cost is 1.4–8x lower and latency up to 15x lower than the
  originating multi-agent systems.

## Novelty boundary

AdaSkill occupies metric-conditioned textual-skill utility and demonstrates a
within-task-family reversal. Its GPT-5.1 replication reuses skill definitions,
but each task/system has one AdaSkill artifact; there is no same-task
candidate-artifact ordering. `F` is also computed from a particular base
reader's outputs, so it is reader/distribution dependent despite the stronger
“metric-level” wording.

For [[skill-lifecycle-research-snapshot-2026-07-17]], freeze the scoring metric
or index the tensor as `U(reader, artifact, placement, metric)`. Reported costs
exclude internalization and amortized lifecycle cost.

## Audit cautions

Theorem 3.1 states a bound proportional to `rho`, while its appendix proves a
form proportional to `1 + 2*rho` and calls it equivalent up to a universal
factor; that ratio is unbounded as `rho` approaches zero. The matching lower-
bound theorem states 1/2 but derives 1/4. Treat the empirical correlation as
usable and the current theoretical justification as unresolved.

The v3 PDF and README also diverge on when Stage 2 is enabled and on the
diagnostic sample count. Use the PDF as method truth pending reconciliation.

## Code custody

- Official repository: https://github.com/Tencent/AdaSkill.
- EIT checkout pinned at
  `592223bdc6dfd13f58db915a21c33efe7906354f` on 2026-07-17; MIT license.
- Code and benchmark runners are present, but source result logs are absent.
- PDF SHA-256:
  `bb0242a97a25bfd16b2f412f00445351c7869e294c03704868a59264008ea085`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillsbench]] ·
[[skillaudit]] · [[skill-usage-in-the-wild]]
