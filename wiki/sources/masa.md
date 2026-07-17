---
title: MASA — Model-Aware Skill Adaptation
type: source
tags: [skills, cross-model, granularity, contextual-utility, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.30723
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.30723.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/MASA
authors: MASA contributors
year: 2026
---

# MASA

## TL;DR

MASA directly shows that fixed external skill variants have reader-dependent
contextual ordering. Its preliminary experiment crosses concise, moderate,
and detailed encodings of the same behavioral principles over seven
Qwen/Gemma readers on one ALFWorld validation split. Preferred granularity
changes with model family and scale, and some skills underperform no skill.

MASA leaves every consumer frozen. A trained Qwen3-4B rewriter emits external
text, so this is contextual adaptation rather than artifact acquisition or
weight internalization.

## Fixed-variant evidence

The three nonempty variants retain the same task coverage and behavioral
principles while changing textual granularity. They are fixed across readers.

| Reader | No skill | Concise | Moderate | Detailed |
|---|---:|---:|---:|---:|
| Qwen3-4B | 17.1 | 16.4 | 20.0 | 12.8 |
| Qwen3-8B | 32.1 | 27.9 | 25.0 | 17.1 |
| Qwen3-14B | 37.9 | 36.8 | 42.1 | 47.5 |
| Qwen3-32B | 36.4 | 40.7 | 41.4 | 42.9 |
| Gemma3-4B | 10.7 | 12.1 | 8.6 | 0.0 |
| Gemma3-12B | 7.9 | 15.7 | 9.3 | 15.0 |
| Gemma3-27B | 21.4 | 36.4 | 35.0 | 44.3 |

Moderate is best for Qwen3-4B; detailed is best for Qwen3-14B, Qwen3-32B,
and Gemma3-27B; concise is best for both smaller Gemma readers. Qwen3-8B's
no-skill score beats all three variants.

Selecting a nonempty variant with one source reader and deploying it to
another creates up to 12.1 points of target regret across the seven readers,
or 10.8 points if source and target are restricted to Qwen. If no skill is an
eligible candidate, the maximum rises to 22.9 points. These regrets are **our
calculations from the reported table, not paper claims**.

## Main method and results

The main MASA search uses frozen Qwen3-4B/8B/14B/32B targets, a
DeepSeek-V4-Pro teacher, hill climbing for general skills, and UCB allocation
for task-specific skills. On ALFWorld, MASA reports 31.4, 57.9, 64.3, and 65.7
for those four targets, improving over the strongest measured baseline by
+4.3, +25.8, +20.0, and +20.7 points respectively. It also ranks highest in
the paper's WebShop comparison.

Those main numbers are model-specific search outcomes, not a common crossed
artifact set. The preliminary granularity table is the relevant transport
evidence.

## Exact claim boundary

MASA occupies fixed-candidate contextual ordering across model family and
scale, including nonzero wrong-reader selection regret. It does not compare
the same artifacts as curricula, update target weights, remove an acquired
skill at deployment, or test retention. There is no matched direct-training
or context-versus-weight baseline.

## Artifact gaps and limitations

The paper reports point estimates without confidence intervals or independent
seeds for the fixed-variant sweep. The repository includes code and
model-specific evolved JSON libraries, but not the exact concise, moderate,
and detailed artifacts or raw run logs behind the preliminary table.

The README labels the project Apache-2.0, but the pinned checkout contains no
`LICENSE` file. Treat reuse rights as unresolved until a license file or
release artifact is supplied.

## Code custody

- Audited arXiv v1 PDF:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.30723.pdf`.
- PDF SHA-256:
  `915452ead8006c64d9aebdb6cbcae06544fb742f21d1afc638928b053cf47dc9`.
- Official repository: https://github.com/jianxiangyu/MASA_.
- EIT checkout pinned at
  `3c289d2c0c93e38c03baa5b279da7f1dd99c64c5` on 2026-07-17.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillgen-verified]] ·
[[skilllens]]
