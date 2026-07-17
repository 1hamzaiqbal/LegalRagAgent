---
title: SkillLens — Reader Heterogeneity for Fixed Skills
type: source
tags: [skills, cross-model, contextual-utility, evaluation, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.23899
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.23899.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/SkillLens
authors: SkillLens contributors
year: 2026
---

# SkillLens

## TL;DR

SkillLens studies how experience quality, extractor, target model, and domain
shape the value of external skills. Its clean fixed-artifact experiment
crosses the same strong-pool and weak-pool SpreadsheetBench skills over six
target readers. Effects vary substantially, including negative weak-skill
effects, but the strong artifact beats the weak artifact for all six readers.

This establishes fixed-skill reader heterogeneity, not a rank reversal. The
larger extractor × target analysis is not a fixed-artifact cross because each
target contributes a different experience pool and therefore changes the
skill being evaluated.

## Design

The main framework lets target model `M` generate experience and consume the
resulting skill, while extractor `E` may differ. Across five domains, six
targets, and five extractors, the object is `S_(E,M,D)`: changing the target
also changes its experience pool and the artifact.

Section 5.3 supplies the cleaner control. A fixed GPT-5.4-Mini extractor
creates two SpreadsheetBench skills: one from a strong GPT-5.4 experience
pool and one from a weak Qwen3.5-9B pool. The identical two artifacts are then
given to all six targets.

## Fixed-artifact evidence

| Target reader | Strong-pool skill | Weak-pool skill |
|---|---:|---:|
| GPT-5.4 | +9.0 | -2.0 |
| GPT-5.4-Mini | +3.5 | +1.0 |
| Gemini-3.1-Pro | +1.8 | -1.5 |
| Gemini-3.1-FL | +3.2 | +1.8 |
| Qwen3.5-35B | +9.5 | +3.3 |
| Qwen3.5-9B | +4.0 | +3.2 |

The strong-minus-weak margin ranges from 0.8 to 11.0 points. Because strong
beats weak in all six rows, selecting either artifact with any source reader
cannot induce a cross-reader reversal. The resulting two-artifact
source-selection regret is zero; that is **our calculation, not a paper
claim**.

Across the broader 150 extractor–target–domain cells, 75% of skills help and
25% harm; ALFWorld alone has 47% negative cells. Those rates demonstrate
heterogeneity but cannot be interpreted as a common candidate ordering
because the artifact changes with the target's experience pool.

SkillLens also evaluates 151 within-target skill pairs with large outcome
gaps. A text-only GPT-5.4 judge reaches 46.4% pairwise accuracy overall and
15.8% on the paper's `delta >= 5` subset; a validated three-dimensional rubric
raises overall accuracy to 73.8%. This is evidence about artifact evaluation,
not cross-reader transfer.

## Exact claim boundary

SkillLens occupies external fixed-artifact reader heterogeneity and shows that
weak experience can produce harmful skills. It does not establish
reader-dependent ordering in its clean two-artifact cross, identify a source
selector, compare alternative artifacts after target learning, or place any
skill in weights. Skills are injected as system-prompt text and the target
weights remain frozen.

## Artifact gaps and limitations

The official repository contains code, test pools, and the meta-skill, but not
the main experience pools, the exact strong/weak skill bytes, or the raw result
matrix. Point estimates do not include independent-seed uncertainty. The
fixed-artifact evidence is one domain with two candidates, so it is a required
heterogeneity control rather than a sufficient ranking-transport study.

## Code custody

- Audited arXiv v1 PDF:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.23899.pdf`.
- PDF SHA-256:
  `a5d88b47f668f7b24f091ff817bc7fade594d09fea6c0f57e2457be37bc51ade`.
- Official repository: https://github.com/microsoft/SkillLens.
- EIT checkout pinned at
  `c5ee10f6b566cd2ccf96f7cef115eba59606b01b` on 2026-07-17.
- Repository license: MIT.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillgen-verified]] ·
[[masa]]
