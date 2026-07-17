---
title: SkillRevise — Execution-Grounded Skill Revision
type: source
tags: [skills, revision, cross-model, contextual-utility, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.01139
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.01139.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/skillrevise
authors: SkillRevise contributors
year: 2026
---

# SkillRevise

## TL;DR

SkillRevise iteratively revises task-specific external instructions against
executor feedback. Its transfer experiment fixes the final GPT-5.5-selected
artifact for each task and applies it to four other executors. The artifacts
remain beneficial in aggregate, but target-conditioned revision is better for
every target.

This is evidence for imperfect portability of a selected final artifact, not
for cross-reader candidate-ordering regret. Target readers never score the
source's version sequence, and no artifact is internalized into weights.

## Method

For each task, SkillRevise starts from a cold instruction artifact and may
produce versions `v0` through `v3`. GPT-5.5 authors/revises the artifact, an
executor attempts the task, and a verifier checks the result. Selection is
success-prioritized: the first verifier-passing version is retained; the
paper's utility rule is a fallback when no version passes.

The main evaluation covers 206 tasks and five executors. At revision `v3`,
overall successes versus no skill are:

| Executor | No skill | Rev `v3` |
|---|---:|---:|
| GPT-5.5 | 79 | 115 |
| Opus 4.7 | 55 | 100 |
| Kimi 2.6 | 47 | 86 |
| Qwen 3.6 Plus | 33 | 77 |
| DeepSeek V4 Pro | 49 | 95 |

## Fixed-artifact transfer

The transfer table conditions on the 57 tasks where the GPT-5.5 source
succeeds. Each target receives either no skill, its own target-conditioned
`v3`, or the fixed GPT-5.5-selected final artifact.

| Target | No skill | Own `v3` | GPT-5.5 artifact |
|---|---:|---:|---:|
| Opus 4.7 | 16/57 | 41/57 | 33/57 |
| DeepSeek V4 Pro | 14/57 | 38/57 | 27/57 |
| Kimi 2.6 | 16/57 | 33/57 | 19/57 |
| Qwen 3.6 Plus | 6/57 | 26/57 | 11/57 |

The transferred artifact beats no skill by 17, 13, three, and five successes.
Target-conditioned revision beats transfer by eight, 11, 14, and 15
successes, equivalent to 14.0–26.3 percentage points on this 57-task subset.

## Exact claim boundary

SkillRevise owns source-selected artifact portability and the value of
target-conditioned revision. The 14.0–26.3-point gaps are adaptation gaps,
not source-selection regret: the targets do not evaluate the GPT-5.5
candidate sequence, so the study cannot say which source version each target
would rank first.

Conditioning on source-success tasks also makes this a portability test, not
an all-task generalization estimate. The Principle Memory is external text;
there is no weight update, withdrawal, retention, or matched direct-learning
control.

## Artifact gaps and limitations

The official repository contains code and benchmark bundles but not the final
selected artifact bytes or result logs underlying the cross-model table.
Results are reported as aggregate counts without independent-seed uncertainty.
Reconstructing selection transport would require every versioned candidate,
its verifier decision, and crossed target outcomes on a common task set.

## Code custody

- Audited arXiv v3 PDF:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.01139.pdf`.
- PDF SHA-256:
  `7adbda6d5a5f2d99fcd2437e5cebb687c9a9650123cba10cc6e0edc85a6e2e41`.
- Official repository: https://github.com/HKUST-KnowComp/skillrevise.
- EIT checkout pinned at
  `25569d8859b9c9abe121f35b3de5dc99356e364e` on 2026-07-17.
- Repository license: MIT.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillgen-verified]] ·
[[skillopt]]
