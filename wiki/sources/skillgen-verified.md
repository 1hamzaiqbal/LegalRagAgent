---
title: SkillGen — Cross-Reader Skill Generation and Verification
type: source
tags: [skills, cross-model, contextual-utility, selection, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.10999
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.10999.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/SkillGen-verified
authors: SkillGen contributors
year: 2026
---

# SkillGen

## TL;DR

SkillGen is the strongest direct collision with contextual artifact-ordering
transport. It crosses six fixed, source-conditioned final skills with six
evaluator models on the same 100 held-out instances for each of four
benchmarks. Of 120 off-diagonal source–evaluator cells, 70% are nonnegative
and 42% improve by more than five percentage points, but the ordering is not
reader-invariant.

It does not cross every source model's candidate sequence, train weights, or
evaluate withdrawal. The crossed artifacts are only the six final selected
skills. It therefore occupies fixed-artifact contextual reader heterogeneity
and supports a derived wrong-reader selection-regret audit, but not
context-versus-acquisition ordering.

## Design

The six source/base agents are GPT-5.4-Mini, Gemma-4-26B, GPT-5.4-Nano,
Grok-4-Fast, Claude-Haiku-4.5, and Qwen-2.5-7B. A fixed GPT-5.4-Mini auxiliary
model performs induction, generation, and verification for every source.
“Source model” therefore identifies whose trajectories condition the skill,
not necessarily the model that writes its text.

For each source, SkillGen generates up to eight candidates. Construction-time
verification pairs each candidate with a cached no-skill baseline and scores
net repaired minus regressed cases. A candidate must clear the paper's active
gate and the best verified candidate is retained. The final cross-reader
experiment applies each retained skill to all six evaluators on a shared
100-instance held-out set for ALFWorld OOD, ScienceWorld, Mind2Web, and
SocialMaze FTS.

## Evidence

- Across the paper's 80 main benchmark–model cells, final skills improve 50,
  leave 25 unchanged, and regress five. Reported average held-out gains across
  the eight main agents range from +3.27 to +10.08 points.
- The 120 off-diagonal cells in the six-by-six transfer matrices are 70%
  nonnegative; 42% exceed +5 points.
- Mean transfer out of the Qwen source is best on ALFWorld at +21.8 points;
  mean transfer out of GPT-5.4-Nano is best on ScienceWorld at +23.0.
- Candidate selection matters even before cross-reader transport. By round
  eight, the latest candidate has expected delta -3.1 points, while the best
  verified candidate reaches +8.1, an approximately 11-point gap.

### Derived wrong-reader regret

The following is **our calculation from Figure 4, not a paper claim**. For
each source reader, select the row artifact that maximizes that source's
column. Apply that choice to each target and compare it with the artifact that
maximizes the target's own column. The maximum contextual regret is:

| Benchmark | Maximum regret |
|---|---:|
| ALFWorld OOD | 38 points |
| ScienceWorld | 0 points |
| Mind2Web | 10 points |
| SocialMaze FTS | 12 points |

The 38-point ALFWorld case is fragile: the source choice turns on a one-point
margin. SocialMaze gives a less brittle example. GPT-5.4-Nano selects the
GPT-5.4-Mini-conditioned artifact with an eight-point source margin; Claude
then gains +16 rather than its +26 target-column oracle, a 10-point regret.

## Exact claim boundary

SkillGen directly owns a fixed artifact × reader matrix, paired contextual
help/harm, and reader-dependent ordering among six final artifacts. It also
shows why a source's best-of-eight verification loop cannot be replaced with
the latest candidate.

It does **not** test whether the actual source-selected best-of-eight artifact
is target-optimal because the intermediate candidate sequences are not
crossed. There is no matched target training, weight placement, withdrawal,
retention, independent-seed uncertainty, or cost-matched direct-learning
baseline.

## Artifact gaps and limitations

The official checkout contains code and benchmark splits but not the final
skill bytes, raw cross-reader matrix, candidate histories, or run logs needed
to reconstruct Figure 4. Reported transfer cells are point estimates on one
100-instance set per benchmark with no reported uncertainty. Those gaps make
the figure strong collision evidence but not yet a reproducible selector
benchmark.

## Code custody

- Audited arXiv v1 PDF:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.10999.pdf`.
- PDF SHA-256:
  `1a27f3a2e1120854f658e57fac75bb0c30ef3ace7dad6426904cb0a93620a9ee`.
- Official repository: https://github.com/yccm/SkillGen.
- EIT checkout pinned at
  `3c4537bb12ac287ceb1b5d410b491206089fdcb7` on 2026-07-17.
- Repository license: Apache-2.0.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[masa]] · [[skilllens]] ·
[[skillrevise]]
