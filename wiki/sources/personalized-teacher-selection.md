---
title: PerSyn — Find Your Optimal Teacher
type: source
tags: [knowledge-distillation, teacher-routing, learnability, data-synthesis]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2510.10925
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2510.10925.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/PerSyn
authors: Zhang et al.
year: 2026
---

# PerSyn — Find Your Optimal Teacher

## TL;DR

PerSyn routes each prompt to one of many teacher models using a combination of
teacher-response quality and a target student's likelihood of that response.
A small Bradley–Terry router learns the resulting per-student preference and
then selects a teacher before response generation.

This is direct prior art for student-specific teaching-source selection across
model scales. Its “optimal” teacher is optimal under a compatibility/quality
proxy, not measured counterfactual student learning gain, and the selected
object is a teacher response rather than a reusable procedural skill.

## Method

- Response learnability is the target student's mean token log-likelihood.
- Quality is a reward-model score, or binary correctness for math.
- After within-prompt normalization, the default objective weights quality
  `0.6` and learnability `0.4`.
- A separate router for each student is trained on about 2,500 prompts with
  parallel teacher outputs, then routes the remaining prompts under a
  “route-then-generate” scheme.

The paper's Oracle router also optimizes this proxy. It does not retrain the
student separately on every candidate response to estimate causal teaching
gain or true teacher-selection regret.

## Evidence

- The study uses 50K instruction prompts with 19 teachers and 10K math prompts
  with 15 teachers; five main students span 0.5B–3B, with extensions through
  14B.
- Across the five main students, PerSyn versus CAR reports average scores
  `34.13 vs 32.77`, `50.63 vs 49.21`, `32.85 vs 31.41`, `58.09 vs 57.17`, and
  `34.81 vs 32.99`.
- For Qwen2.5-3B, relative gains over the strongest-teacher baseline are 8.7%
  on IFEval, 7.6% on TruthfulQA, and 2.9% on SVAMP.
- More than 95% of prompts reportedly route to smaller teachers, reinforcing
  that teacher size/standalone strength is not a sufficient selector.

## Novelty boundary

PerSyn closes broad claims around target-specific teacher selection,
learnability-aware routing, and cross-scale evidence that the strongest
teacher is not always optimal. [[skill-lifecycle-research-snapshot-2026-07-17]]
must instead measure a fixed modular artifact in source context, target
context, and after transfer with the artifact withdrawn. PerSyn-style student
likelihood is a mandatory cheap predictor, not the ground-truth teaching
utility.

## Limits and custody

Likelihood is compatibility/ease, not realized learning. There is no paired
training intervention per teacher response, causal selection regret,
natural-language skill lifecycle, context-only deployment comparison, agent
action, or tool cost.

- ACL-linked anonymous snapshot:
  https://anonymous.4open.science/r/PerSyn-8D85.
- First-author public repository: https://github.com/rattlesnakey/PerSyn. Its
  README matches the directly retrievable anonymous README; this relationship
  was recorded rather than treating an unrelated mirror as official.
- Persistent EIT checkout pinned at
  `17e0820e8bade561b183a53bec7cc640a4e06e65` on 2026-07-17.
- PDF SHA-256:
  `bae6fc2d0f8852aded9b33f797235838c36552fb05c8da8451a85107d50f5386`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[promptkd]] ·
[[lgtm-student-level-kd]] · [[distillation-traps-guards]]
