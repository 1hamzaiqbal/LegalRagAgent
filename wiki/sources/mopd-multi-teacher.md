---
title: MOPD - Multi-Teacher On-Policy Distillation
type: source
tags: [opd, multi-teacher, routing, capability-integration, distillation]
created: 2026-07-20
updated: 2026-07-20
status: primary-source intake and implementation audit complete
url: https://arxiv.org/abs/2606.30406
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.30406.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/verl
authors: Ma et al.
year: 2026
---

# MOPD - Multi-Teacher On-Policy Distillation

## TL;DR

MOPD trains separate domain-specialized RL teachers, then distills their
capabilities into one student on the student's own rollouts. The released veRL
path uses **hard per-sample routing**: a `data_source` value selects exactly one
teacher for the complete sampled trajectory. It is not an ensemble, does not
average several teacher distributions, and does not arbitrate teacher conflict
on the same token.

This is strong prior art for multi-domain capability integration through OPD.
It is a useful later engineering path, but it should not be folded into the
current O-only experiment: the current study asks whether one independently
skill-improved teacher helps a matched task-RL student on M versus O. Adding a
second teacher now would change both the estimand and the preregistered custody
surface.

## Method and implementation boundary

- Each domain first gets its own RL-trained teacher.
- The shared student produces on-policy trajectories over mixed-domain data.
- Each row carries one routing key; veRL maps that key to one frozen teacher
  server and obtains token-level scores on the exact student trajectory.
- The routed teacher signal drives the same sampled-token OPD machinery used by
  the single-teacher path.
- Teacher replica pools are provisioned separately and the configuration fails
  on missing, unknown, or duplicate routing keys.

The canonical veRL example routes GSM8K rows to a text teacher and Geometry3K
rows to a vision-language teacher. That is capability partitioning by known
domain label. It does not learn when to distrust a teacher, compare teacher
counterfactual values, or let multiple teachers score and disagree over one
trajectory.

## Consequence for LegalRagAgent

A later multi-teacher routed OPD campaign needs:

1. at least two teachers that each pass an independent task-skill gate;
2. a sealed source-to-teacher routing manifest;
3. one tokenizer/server/checkpoint/provenance contract per teacher;
4. the chosen teacher key and immutable identity in every student trace;
5. matched per-domain baselines and held-out strata; and
6. an explicit comparison against mixed task RL, sequential/cascade training,
   off-policy teacher text, and parameter merging.

The failed M teacher from the current campaign is permanently ineligible. A
future MOPD experiment must train and gate a genuinely new teacher or use a
different domain pair. If the research question becomes “when should the
student disobey one of several teachers?”, every candidate teacher must score
the same trajectory and the selection/arbitration rule must be explicit; that
is not supplied by veRL's routed MOPD implementation.

## Naming warning

ArXiv `2605.12652` independently uses **MOPD** for *Multi-Rollout On-Policy
Distillation*, which conditions a teacher on peer successes and failures. Use
“multi-teacher routed OPD” for the method here to keep the two lines distinct.
See [[mopd-multi-rollout]].

## Version and custody

- Paper: arXiv v1, submitted 29 June 2026.
- PDF SHA-256:
  `cd7f1cc0584ee14f7a0a9380e25348df4ca1421dbad6ec34e7412754e3a2f188`.
- Upstream implementation: veRL commit
  `6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`, pinned 20 July 2026.
- Code anchors:
  `examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh`,
  `verl/workers/config/distillation.py`, and
  `verl/experimental/teacher_loop/teacher_manager.py`.

## Links

[[verl-opd-trainer]] · [[opd-math-source-transfer]] · [[opd-distillation]] ·
[[ema-policy-gradient]]
