---
title: SkillMaster — Skill-Guided Policy Optimization and Withdrawal
type: source
tags: [skills, reinforcement-learning, withdrawal, utility, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.08693
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.08693.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/Skill-Master
authors: SkillMaster contributors
year: 2026
---

# SkillMaster

## TL;DR

SkillMaster trains a Qwen2.5-7B policy to act and manage an external skill bank.
Its counterfactual probe utility compares task performance before and after a
candidate bank mutation, and its ALFWorld policy retains almost all aggregate
performance when skill retrieval is removed: 98.7 with skills versus about
98.0 without.

That 0.7-point withdrawal gap is evidence consistent with parametric
persistence, not proof that a named skill was internalized. Action tokens also
receive task reward, cold-start SFT uses skill-augmented traces, and the paper
does not cross fixed artifacts, reset from matched checkpoints per artifact,
or compare against cost-matched direct training.

## Method

SkillMaster begins with cold-start SFT from SKILLRL skill-augmented traces and
then applies GRPO to a policy that both acts and proposes skill-bank changes.
After an episode, the policy may add, update, or retain a skill. A proposed
mutation is evaluated on `K = 4` same-family probe tasks by comparing the same
probe bank before and after the mutation. The score combines success with step
efficiency. DualAdv-GRPO then updates both action and skill-management tokens.

The probe comparison is the closest bridge to student-specific action value:
it asks whether an external action—changing the bank—improves future outcomes
for the current policy. It remains one-policy, aggregate utility rather than a
crossed reader or artifact-placement experiment.

## Evidence

- ALFWorld success is 98.7 for SkillMaster, 89.9 for SkillRL, and 77.6 for
  GRPO.
- WebShop score/success is 95.0/82.0 for SkillMaster versus 85.2/72.7 for
  SkillRL.
- The same trained ALFWorld checkpoint scores 98.7 with retrieval and about
  98.0 without, a 0.7-point aggregate gap.
- Withdrawal changes none of four of six ALFWorld families; the reported gaps
  are 0.9 points for Heat and 4.0 for Cool.

The authors describe internalization as a plausible explanation and
acknowledge alternatives. The aggregate persistence result does not attribute
behavior to any fixed skill or distinguish direct RL learning from transfer
out of external text.

## Exact claim boundary

SkillMaster occupies skill-guided policy optimization, counterfactual
before/after bank utility, and aggregate post-training retrieval withdrawal.
It does not establish artifact-specific acquisition, candidate-order
preservation, cross-reader transport, or causal internalization. A clean
placement study must freeze exact artifact bytes, reset the same checkpoint
for each arm, keep action-token training matched, and compare with direct
task-only SFT/RL.

## Reproducibility discrepancy and artifact gaps

The paper's Equation 3 defines unclipped utility as mean probe delta plus
`alpha * (wins - losses) / K` and reports `alpha = 0.3`. The pinned code wraps
the expression in `max(0.0, ...)`, while current run scripts set
`same_delta_win_loss_gamma=0.2`. This is a material unresolved paper/code
discrepancy; paper-faithful reproduction requires an explicit decision and
logged override.

The repository says model and dataset resources will be released after
acceptance. The pinned checkout contains neither the trained checkpoint nor
the evaluated skill bank or raw result logs. The withdrawal claim therefore
cannot currently be traced to artifact-level records from the official repo.

## Code custody

- Audited arXiv v2 PDF:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.08693.pdf`.
- PDF SHA-256:
  `f1043a0090f86c655b5ff9ea38fde0a632aafe5db70f23aa023623e8469dafb0`.
- Official repository: https://github.com/sduyangmin/Skill-Master.
- EIT checkout pinned at
  `b6916baec73121241d6656e0d8e00b8f173a79ac` on 2026-07-17.
- Repository license: MIT.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] ·
[[10-student-specific-action-value]] · [[skill0]] · [[skillc]] · [[sapo]]
