---
title: SmartAD — Capacity-Aligned Agent Distillation
type: source
tags: [distillation, agents, tools, student-compatibility, trajectory-selection]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://aclanthology.org/2026.findings-acl.1349/
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/acl_2026.findings-acl.1349.pdf
authors: Tang and Zhao
year: 2026
---

# SmartAD

## TL;DR

SmartAD is direct prior art for **capacity-aligned agent distillation**. From
several already-correct teacher reason–act–observe trajectories, it selects the
one with minimum negative log-likelihood under the named student, then weights
action and final-answer tokens more heavily than reasoning tokens during SFT.

It studies which successful trajectory the student can imitate, not whether
the external action is causally useful for that student. There is no matched
`do(external)` versus `do(internal)` outcome panel, identical canonical
payload, action cost, teacher/target forced-action advantage, or target regret
from transferring a teacher action boundary.

## Method

- Teacher: Qwen2.5-32B-Instruct.
- Students: Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct.
- For each example, sample `K = 10` teacher trajectories, retain correct
  candidates, and choose the minimum student-NLL trajectory.
- Segment each trajectory into reason, action, and final spans. The reported
  token-loss weights are `1.0`, `1.5`, and `2.0`, respectively.
- Fine-tune each student with LoRA for two epochs and evaluate nine in-domain
  and out-of-domain multi-hop QA/math datasets.

The candidate set is conditioned on teacher success. Minimum NLL measures
compatibility with the student's current distribution; it is not action value
or measured learning gain for each candidate.

## Evidence

| Student | Vanilla agent distillation | SmartAD | Difference |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 24.19 | 28.42 | +4.23 |
| Qwen2.5-3B | 30.92 | 34.00 | +3.08 |

For the 1.5B ablation, vanilla agent distillation scores 24.19,
minimum-NLL selection 25.60, segment weighting alone 27.30, and the combined
method 28.42. On HumanEval with the 3B student, SmartAD scores 52.34 versus
49.67 for vanilla agent distillation.

These results make minimum-NLL trajectory selection and segment-weighted
agent SFT mandatory baselines if the action-value project later reaches a
training phase. They do not answer the measurement-first question.

## Exact claim boundary

SmartAD occupies the claims that successful tool-agent traces vary in
student compatibility, that student NLL can select an easier trace, and that
action/final spans may deserve more SFT weight. It does not test whether the
student should take the teacher's action, whether one action has different
signed immediate payoff for teacher and student, or whether increased
teacher-action agreement lowers target utility.

This yields a clean three-way separation:

1. **acting utility** — does the external intervention help this reader now?;
2. **trajectory compatibility** — can the reader imitate this trace?;
3. **training utility** — does learning from it improve later performance?

SmartAD addresses 2 and aggregate 3. The proposed forced-action panel addresses
1 before considering training.

## Limitations and custody

- One teacher checkpoint/family and two closely related student sizes.
- Teacher trajectories differ in tool calls and observations; the treatment
  is not a canonical action payload crossed over readers.
- No action price, forced-arm causal estimate, held-out teacher-policy regret,
  or independent training-seed uncertainty is reported.
- Factual QA uses Qwen-Plus as an evaluator, while math is exact-scored.
- No official repository is cited in the paper or was found by exact-title and
  author search as of 2026-07-17.

PDF SHA-256:
`853f35be7297d52cc3e37d6ab1cc9a0c0330da598cc72cf6c2ced14a0da78893`.
Primary PDF:
https://aclanthology.org/2026.findings-acl.1349.pdf.

## Links

[[research-question-recommendation-2026-07-17]] ·
[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] ·
[[informative-alignment-rsr]] · [[token-teachability]]
