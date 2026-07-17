---
title: Experiment Specification
type: plan
tags: [experiments, distillation, price-conditioning, evaluation]
created: 2026-07-17
updated: 2026-07-17
status: proposed
---

# Experiment specification

## Phase 0: establish the teacher switching surface

Run one capable open-weight teacher on the same validation/test tasks under:

- fixed hard output-token cap, initially 2,048;
- at most two Python calls;
- normalized tool prices `lambda_tool in {0, 0.25, 1, 4, 8}`;
- with and without a short tool-use/verification skill;
- three stochastic rollouts per task-price condition.

The price unit is deliberately abstract in Phase 0. It becomes meaningful
through utility `success - lambda_tool * calls`, not by pretending to measure
real dollars. Real token/tool latency and dollar usage are still logged.

Proceed only if:

- at least two task families show task-dependent switching;
- tool-use probability decreases materially with price without success
  collapsing immediately;
- the learned/elicited teacher frontier dominates always-tool and never-tool;
- the skill changes held-out utility or calibration rather than merely adding
  tokens;
- price instructions are followed consistently enough to train from.

## Price splits

- Seen during training: `{0, 0.25, 1, 4}`.
- Unseen interpolation: `{0.5, 2}`.
- Unseen extrapolation: `{8}`.
- Sensitivity grid after the pilot: log-spaced values centered on the observed
  per-task switching region, not an arbitrary fixed range.

The same task is evaluated across all prices. This paired intervention is the
unit of analysis.

## Phase 1 arms

| Arm | Purpose |
|---|---|
| Student, no price instruction | Detect spontaneous fixed policy |
| Student, prompt-only price | Cheapest controllability baseline |
| Always tool / never tool / difficulty heuristic | Non-learning policy baselines |
| Fixed-price SFT | Show whether one operating point transfers |
| Mixed-price conditional SFT from unskilled teacher | Basic metapolicy distillation |
| Mixed-price conditional SFT from skilled teacher | Test privileged skill benefit |
| Direct price-conditioned task RL | Critical baseline; may solve the problem |
| Task RL + reward/gap-gated OPD | Test dense teacher guidance without bare-OPD collapse |
| Bare OPD | Failure diagnostic only, not the proposed method |

Optional later arms: OPID-style hindsight skills and an external SkillOpt or
CostCraft skill. Do not include them in the first matrix unless the simple
arms leave an interpretable gap.

## Metrics

### Primary

1. **Task success** and native task reward, before cost.
2. **Tool-use curve** `P(call | lambda, task, difficulty)`.
3. **Frontier regret:** utility gap to the best teacher/per-item oracle policy
   at each price.
4. **Counterfactual action agreement:** agreement with the teacher's same-task
   call/no-call decision across prices.
5. **Switch preservation:** accuracy and location error for task-specific
   teacher action reversals.
6. **Unseen-price regret:** interpolation and extrapolation evaluated
   separately.

### Guardrails

- price monotonicity violations;
- success loss at zero price;
- verification/backtracking rate;
- tokens, calls, wall time, and actual dollar cost;
- invalid calls, timeouts, truncations, and refusals;
- action diversity and always/never-tool collapse rate;
- frontier hypervolume only as a secondary summary, never the sole metric.

## Statistical unit

- Pair by task ID across prices and arms.
- Use seed-stratified bootstrap confidence intervals clustered by task.
- Predeclare the model rollout seed policy.
- Use at least three rollouts per task-condition in the teacher gate; determine
  final sample size from observed switching prevalence and variance.
- Report each family separately before pooling.

## Teacher/student pairs

Start within one tokenizer/model family to make OPD technically honest:

- small student: Qwen 1.5B–4B class;
- teacher: Qwen 8B–32B class.

Choose exact current checkpoints only after an EIT GPU/memory smoke. Add an API
frontier teacher only for trace SFT or evaluation. API top-k logprobs are not a
substitute for arbitrary-continuation scoring, tokenizer-aligned OPD, or
J-space access.

## Specialist-over-frontier test

This is optional and downstream. If attempted:

- freeze test seeds before teacher data generation;
- prevent train/test generator overlap;
- match tool access, price display, hard token caps, and sampling count;
- compare actual utility and success, not only raw accuracy;
- state clearly that the student is specialized to the task distribution.

## Method trigger

Only implement a paired-counterfactual or rank-preserving loss if mixed-price
conditional SFT/OPD learns average behavior but fails to preserve teacher
switches on unseen prices. The failure must replicate across families and
model seeds first.
