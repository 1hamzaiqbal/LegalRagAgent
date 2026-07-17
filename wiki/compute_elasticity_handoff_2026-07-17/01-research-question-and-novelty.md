---
title: Research Question and Novelty Boundary
type: plan
tags: [novelty, compute-elasticity, metareasoning]
created: 2026-07-17
updated: 2026-07-17
status: proposed
---

# Research question and novelty boundary

## The simple question

For input `x`, price `lambda`, and one external capability, a teacher induces a
conditional policy `pi_T(a | x, lambda)`. We want to know whether a smaller
student learns the *function of price*, not just the most common teacher action:

`pi_S(a | x, lambda) approximately pi_T(a | x, lambda)`.

The important test is counterfactual. The same task is evaluated repeatedly at
different prices. Some prices are withheld during training. A successful
student should substitute internal reasoning for the tool when appropriate,
retain tool use on hard/high-value cases, and interpolate to unseen prices.

Call the measured property **resource-response preservation** or
**metapolicy elasticity** in drafts. Do not claim ownership of “compute
elasticity”: ElasticLM already uses that phrase for elastic distilled models.

## Why this remains plausible

Recent work occupies nearly every ingredient but does not, in the sources
audited here, make cross-scale preservation of paired, counterfactual
same-task price-response curves the central evaluation object:

- BARD distills token-budget-conditioned reasoning behavior.
- INTENT and BAVT plan under dynamic tool prices or remaining budgets at
  inference time.
- MOC trains a preference-conditioned multi-objective policy that generalizes
  to unseen preferences.
- CoRL trains a budget-conditioned controller over multiple LLMs.
- Agent Distillation transfers code/retrieval trajectories across scale.
- ClawTrace/CostCraft distills cost-aware external skill files from traces.
- OPID distills hierarchical hindsight skills during on-policy RL.
- SkillMOO, SkillOpt, SkillGrad, and related work optimize external skill or
  harness text/code for quality, cost, or runtime.

The gap is therefore an **intersection and measurement claim**, not a broad
method claim. It may disappear under another search or a simple baseline. The
experiment is designed to find that out quickly.

## Claims that are already closed

Do not claim any of the following:

- variable lambda across rollouts is new;
- budget-conditioned reasoning is new;
- a single policy spanning several cost/quality tradeoffs is new;
- tool-use penalties or budget-aware planning are new;
- skill internalization, skill optimization, or skill-conditioned OPD are new;
- a smaller specialist beating a larger model on a narrow task is itself new;
- building another agent harness is the research contribution.

## Candidate contribution, in descending order of defensibility

1. **Evaluation protocol:** paired same-task interventions over a train/test
   price grid, with exact task verification and separate cost accounting.
2. **Finding:** common distillation methods collapse, preserve, or distort the
   teacher's resource-response curve in identifiable ways.
3. **Transfer comparison:** direct conditioned RL, conditional trace SFT,
   teacher-only skills, and reward-gated OPD preserve different parts of the
   curve.
4. **Method only if required:** a paired-counterfactual loss that preserves
   teacher action reversals or relative preferences across prices.

The fourth item should not be implemented until the first three reveal a real
failure that simpler baselines cannot fix.

## Factorial interpretation

Treat skills, distillation channel, and price conditioning as separable axes:

| Axis | Levels |
|---|---|
| Teacher context | no skill; tool/verification skill |
| Student learning | prompt only; trace SFT; direct RL; task RL + gated OPD |
| Price exposure | fixed; mixed seen prices; unseen interpolation/extrapolation |

This prevents “skills helped” from being confused with “distillation preserved
elasticity.”

## What would make this a good paper

- A clear, reproducible phenomenon across at least two task families and two
  teacher/student scale pairs.
- Strong simple baselines, especially direct conditioned RL.
- An evaluation object that cannot be recovered from a single average reward.
- A negative result that explains when distillation collapses a metapolicy can
  still be useful.
- Any “beats frontier model” result uses frozen held-out tasks, matched tools,
  matched budgets, and contamination controls; it is not a prerequisite.

## What would kill it

- The teacher does not exhibit task-dependent switching across prices.
- Prompt-only conditioning already transfers perfectly to the student.
- Apparent elasticity comes only from instruction compliance, not utility.
- A newly found paper directly evaluates cross-scale unseen-price response
  preservation with the same controls.
- The effect exists only for one synthetic generator or one arbitrary price
  normalization.
