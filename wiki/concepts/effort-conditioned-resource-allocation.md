---
title: Effort-Conditioned Resource Allocation
type: concept
tags: [three-dial, effort-control, agentic-rl, cost, opd]
created: 2026-07-17
updated: 2026-07-17
status: draft
---

# Effort-conditioned resource allocation

## Definition

Train one policy on rollouts carrying an explicit effort/cost condition and a
matching reward or constraint, so that inference-time users can choose how
much resource the policy spends. The condition and the incentive must agree:
varying a penalty without telling the policy which penalty applies asks one
policy to average incompatible objectives rather than learn controllable
behavior.

For the three-dial setting, the candidate objective is a resource vector, not
only a reasoning-length penalty:

`R = reader_task_utility - lambda_think*T_think - lambda_search*N_search - lambda_context*T_evidence - lambda_verify*N_verify`

The policy observes the question, reader, current evidence set, remaining hard
budget, and the cost vector. It chooses among thinking, retrieval, verification,
answering, abstaining, and stopping. A hard budget `B` and a price `lambda` are
not interchangeable: `B` forbids overspend, while `lambda` expresses how much
task utility an extra unit must buy.

## Why the initial idea is useful but not itself novel

[[inkling-controllable-effort]] reports that Thinking Machines varied both an
effort instruction and per-token cost across samples. This is evidence that a
single large model can learn a controllable accuracy/compute curve. The exact
formula `task reward - lambda * reasoning tokens` is a reasonable
interpretation, but the public post does not publish that equation, the
sampling distribution over `lambda`, or the relevant ablations.

The broad claim is already crowded:

- [[training-language-models-to-reason-efficiently]] trains separate models at
  fixed length-penalty strengths.
- [[l1-length-control]] already trains one model on prompt-specified token
  targets sampled per example, so controllable reasoning length predates the
  Inkling report.
- BudgetThinker, SelfBudgeter, Thinkless, and Adaptive Length Penalty also
  learn budget-aware or difficulty-adaptive reasoning.
- [[acting-less-otc]] optimizes correct answers with fewer tool calls.
- [[autosearch]] learns capability-aware minimal search depth from intermediate
  answers and explicitly rewards the marginal F1 gain of a search step.
- [[budget-aware-tool-use]] jointly accounts for token and tool cost and shows
  that a prompt-level budget tracker is already a strong training-free
  baseline.
- Search-R1, Agent-R1, Agent Lightning, OpenRLHF, and veRL mean that an RL
  harness or multi-turn search loop is implementation infrastructure, not the
  research contribution.

Therefore we should not claim novelty for varying `lambda`, controllable
effort, penalizing tool calls, adapting search depth, or combining token and
tool accounting.

## Candidate research gap

The narrower opening is **reader-conditioned substitution between internal
reasoning and external evidence acquisition, supervised by the evidence set's
counterfactual effect on that reader**.

This differs from the closest work only if all of the following are present:

1. **Cross-resource choice:** the policy can spend on thinking, retrieval,
   context, or verification rather than optimizing one resource in isolation.
2. **Reader conditioning:** the same evidence can help a weak reader, fail to
   convert for another reader, or harm a strong reader.
3. **Set-state conditioning:** stopping depends on sufficiency, redundancy,
   conflict, and authority in the accumulated set—not search count alone.
4. **Outcome-grounded marginal utility:** supervision comes from downstream
   reader success/harm under paired evidence interventions, not only retrieval
   relevance or the searching agent's intermediate-answer F1.
5. **Controllable frontier:** one policy responds monotonically and usefully to
   unseen cost/budget combinations instead of producing one fixed compromise.

This is a candidate gap, not a settled novelty claim. [[autosearch]] is a
particularly close mandatory baseline because it already combines capability,
minimal depth, and marginal answer-quality reward.

## Minimal experiment ladder

### A. Establish that a nontrivial switching surface exists

Use the existing paired reader/evidence outcomes to estimate, for each reader
and cost vector, which available action is oracle-optimal. Proceed only if the
optimal action changes across readers, questions, or prices and the changes are
stable under repeated outcomes. This is an analysis gate, not a learned-policy
result.

### B. Small controllable environment

Start with `answer`, `retrieve(k in {1,5,10})`, `verify`, and `stop`; two readers;
and logged token, evidence-token, retrieval, and latency costs. Sample cost
vectors during training and encode the same vector in a structured prompt.
Evaluate held-out questions, readers, and interpolated/extrapolated cost
vectors.

### C. Required baselines

1. best fixed action and fixed search depth at every budget;
2. prompt-only budget tracker/BATS-style orchestration;
3. hard token/tool limits without training;
4. fixed-penalty separate policies in the style of
   [[training-language-models-to-reason-efficiently]];
5. reasoning-only length control in the style of [[l1-length-control]];
6. [[acting-less-otc]]-style minimal tool-call reward;
7. [[autosearch]]-style minimal sufficient depth and marginal-F1 reward;
8. task RL without a cost condition;
9. direct cost-conditioned task RL;
10. direct task RL plus gap-gated OPD from a privileged teacher.

### D. Metrics that can falsify the idea

- task accuracy, abstention utility, and evidence-induced harm;
- realized reasoning, retrieval, context, verification, latency, and dollar
  cost;
- Pareto dominance/hypervolume, not one hand-picked `lambda`;
- budget adherence and monotonic effort response;
- utility calibration: predicted versus realized value of the next action;
- generalization to unseen prices, budgets, readers, and datasets;
- action substitution: whether cheaper thinking replaces search, or vice versa,
  when relative prices change.

## OPD relationship

This is a possible **task for** OPD, not evidence for OPD. First show a teacher
skill gap and a real controllable frontier. Then compare direct task RL against
task RL plus gap-gated OPD, following [[sdar]]. Bare OPD remains a collapse
diagnostic. If direct task RL matches the OPD arm, distillation is unnecessary;
if the privileged teacher does not improve decisions, there is no skill to
distill.

The domain-general sibling [[compute-elasticity-distillation]] asks a smaller
question: can a student internalize a teacher's price-conditioned choice
between internal reasoning and one external tool, when the teacher alone sees
the procedural skill? BARD already closes generic token-frontier distillation.
The present page remains the richer three-dial application, where effort must
be substituted across reasoning, evidence, and verification for a specified
downstream reader.

## Kill rules

- Stop if prompt-only budget awareness matches learned control.
- Stop if one fixed policy dominates across the tested prices/readers.
- Stop the OPD arm if the teacher skill gap is absent.
- Drop the controllability claim if effort is non-monotonic or fails on unseen
  cost values.
- Do not proceed to multi-turn RL until the paired outcome table shows stable
  marginal-utility signal beyond [[autosearch]]-style intermediate-answer
  rewards.

## Links

[[three-dial]] · [[opd-distillation]] · [[budget-constrained-agentic-search]] ·
[[offline-bandit-v0]] · [[judge-answer-conversion]] · [[sdar]] ·
[[compute-elasticity-distillation]]
