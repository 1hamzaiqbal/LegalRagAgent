---
title: Compute-Elasticity Distillation — Broad Claim Closed, Agentic Candidate
type: concept
tags: [compute-elasticity, distillation, opd, skill0, effort-control, agents]
created: 2026-07-17
updated: 2026-07-17
status: candidate direction — broad novelty rejected; bounded pilot specified
---

# Compute-elasticity distillation

## Bottom line after the primary-source audit

The initial broad idea is **not novel**:

> “Distill a stronger teacher into a smaller model while preserving a
> user-controllable accuracy–token frontier.”

[[elastic-language-models]] already uses “compute elasticity” for a distilled
model with multiple structural operating points. More decisively,
[[bard-budget-aware-reasoning-distillation]] distills a teacher into an 8B
reasoner controlled across 500–8,000-token budgets and explicitly reports
budget-dependent changes in exploration, verification, and self-correction.
[[crisp]] also transfers concise behavior from a prompted self-teacher into an
unprompted policy. We must not claim the term, the token-budget frontier, or
prompt-withdrawal distillation as new.

The July 2026 follow-up search found additional direct collisions:
[[intent-budget-constrained-agents]] studies changing tool prices,
[[moc-one-model-for-all]] studies unseen preference conditions,
[[clawtrace-costcraft]] performs cost-aware skill distillation, and [[opid]]
combines on-policy RL with hierarchical skill distillation. The narrower,
better research object is therefore an empirical transfer property:

> **Counterfactual resource-response preservation:** can a smaller student
> preserve a teacher's task-dependent action policy as a resource price
> changes—including prices withheld during training—or does distillation
> collapse it to one average, always-tool, or never-tool strategy?

In the minimal experiment, the expensive capability is one external action
(for example, Python/code execution or retrieval). The student observes the
task, a fixed token budget, and a variable tool price. It must choose to solve
internally, call the tool, optionally verify once, or stop. The teacher sees the
same state. A teacher-only skill file is a factorial condition, not the core
novelty claim; the deployment student never sees it.

This retains the motivating SKILL0 × Inkling connection in a bounded sense:

- [[skill0]]: procedural scaffolding exists at training but is withdrawn at
  deployment;
- [[inkling-controllable-effort]]: the desired resource condition varies
  across rollouts rather than producing one fixed policy;
- [[agent-distillation-tools]]: full reasoning/tool trajectories transfer
  across scale;
- the candidate object: whether the student's *price-conditioned routing
  function* matches or improves on the teacher across paired counterfactual
  interventions.

This is a candidate gap, not a priority claim. The 2026 literature is moving
quickly, and a venue-date search is mandatory before submission.

## What the closest work already owns

| Neighbor | Already demonstrated | What remains different, if anything |
|---|---|---|
| [[elastic-language-models]] | Distilled nested submodels and online latency/performance elasticity | Architectural depth and request-load scheduling, not deliberative tool choice |
| [[bard-budget-aware-reasoning-distillation]] | Cross-scale, budget-conditioned reasoning distillation with strategy changes | Token budgets only; no external action price or skill-withdrawal comparison |
| [[rational-metareasoning]] | Value-of-computation training for selective reasoning | Same-policy token cost; tool/API cost is future work |
| [[agent-distillation-tools]] | Cross-scale distillation of retrieval and code-tool behavior | Fixed tool regime; no inference-time price condition or substitution curve |
| [[strategy-guided-policy-optimization]] | Reusable strategy descriptions, adaptive usefulness weighting, token-level transfer | No external tool price and no user-controllable resource condition |
| [[cost-aware-skill-rewriting]] | Quality/cost-aware rewriting and task-conditioned selection of external skills | Edits the skill document; does not internalize a price-conditioned policy into weights |
| [[acting-less-otc]] / [[budget-aware-tool-use]] | Same-policy training or prompting for cheaper tool use | Not cross-scale skill-privileged distillation |
| [[privileged-information-distillation]] | Teacher-side PI transfer, including action-only frontier agents | No conditioned price-response target |

The novelty cannot be “cost aware,” “tool distillation,” “strategy
distillation,” “skill internalization,” or “budget conditioned” separately.
Even their intersection is now crowded. The plausibly under-owned object is
the paired evaluation of cross-scale response-curve preservation, with skills
as an ablation and direct conditioned RL as a mandatory baseline. See the
self-contained [[compute_elasticity_handoff_2026-07-17/README]].

## Research object: a metapolicy over capabilities

Let the resource condition be `c = (B_tokens, lambda_tool)`, with a fixed hard
token cap and a variable tool price. A skill-augmented teacher is

`pi_T(trajectory | x, c, skill)`

and the deployment student is

`pi_S(trajectory | x, c)`.

The task reward is

`U = success - lambda_token * tokens - lambda_tool * calls`.

The important learned object is not one answer or one skill. It is the
counterfactual response:

- when the tool is cheap, use it where it buys reliability;
- as its price rises, substitute internal reasoning on solvable items;
- avoid both the always-tool and never-tool shortcuts;
- spend verification only when its expected value exceeds its cost.

This is a small rational-metareasoning problem with verifiable outcomes. It
does not require legal data, many tools, a retrieval tree, or a multidimensional
agent harness.

## Three transfer channels

### Behavioral traces

Generate teacher trajectories at multiple prices and train the student with
SFT/sequence-level KD. This works with API teachers and is the simplest
baseline. It can fail by averaging price regimes or copying a trajectory that
does not fit student capacity.

### On-policy distributional transfer

Use the teacher to score student-visited states. The local Qwen stack already
completed a three-step smoke run ([[opd-skill0-design]]), but that validates
only plumbing.

The primary-source audit imposes strict controls:

- [[rethinking-opd]]: teacher/student thinking patterns must be compatible and
  the teacher must contribute genuinely new behavior;
- [[reward-gated-opd]]: verifier reward should gate misleading teacher
  supervision;
- [[rethinking-privileged-opd]]: privileged OPD can erase verification and
  backtracking and degrade long-budget performance;
- [[turnopd]]: deep agent turns can be undertrained by token-normalized KL.

Bare OPD remains a failure diagnostic. The credible arm is task reward plus
reward/gap-gated teacher supervision.

### White-box latent diagnosis

[[jacobian-global-workspace]] can test whether price conditions recruit
concepts such as “verify,” “uncertain,” “search,” or “stop,” and whether those
representations disappear after distillation. It requires open weights,
residual activations, and backward/Jacobian access.

It is not the main method. [[implicit-cot-distillation]], [[coconut]], and
[[lori]] already make generic hidden-state and low-rank reasoning transfer a
crowded area. Use J-space only after a behavioral result exists, and only claim
it if causal intervention changes price-responsive behavior.

## Open weights versus API models

### Primary lane: open-weight teacher and student

Open weights are the preferred scientific lane because they permit:

- direct price-conditioned RL;
- teacher scoring on arbitrary student continuations;
- same-tokenizer OPD;
- hidden-state/J-space analysis;
- reproducible cost accounting and ablations.

Qwen is the practical first family because the OPD scaffold and smoke run
already use it. Begin with a 1.5B–4B student and an 8B–32B teacher. Scale only
if the price-conditioned skill gap exists.

### API lane: black-box teacher and frontier evaluation

An API teacher can generate price-conditioned traces, tool decisions, labels,
critiques, or action-only PI. It can also serve as the held-out frontier
baseline for a specialist claim.

Ordinary output logprobs do not provide J-space and may not support dense OPD.
Reverse-KL OPD needs the teacher to score the student's sampled tokens at the
student's visited states, ideally over a relevant token support. Top-k
logprobs only for the API model's own generated text are insufficient, and
tokenizer mismatch complicates alignment. When arbitrary continuation scoring
is unavailable, use trace KD or action-only PI rather than pretending to have
dense OPD.

## Minimal experiment

### Phase 0: demonstrate a nontrivial teacher switching surface

Use the audited procedural tasks in [[reasoning-gym]] with one priced Python
action. For repeated held-out items and a grid of tool prices, run the same
open-weight teacher:

1. without the skill file;
2. with a short tool-use/verification skill file.

Measure success, tokens, tool calls, verification, and total utility. Continue
only if:

- the optimal action changes across tasks/prices;
- the teacher responds monotonically enough to price;
- the skill improves utility or calibration on a held-out split;
- one fixed always/never-tool policy does not dominate.

This is the E2 skill-gap gate the current project never ran.

### Phase 1: one tool, one varying price

Train/evaluate these arms:

1. base student and prompt-only price condition;
2. best fixed always/never/heuristic tool policy;
3. BARD-style token-budget student without a tool-price objective;
4. fixed-regime Agent Distillation from teacher trajectories;
5. direct price-conditioned task RL;
6. price-conditioned trace KD from the unskilled teacher;
7. price-conditioned trace KD from the skill-augmented teacher;
8. task RL plus reward-gated OPD from the skill-augmented teacher;
9. bare OPD only as a collapse diagnostic.

The primary comparison is 5 versus 7/8: does teacher/skill transfer add
anything beyond direct conditioned RL?

### Phase 2 only if Phase 1 works

Add either retrieval as a second environment or a single explicit `verify`
action. Do not begin with the complete three-dial legal controller. The same
research question should survive with one tool and one price.

## Evaluation

Report the surface, not one selected operating point:

- success and utility versus realized tokens/tool calls/dollars;
- tool-use probability and verification rate across held-out prices;
- price monotonicity and interpolation/extrapolation;
- regret/hypervolume relative to the teacher and per-item oracle;
- fixed-policy, prompt-only, direct-RL, BARD, and Agent-Distillation baselines;
- preservation of alternative strategies and long-budget self-correction;
- held-out task types and contamination controls.

The most interesting diagnostic is **substitution elasticity**: as the tool
price increases, does the student selectively replace tool calls with internal
reasoning on tasks it can solve, while retaining tools on tasks where they are
worth the price?

## “Small model beats frontier” target

[[distilling-step-by-step]], [[deepseek-r1-distillation]],
[[agent-distillation-tools]], [[turnopd]], and
[[thinking-machines-expert-judgment]] show that specialists can exceed much
larger or frontier baselines on selected tasks. That makes the target feasible,
not novel.

A defensible claim requires a frozen held-out set unused for trace generation,
reward design, skill writing, or checkpoint selection; identical tools and
sampling protocols; current frontier API baselines; uncertainty over
items/seeds; and separate reporting of training cost versus deployment cost.
The claim must remain task-specific.

## Kill rules

- Stop if the teacher has no skill gap or price-responsive switching surface.
- Stop distillation if direct price-conditioned task RL matches it.
- Drop controllability if the student learns always-tool/never-tool or fails
  held-out prices.
- Drop OPD if it suppresses verification/backtracking or long-budget gains.
- Keep J-space diagnostic-only unless an intervention causally improves the
  cost/quality surface.
- Do not claim frontier superiority under unmatched tools, best-of-N, or cost.

## Relationship to the existing tracks

This direction is domain-general. [[effort-conditioned-resource-allocation]]
remains the richer three-dial application over thinking, evidence, and
verification for a specified reader. The present proposal is deliberately
smaller: one external capability, one variable price, one cross-scale
skill-withdrawal question. Legal retrieval becomes a later stress test, not the
benchmark dependency.

## Links

[[opd-distillation]] · [[opd-skill0-design]] · [[skill0]] ·
[[inkling-controllable-effort]] · [[bard-budget-aware-reasoning-distillation]] ·
[[agent-distillation-tools]] · [[strategy-guided-policy-optimization]] ·
[[rational-metareasoning]] · [[jacobian-global-workspace]]
