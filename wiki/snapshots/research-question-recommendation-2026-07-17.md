---
title: Research Question Recommendation — Action-Value Transport Across Readers
type: decision
tags: [research-question, action-value, policy-transfer, three-dial, distillation, skills]
created: 2026-07-17
updated: 2026-07-17
status: recommended research direction; no experiment launched
---

# Research question recommendation — 2026-07-17

## Recommendation

The best current pilot—after a targeted 2026-07-17 venue-date search, but
pending a submission-time repeat and resolution of the remaining Tool-Call
Boundary Drift/RG-OPD/CRAFT code-custody gaps—asks:

> **When does a strong teacher's optimal costly external action become
> suboptimal for the student that must execute or use it, and what structure
> in that action value transfers across model scale and family?**

The first paper should be a causal measurement study, not a new agent
architecture. For the same task item, force both teacher and student to solve
it internally or receive the **identical canonical external payload** through
the harness. Estimate the intervention's value separately for each full reader
configuration, then ask whether its sign, ordering, threshold, or
difficulty-conditioned shape transports. Letting each model author and execute
its own tool call is a different estimand—tool-access/execution utility—and
belongs in a later arm. Add OPD, SFT, advice calibration, or a new metric only
after this panel reveals a stable phenomenon that one of them can explain or
improve.

This is the most defensible way to connect the three dials:

1. **external intervention** — tool call, evidence payload, or verification;
2. **reader/actor conversion** — what the named model can do with it;
3. **cost** — tokens, calls, latency, or a deployment price.

The underlying thesis is not that students should disobey. It is that an
action label exported by a teacher silently combines the task, the teacher's
capability, and the teacher's price threshold. A target student's optimal
decision depends on its own counterfactual payoff.

No experiment was launched in producing this recommendation.

## Exact empirical object

Let reader configuration `r` include the exact checkpoint, harness, system
prompt, renderer/tool adapter, decoding protocol, and tokenizer. Freeze and
version the evaluation metric `q`; utility can reverse when the metric changes.
For item `x` and binary action `a` in `{internal, external}`, define the
repeated forced-action success or task-reward expectation

`p_(r,q)^a(x) = E[Y_q | do(a), r, x]`

and external-action success/reward advantage

`A_(r,q)(x) = p_(r,q)^external(x) - p_(r,q)^internal(x)`.

Record a cost vector `C` separately and estimate incremental expected cost

`DeltaC_r(x) = E[C | do(external), r, x] - E[C | do(internal), r, x]`.

For deployment price vector `lambda`, the net external-action advantage is

`DeltaV_(r,q)(x; lambda) = A_(r,q)(x) - lambda^T DeltaC_r(x)`.

Choose external iff `DeltaV_(r,q)(x; lambda) > 0`. This formulation permits
nonzero internal cost and reader-dependent incremental cost; the simpler
`A > lambda*c` rule is only a special case.

The primary transport test is target-student regret from a teacher oracle
derived from forced outcomes. Define

`pi_T^lambda(x) = 1[DeltaV_T(x; lambda) > 0]`

and the analogous target oracle `pi_S*`. For policy value
`J_S(pi; lambda) = E[Y_q - lambda^T C | actions chosen by pi]`, report

`Regret_(T->S)(lambda) = J_S(pi_S*; lambda) - J_S(pi_T^lambda; lambda)`.

Estimate or fit the teacher rule on disjoint items/repeats and evaluate target
regret on held-out items and independent outcome repeats. Otherwise noisy
item effects can manufacture an apparent transport failure. Call this a
forced-outcome teacher oracle, not the teacher's observed free-choice policy.

The central data product is a crossed
`item × action × reader configuration × repeat` outcome/cost panel, not a hard
teacher label. From that panel we can ask:

- How often do teacher and student advantages have different signs?
- Do they rank items similarly even when their values and thresholds differ?
- Is disagreement concentrated in a zone of intermediate task difficulty?
- Does directionality matter—strong-to-weak versus weak-to-strong advice?
- Can teacher information reduce the number of forced student interventions
  needed to estimate a useful student policy?
- How much held-out target regret results from transferring the teacher oracle?
- Does standard distillation increase teacher agreement while increasing that
  target regret on consequential disagreement items?

Use ordinary causal lift, policy value, and decision regret first. A named
metric is justified only if these fail to expose a replicated decision
problem. A scaling or transfer law is justified only if a relationship
discovered on some sizes predicts untouched sizes and at least one held-out
model family.

## Why this is the strongest opening

The closest work owns the ingredients but not their joint causal object:

- [[llm-specific-utility]] shows that a passage's usefulness depends on the
  downstream reader, but uses deterministic binary passage labels and does not
  study action price or teacher-policy transfer.
- [[model-adaptive-tool-necessity]] shows model-dependent tool need through
  no-tool reliability, not forced tool benefit.
- [[tool-call-boundary-drift]] shows that OPD can move call boundaries, but
  evaluates fixed `should-call` labels rather than the acting student's
  counterfactual payoff.
- [[reward-gated-opd]], [[craft-counterfactual-credit]], and
  [[token-teachability]] show that teacher signals can be harmful,
  incompatible, or deserve negative credit. They do not jointly estimate
  teacher and student values for the same external action.
- [[smartad]] selects already-correct agent trajectories by target-student NLL
  and weights action/final spans more heavily; [[informative-alignment-rsr]]
  predicts post-training benefit across 11 teachers and five students. They
  occupy student-specific trajectory compatibility and teaching utility, not
  the student's immediate causal payoff from the external action.
- Budget-conditioned agents and rational metareasoning already establish that
  actions should respond to cost. They do not audit whether one model's value
  boundary can safely supervise another model.

The provisional opening—not yet a submission-level priority claim—is narrow
and falsifiable:

> **Cross-reader transport of signed, cost-sensitive forced-action value.**

This is more architecture-independent than the earlier OPD plan and more
specific than “retrieval helpfulness.” It can yield a legitimate contribution
even if the result is a boundary or a strong null.

## Contribution ladder — let the evidence choose the paper

The study remains useful under several outcomes:

1. **Causal atlas:** quantify the four teacher/student regimes—both benefit,
   neither benefits, inherited underuse, and inherited overuse—and identify
   where they occur.
2. **Empirical regularity:** discover that ranks transfer but magnitudes or
   zero-crossings do not, or that value follows a capability-relative
   difficulty curve.
3. **Transport failure:** show, with cross-fitted forced outcomes, that the
   teacher oracle has measurable held-out target regret even while teacher
   agreement rises.
4. **Sample-efficiency result:** show that teacher values plus a small target
   calibration set beat a student-only predictor at the same number of target
   interventions.
5. **Strong null:** establish that values transport within uncertainty under
   specified action/task regimes, sharply limiting when model-specific
   policies are necessary.

Only outcomes 2–4 invite a law, calibration method, or training intervention.
Outcome 1 can still be a measurement/benchmark paper if the panel is rigorous
and the heterogeneity is consequential. Outcome 5 is publishable only if the
study is broad and powered enough to close a plausible concern.

## Smallest useful experiment

### Discovery pilot

Use one same-family teacher/student pair, two fresh exact-scored task
generators, and one deterministic external intervention: inject the exact
canonical result of one Python/calculator computation. For every item and
reader configuration, randomize forced internal versus forced canonical-
payload execution and collect repeated stochastic outcomes. Enforce the arm
through the harness rather than materially different task prompts. Keep the
task prompt, token cap, payload format/position, verifier, metric, and decoding
protocol matched. A separate later arm can let each reader author and execute
its own call, explicitly measuring tool-access/execution utility.

Do not show a price in the forced arms. Measure outcomes and incremental costs
once, then analytically rescore the same panel over a dense price range. This
estimates an oracle boundary under hypothetical prices, **not** the model's
behavioral price response. Any behavioral claim requires a separate
free-choice experiment with displayed, randomized prices; that experiment
measures metacognition as well as value sensitivity.

The pilot answers only whether the estimand is measurable, heterogeneous, and
worth scaling. It cannot establish a scaling law or general cross-model
claim.

### Confirmatory panel, only if the pilot survives

- one dense model family with at least four useful sizes;
- a second family with at least three sizes as an architecture holdout;
- canonical deterministic-computation outputs and fixed-information/evidence
  payloads, with autonomous tool execution treated as a separate estimand;
- common held-out items and identical action payloads across readers;
- repeated forced outcomes with item-clustered or hierarchical uncertainty;
- disjoint data for discovering a relationship, fitting a mapping, and
  evaluating untouched sizes/families;
- rich row-level traces, failures, tokens, latency, calls, and verifier
  components so alternative explanations can be tested later.

Legal retrieval should be an external-validity test, not the first surface.
Its noisy retrieval, authority, and answer-evaluation layers would obscure the
basic causal object. A later legal arm can replace the deterministic tool with
a fixed evidence set and ask whether reader conversion creates the same
transport failures.

## Go/no-go gates before training

Continue past measurement only if all of the following hold:

1. forced-action effects are estimable with useful uncertainty;
2. teacher/student differences are not confined to negligible near-zero
   effects;
3. held-out teacher-oracle regret, a disagreement regime, or another stable
   cross-scale structure repeats across independent item families;
4. teacher-side information predicts something about the student's value
   beyond cheap student/task features;
5. the result survives held-out templates and is not a prompt, parser, or
   action-interface artifact.

Stop or reduce the claim if values are too noisy, disagreement vanishes after
uncertainty handling, or a student-only predictor is equally data-efficient.
Do not use OPD merely because the scaffold runs. Distillation becomes relevant
only if teacher policy/value information changes a student decision at lower
target-data cost.

## The skill direction: legitimate, but secondary

The clean remaining skill question is:

> **For a fixed student and several exact versioned procedural skills, does
> the artifact that helps most as runtime context also produce the largest
> no-context gain after matched training and skill withdrawal?**

This “useful context or useful curriculum?” study remains interesting, but it
is riskier and more expensive as a first move. [[skillgen-verified]], [[masa]],
and [[skilllens]] already cross fixed skills over readers; [[skillsbench]] and
[[skillaudit]] already establish model–harness-specific contextual utility;
[[lifeskill]], [[skill0]], [[skillc]], [[skillmaster]], and related work already
establish forms of skill-guided internalization or post-training persistence.
[[informative-alignment-rsr]] further provides a strong target-conditioned
predictor of trajectory teaching value, and [[smartad]] supplies a direct
student-NLL agent-distillation baseline.
The remaining contribution is the **same artifact's placement-conditioned
ordering** under reset-from-base, matched training—not the SkillOpt → SKILL0 →
OPD pipeline itself.

Treat this as the best secondary/backup study. It should not block the cheaper
forced-action pilot. If action-value reversals exist, the two directions can
later meet elegantly: encode the source model's action boundary in a skill,
then test whether contextual use or internalization imports a boundary that is
wrong for the target student. Without that prior phenomenon, combining the
projects would add machinery rather than insight.

## What not to claim

- Do not reuse “does it pay to disobey?” as a title or novelty claim;
  [[student-teacher-deviations]] already owns it.
- Do not claim that tool need, retrieval utility, skill utility, or
  teachability is model-specific as the contribution; all are established.
- Do not call a fitted trend a scaling law without untouched-size and
  held-out-family prediction.
- Do not brand ordinary value or regret as a new metric unless a demonstrated
  evaluation failure requires it.
- Do not lead with OPD, SkillOpt, SKILL0, a router, or a new loss. They are
  downstream interventions or baselines.
- Do not use the prior Legal-RAG results as confirmatory evidence. They are
  hypothesis generators for a new and materially different study.

## Literature addendum - OPSD, SDFT, and SDPO

The 2026 self-distillation cluster in
[[self-distillation-cluster-update-2026-07-17]] strengthens the boundary around
this recommendation without changing it. [[opsd-self-distilled-reasoner]],
[[sdft-continual-learning]], and [[sdpo-rich-feedback]] already show that a
same-model privileged view built from verified solutions, demonstrations, or
rich environment feedback can be distilled into an unconditioned policy.
Generic “context or feedback can be baked into weights” is therefore not an
available contribution.

The trio does not estimate the primary object here: repeated task outcomes
under forced `do(internal)` and `do(external)` arms for the same payload and
reader, incremental action price, or the target-student regret induced by a
teacher's action oracle. SDPO's token/logit “advantage” is a
feedback-conditioned probability ratio, not causal action value. The papers
instead motivate a later three-gate analysis that keeps **acting utility**,
**privileged-view teacher quality**, and **post-withdrawal acquisition
utility** separate.

Their scale results also suggest a capability-relative self-teaching boundary,
but not yet a scaling law: SDFT is worse than SFT at 3B and better at 7B/14B,
while SDPO loses or ties at the smallest Qwen2.5 sizes and improves at larger
sizes. Any eventual regularity must predict untouched sizes and a held-out
family. Finally, SDFT's paper describes reverse KL while its official code
states that all headline results used forward KL on student/on-policy
prefixes; [[sdft-continual-learning]] records that custody warning. Treat all
three methods as downstream baselines only if the architecture-free pilot
first reveals a stable phenomenon.

## Decision

Proceed, when ready, with the **student-specific forced-action value pilot**.
Keep the skill-lifecycle question documented and independent. The primary
scientific wager is that forced-action value may fail to transport across
reader configurations even when useful task structure does. The experiment
should be designed to discover whether that wager is true—not to guarantee a
metric, law, or method paper.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] ·
[[action-value-transport-reading-packet-2026-07-17]] ·
[[skill-lifecycle-research-snapshot-2026-07-17]] · [[three-dial]] ·
[[opd-distillation]] · [[self-distillation-cluster-update-2026-07-17]]
