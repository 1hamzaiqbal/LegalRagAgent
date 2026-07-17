---
title: Action-Value Transport Reading Packet — 2026-07-17
type: review
tags: [distillation, tool-use, reader-utility, causal-action-value, novelty]
created: 2026-07-17
updated: 2026-07-17
status: maintained
---

# Action-value transport reading packet — 2026-07-17

## Bottom line

Seven close papers make the broad “students should sometimes disobey
teachers” framing non-novel. They do **not** close the narrower, more useful
question:

> When the same costly external action is available to two models, does a
> teacher's action or advice help the target student, hurt it, or cease to be
> informative—and how does that vary across items, models, actions, and costs?

The immediate opportunity is a rich causal measurement surface, not a
preselected paper form. A metric, scaling/transfer law, taxonomy, calibration
rule, or training method would be a welcome **result of the investigation**,
not an assumption built into its design.

One plausible pattern worth testing—among several—is:

> **Action-value rankings may partially transfer across model capability even
> when cardinal values and utility-maximizing thresholds do not.**

It may hold, partially hold, reverse by task, or fail entirely. Each outcome is
informative if the experiment measures the relevant counterfactuals cleanly.

No experiment was launched during this reading pass.

## Research posture: questions first, contribution later

The first pass should be designed to answer questions rather than validate a
chosen artifact:

1. Does the same external action have measurably different signed value for
   different acting models?
2. When teacher and target values differ, which direction dominates:
   inherited underuse, inherited overuse, or task-dependent reversals?
3. Is the teacher's self-policy different from its advice for a named target,
   and does target-conditioned advice actually improve target outcomes?
4. Which structure, if any, transfers across scale and family: signs, ranks,
   magnitudes, thresholds, difficulty ordering, or nothing stable?
5. What explains heterogeneity—unaided competence, payload quality,
   readability, context burden, tool execution, verification ability, or
   surface task features?
6. Are valuable signals also teachable, and are harmful teacher signals
   sometimes especially easy to imitate?
7. If training is later introduced, does it improve target utility, merely
   increase teacher agreement, or change the action-value surface itself?
8. Does the phenomenon replicate across deterministic tools, fixed evidence,
   and verification actions?

These questions support multiple legitimate outcomes. A stable relationship
could justify a law; a useful summary could justify a metric; a low-dimensional
repair could justify calibration; a heterogeneous failure could justify a
taxonomy or benchmark; and a null result could sharply delimit when teacher
policy transfer is safe.

## What the seven papers actually identify

| Layer | Primary work | What it identifies | What it does not identify |
|---|---|---|---|
| Predictive deviation | [[student-teacher-deviations]] | Why distilled predictive probabilities can systematically deviate and sometimes generalize better. | Executable actions, action prices, or target-student potential outcomes. |
| Need/reliability | [[model-adaptive-tool-necessity]] | Whether a model fails at least once in ten no-tool samples, plus hidden-state need/action probes. | Whether the tool causally improves that model's outcome. |
| Behavior boundary | [[tool-call-boundary-drift]] | How multi-teacher OPD shifts a fixed should-call boundary and how to calibrate the operating point. | Whether the fixed label is utility-optimal for the acting student. |
| Reward-aligned trust | [[reward-gated-opd]] | Whether realized group-relative reward and teacher likelihood directionally agree. | The value of the opposite action or a cost-sensitive target-student policy. |
| Counterfactual token credit | [[craft-counterfactual-credit]] | A signed, sibling-rollout importance-weighted proxy for privileged self-teacher token influence. | Repeated fixed-state interventions or cross-scale action-policy transport. |
| Signal absorbability | [[token-teachability]] | Whether teacher probability mass lies on alternatives already plausible to the student. | Whether imitating that signal improves task utility. |
| Reader-specific evidence value | [[llm-specific-utility]] | A deterministic per-model `incorrect → correct` passage flip and cross-reader transfer matrix. | Signed/repeated causal effect, price, evidence-set marginal value, or teacher advice. |

The main conceptual correction is that four questions must remain separate:

1. **Task/action value:** Does the external action improve this model's
   outcome?
2. **Policy transport:** Does the teacher's threshold or recommendation
   maximize the target student's value?
3. **Signal absorbability:** Can this student locally imitate the teacher's
   distribution?
4. **Training credit:** Which teacher tokens should be reinforced, ignored, or
   opposed?

LLM-Specific Utility is closest to (1), the proposed project owns (2), Token
Teachability studies (3), and RG-OPD/CRAFT study parts of (4). Conflating these
layers would make the contribution both less novel and less causally sound.

## Exact remaining object

For model `m`, item `x`, and binary external action `a`, estimate

`p_m^a(x) = P(success | do(a), m, x)`

and the external-action advantage

`A_m(x) = p_m^external(x) - p_m^internal(x)`.

At price `lambda*c`, the model-specific net value is

`V_m(x,lambda) = A_m(x) - lambda*c`.

The teacher's action label is a thresholded statement about `A_T`; deploying
that label for the student requires `A_S`. The scientific question is not
whether teacher and student actions differ, but whether applying a teacher
policy produces target-student regret under randomized target-student
potential outcomes.

For teacher advice policy `pi_T→S`, the primary population quantity is the
teacher-following causal value

`TFCV_T→S(lambda) = E[(2*pi_T→S(x,lambda)-1) V_S(x,lambda)]`.

Always compare its policy value with the better constant action. Otherwise an
always-call or never-call teacher can appear “adaptive” by recommending only
the globally stronger arm.

The normalized secondary transport summary is

`AWTS_T→S = E[(2*pi_T→S-1)V_S] / E[|V_S|]`,

with raw TFCV kept primary because the denominator can be noise-inflated. A
price-integrated regret curve provides the third view:

`ITR_T→S = E_(x,lambda)[|V_S| 1[pi_T* != pi_S*]]`.

These are standard policy-value/regret constructions. The contribution is not
renaming regret; it is constructing the model-indexed forced-action matrix and
using it to audit cross-scale policy transport.

## Three labels that should be tested head-to-head

The most elegant first analysis compares, on identical model/item/action
cells:

1. **Reliability need:** `1[not 10/10 correct without tool]` from
   Model-Adaptive Tool Necessity.
2. **Benchmark boundary:** the fixed `should-call` target used by Tool-Call
   Boundary Drift and many tool-use datasets.
3. **Causal decision:** `1[A_S(x) > lambda*c]` from repeated forced outcomes.

These labels can disagree for principled reasons. Reliability need ignores
the tool arm; fixed should-call labels ignore the acting model; causal value
depends on both arms and price. If the third label better predicts
independently repeated target-student reward, the project has a crisp
measurement contribution without inventing an architecture.

Sensitivity to the number and temperature of no-tool samples should be
reported rather than treating the reliability label as ground truth. Under the
ten-draw rule, a model with true no-tool accuracy `.95` is labelled necessary
with probability `1-.95^10 = .401`.

## Utility × teachability: the missing 2×2

Token Teachability supplies an orthogonal axis to target-student task value:

| Target-student value of teacher-preferred behavior | Token-level teachability | Interpretation |
|---|---|---|
| positive | high | Distill; this is the easy success case. |
| positive | low | Valuable but incompatible; use scaffolding, value transfer, or target calibration rather than literal trace imitation. |
| negative | high | **Most dangerous:** the student can easily learn behavior that is wrong for its own utility. |
| negative | low | Reject or abstain; the signal is both harmful and hard to absorb. |

The third quadrant is the cleanest training-risk claim. It predicts a decisive
pattern after distillation: teacher agreement rises while independently
measured target-student utility falls. Neither reward gating nor teachability
alone guarantees protection against this failure.

Candidate compatibility covariates for external evidence/actions include
payload perplexity, answer-support overlap, decomposition depth, schema
complexity, context length, and the student's probability mass on the
teacher's action/trace alternatives. These are possible mediators, not labels.

## What CRAFT changes

CRAFT means we cannot claim the first signed counterfactual credit for
teacher-preferred actions. Its strongest reusable role is as an inexpensive
proxy baseline. On matched prefixes, collect:

- CRAFT's sibling-rollout SNIS estimate;
- an actual forced reroll/intervention estimate;
- effective sample size and weight concentration;
- sign agreement and magnitude bias by token position.

The paper's own exchangeability qualification predicts that agreement should
be best at `t=0` and degrade downstream. Verifying that curve would both audit
a close method and clarify why direct action-level intervention is needed.

## What LLM-Specific Utility changes

We cannot claim that passage utility is model-specific or that human-gold
passages are not universally optimal. The more defensible contribution is to
replace its one-shot deterministic label with a repeated, signed,
cross-fitted effect.

The direct benchmark extension is:

1. reproduce the model×passage transfer matrix with multiple independent
   generations per arm;
2. construct utility labels on one outcome split and evaluate selected
   passages on another;
3. report benefit, neutrality, and harm instead of only `0→1` flips;
4. analytically rescore over context/retrieval price;
5. move from single passages to marginal evidence-set additions;
6. hold out an entire model family when fitting a transport rule.

This test is important because the original diagonal dominance is partly
mechanical: each model's “gold” passages are selected using the same
deterministic outcomes later used to show that own-gold evidence is best.

## Candidate patterns to test, not assumed results

### 1. Rank may transport while thresholds do not

Fit teacher-versus-student advantage maps and separately report:

- Spearman/Kendall rank transport;
- calibration slope/intercept and monotonicity;
- zero-crossing displacement;
- regret from the uncalibrated teacher threshold;
- target-student labels required by an affine or isotonic calibration to
  reach a prespecified regret.

One possible positive result is that a one- or two-parameter map outperforms
teacher action imitation and reaches the student-only predictor's regret with
fewer student interventions. Equally important alternatives are no ordinal
transport, near-perfect direct transport, family-specific maps, or
non-monotonic reversals.

### 2. Zone of actionable difficulty

Fit unaided model capability and item difficulty, then ask whether external
action gain follows a held-out-predictive curve in relative difficulty:

`A_mi = g(item_difficulty - model_capability, payload_quality, action_type)`.

One hypothesized curve is small when the item is already easy, positive when
external help can rescue the model, and small or negative when the payload is
beyond the model's conversion ability. This is the cleanest connection to the
three-dial reader-conversion story. It is a scaling law only if it predicts a
held-out size and family; otherwise it is one possible heterogeneity pattern,
not a law.

### 3. Teacher suitability is target-specific

Raw teacher accuracy or scale may not predict advice quality. Build a directed
advisor×target matrix of TFCV/AWTS and test whether the best advisor for one
target is suboptimal for another. This turns “stronger teacher” into a
measurable transport relation and can reveal both inherited underuse and
inherited overuse.

## Measurements to collect before any training

The measurement pass should precede OPD:

1. randomized, repeated forced internal/external outcomes for a same-family
   model ladder;
2. held-out family and task/action type;
3. teacher self-policy versus explicitly target-conditioned advice;
4. rank, threshold, and calibration analysis;
5. reliability-label/fixed-label/causal-label disagreement;
6. all four teacher/student value regimes, weighted by `|V_S|`;
7. teacher-informed versus student-only sample-efficiency curves;
8. free-choice behavior only after causal values are fixed.

Training becomes justified only if action labels import stable regret that a
simple calibration baseline exposes but does not fully solve. A later training
comparison should include direct task reward, scalar entry-bias/threshold
calibration, vanilla OPD, RG-OPD, CRAFT or its proxy, teachability filtering,
and student-value gating at matched compute.

### Overcollect observables, not claims

The run schema should retain more than the minimum needed for the first plot:

- immutable item, task-family, model, checkpoint, prompt, action, payload, and
  tool-interface identifiers;
- randomized arm, generation seed, repeat index, raw answer, parsed answer,
  verifier output, and failure reason;
- requested and realized tool calls, schema validity, execution success,
  observations returned, and whether the final answer used them;
- input/output/reasoning tokens, context length, wall time, tool latency, and
  enough raw accounting to apply alternative cost functions later;
- no-action confidence/consistency, available token probabilities or logits,
  free-choice action, teacher self-action, target-conditioned advice, and raw
  rationales where collection is affordable;
- item difficulty, unaided model competence, payload quality/readability,
  evidence overlap, distractor/conflict features, and teachability proxies;
- checkpoint and training-stage identifiers if a later intervention changes
  the student.

Overcollection should not turn exploratory mining into confirmatory evidence.
Freeze schemas and randomization before running, preserve an untouched
held-out slice, label analyses as planned versus discovered, and require any
mined relationship to predict new items, seeds, models, or families before
calling it a transferable finding.

## Connection to the three dials

This is not a fourth dial. It sharpens the existing reader-conversion dial:

- retrieval/selection chooses a fixed evidence set;
- reader conversion determines `A_m(x,E)` for the named model;
- cost/effort determines whether `A_m(x,E)` exceeds the action price.

A strong teacher can rationally skip evidence it already knows while a weaker
student needs it. The reverse can also occur: the teacher can integrate a
complex or conflicting evidence set that distracts the student. The proposed
transport audit measures exactly when exporting the teacher's boundary
misprices this reader-specific conversion.

The earlier BarExam/Housing observations remain hypothesis generators only.
They are single-generation, subset-dependent results and must not select the
confirmatory items, model pairs, or prices.

## Claims to avoid and claims still available

Closed claims:

- “Does it pay to disobey?” as a title or broad contribution.
- Students can improve by deviating from teachers.
- Tool necessity is model-specific.
- OPD can shift or over-call at the tool boundary.
- Teacher supervision should sometimes be ignored.
- Teacher-preferred tokens can deserve negative credit.
- Not all teacher disagreement is locally learnable.
- Passage utility is LLM-specific.

Provisional questions with room for a contribution, subject to a final
date-bounded search and the observed results:

- do teacher and target-student **forced-action advantages** imply different
  cost-optimal boundaries?
- can following a teacher have negative target-student causal value?
- which aspects of action value, if any, transfer across models?
- does target-student feedback make teacher information useful, and at what
  sample cost relative to student-only learning?
- do utility and teachability form distinct empirical axes, including a
  harmful-but-easy-to-distill regime?

## Primary-source and code custody

| Source | PDF in EIT | Code/artifact status |
|---|---|---|
| [[student-teacher-deviations]] | checksummed | No official code located. |
| [[model-adaptive-tool-necessity]] | checksummed | Official Git checkout pinned; official HF dataset revision recorded; no detected license. |
| [[tool-call-boundary-drift]] | checksummed | Authors say release is planned after internal review; not yet available. |
| [[reward-gated-opd]] | checksummed | Advertised GitHub URL returns 404. |
| [[craft-counterfactual-credit]] | checksummed | Repository withheld during double-blind review. |
| [[token-teachability]] | checksummed | Official TA-OPD Git checkout pinned; published result artifacts absent. |
| [[llm-specific-utility]] | checksummed | Anonymous 4open.science code snapshot archived; result/label outputs deferred until deanonymization. |

Exact PDF hashes, Git commits, and anonymous-snapshot checksums live under
`wiki/literature/manifests/`; large primary sources remain in the persistent
EIT vault.

## Decision

The niche survives, but only in its causal, target-specific form. The next
scientific action, when separately authorized, should be a small forced-action
measurement pilot—not a new loss, a large OPD run, or a claim based on prior
Legal-RAG outcomes.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] ·
[[compute-elasticity-distillation]] · [[three-dial]] ·
[[effort-conditioned-resource-allocation]] · [[opd-distillation]]
