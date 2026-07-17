---
title: Student-Specific Action Value — Auditing Teacher Policy Transfer
type: concept
tags: [distillation, tool-use, causal-utility, policy-transfer, three-dial]
created: 2026-07-17
updated: 2026-07-17
status: research candidate — literature-audited, no experiment launched
---

# Student-specific action value and teacher policy transfer

## Bottom line

There is a promising paper here, but it is narrower than the slogan “when
should a student disobey its teacher?” Generic teacher disagreement, selective
distillation, and students surpassing teachers are already crowded topics. The
defensible object is:

> **A strong teacher's action can be correct for the teacher and still be wrong
> for the student, because the value of an external action depends on the model
> that must execute and use it.**

The right starting posture is **questions first, measurement second, paper
shape later**:

1. ask whether and how the same action's value differs across acting models;
2. collect repeated forced-action outcomes and rich execution/cost metadata;
3. examine several possible structures—agreement, reversals, ranking,
   thresholds, advice quality, teachability—without choosing the answer;
4. add training or calibration only if the measurement pass exposes a stable
   question they can answer.

One question-generating hypothesis, not a thesis to assume, is:

> **Do values or task structure transfer better than action labels, and when
> does the acting model need its own boundary?**

No experiment was launched in this research pass.

The concise decision, priority relative to the skill-lifecycle direction, and
smallest useful experiment are recorded in
[[research-question-recommendation-2026-07-17]]. This page remains the full
technical design.

The complete seven-paper primary-source review, code-custody status, and
cross-paper synthesis are in
[[action-value-transport-reading-packet-2026-07-17]].

## Important title and novelty warning

The exact rhetorical territory is not new. Nagarajan et al.'s NeurIPS 2023
paper [[student-teacher-deviations]] is titled “On student-teacher deviations
in distillation: does it pay to disobey?” It studies confidence and implicit
bias in ordinary predictive distillation, not agent action utility, but it
means we should not reuse “does it pay to disobey?” as our title or broad
novelty claim.

Likewise, the following broad claims are closed:

- “tool necessity depends on the model”;
- “distillation can shift a tool-call boundary”;
- “teacher supervision can be harmful and should sometimes be gated”;
- “students can surpass teachers”;
- “tool/retrieval trajectories can be distilled across scale”;
- “a policy can respond to cost, budget, or preference conditions.”

The candidate research space survives only at their intersection:
**cross-scale, causal, cost-sensitive action-value transport**. Its eventual
contribution could be a phenomenon, null boundary, taxonomy, dataset, metric,
empirical relationship, simple calibration, or training result.

## Formal object

Start with one binary external intervention: the harness either injects one
canonical deterministic-computation result/fixed evidence payload or does not.
Let reader configuration `r` include the exact checkpoint, harness, system
prompt, renderer/tool adapter, decoding protocol, and tokenizer; freeze and
version evaluation metric `q`. For item `x` and action `a` in `{internal,
external}`, define the forced-action task-reward expectation

`p_(r,q)^a(x) = E[Y_q | do(a), r, x]`.

The reader-specific success/reward advantage is

`A_(r,q)(x) = p_(r,q)^external(x) - p_(r,q)^internal(x)`.

Record a cost vector `C` separately and define incremental cost

`DeltaC_r(x) = E[C | do(external),r,x] - E[C | do(internal),r,x]`.

For deployment price vector `lambda`, define net advantage

`V_r(x,lambda) = A_(r,q)(x) - lambda^T DeltaC_r(x)`.

Define `a_r*(x,lambda)` to choose external iff `V_r(x,lambda) > 0`. This permits nonzero internal
cost and reader-dependent incremental cost. The simpler `A > lambda*c` rule
is only a fixed-cost special case.

The teacher–student disagreement set is

`D(x, lambda) = 1[a_T*(x, lambda) != a_S*(x, lambda)]`.

If the student obeys the teacher's oracle action, its per-item regret is

`R_T->S(x, lambda) = |V_S(x,lambda)| * D(x, lambda)`.

Let `pi_T^lambda(x) = 1[V_T(x,lambda)>0]` be the forced-outcome teacher
oracle, with analogous target oracle `pi_S*`. For policy value
`J_S(pi;lambda) = E[Y_q - lambda^T C | actions chosen by pi]`, the primary
transport test is held-out regret

`Regret_T->S(lambda) = J_S(pi_S*;lambda) - J_S(pi_T^lambda;lambda)`.

Fit/estimate the teacher rule on disjoint items or repeat folds and evaluate
the target value with held-out items and independent outcome repeats. Do not
call the teacher oracle its observed policy; free-choice behavior is a
separate metacognitive measurement.

There are actually two teacher policies worth separating:

- `a_T^self`: what the teacher would do for itself;
- `T_→S`: what the teacher recommends for a named student after receiving only
  training-side information about that student's capabilities.

The first is a demonstration; the second is advice. Their empirical difference
tests whether a teacher can model the target rather than merely export its own
policy.

Because item-level advantages are noisy, the primary empirical estimand for an
actual advice policy should be its randomized causal value, not accuracy
against hard per-item oracle labels. Then

`Gamma_T→S(lambda) = E[(2*T_→S(x)-1) * V_S(x, lambda)]`.

This is the student-utility difference between following the binary teacher
recommendation and taking the opposite action in the prespecified population.
Negative `Gamma` means disobedience is beneficial on average. Also report the
weighted harmful-advice mass

`E[|V_S| * 1[T_→S != a_S*]]`,

which distinguishes consequential disagreement from harmless near-ties.

This makes it possible to test when action imitation is a poor default. The
teacher's oracle action is a thresholded statement about `A_T`; the target
decision depends on `A_S` and its incremental costs.
In the special case of equal unit incremental cost and both advantages inside
the evaluated nonnegative price
range, integrating student regret over all prices between the two thresholds
gives

`integrated regret = 0.5 * (A_T(x) - A_S(x))^2`.

Thus response-curve mismatch is not merely a behavioral disagreement. It is a
direct consequence of transferring the wrong model's action value. For
negative advantages or a truncated price interval, compute the integral
numerically rather than using the simplified identity.

### Four distinct gaps must not be conflated

1. **Teacher metacognition gap:** the teacher's observed free-choice action
   differs from its own forced-outcome optimum.
2. **Advice-targeting gap:** the teacher's recommendation for a named student
   fails to account for the student's capability profile.
3. **Policy transport gap:** the teacher's optimum differs from the student's
   optimum because `A_T != A_S`.
4. **Student execution gap:** the trained student's action differs from its own
   optimum.

Current tool-use work usually studies the teacher/student metacognition or
execution gaps. The candidate paper is about advice targeting and policy
transport, and how distillation changes student execution.

## The four action-value regimes

| Teacher optimum | Student optimum | Interpretation |
|---|---|---|
| internal | internal | Safe agreement; external action is not worth its price for either model. |
| external | external | Safe agreement; external action is valuable for both. |
| internal | external | **Inherited underuse:** the teacher can solve internally, but the weaker student needs help. |
| external | internal | **Inherited overuse:** the teacher can exploit the tool/evidence, but the student cannot convert it, or the action harms the student. |

The last regime matters. Model scale does not imply a monotone action-value
ordering because the external arm also depends on tool execution, evidence
integration, susceptibility to distraction, and verification ability.

## Closest primary work and the remaining distinction

| Work | What it already establishes | What remains different here |
|---|---|---|
| [[model-adaptive-tool-necessity]] | Tool necessity and call behavior differ by model; large knowing–doing gaps exist. | Necessity is derived from repeated no-tool reliability. It does not estimate forced tool benefit, cost, or distillation regret. |
| [[tool-call-boundary-drift]] | Multi-teacher OPD shifts the call/no-call boundary and can induce over-calling. | Uses fixed dataset `should-call` labels rather than the acting student's causal utility. |
| [[reward-gated-opd]] | Correctness can identify teacher signals that should be ignored. | Gates realized trajectories; it does not compare teacher- and student-specific counterfactual action values. |
| [[craft-counterfactual-credit]] | Signed counterfactual credit can push toward or away from teacher-preferred tokens. | Self-distilled token credit over sibling rollouts, not cross-scale external-action policy transport. |
| [[token-teachability]] | Teacher disagreement can be learnable or incompatible with student support. | Measures distributional learnability, not whether the teacher's action maximizes student utility. |
| [Demystifying OPD](https://arxiv.org/abs/2607.13399) | A stronger teacher can produce counterproductive guidance under student–teacher mismatch. | Studies signal and reasoning-distribution mismatch, not agent-indexed action payoffs. |
| [Learning Beyond Teacher / G-OPD](https://arxiv.org/abs/2602.12125) | Reward extrapolation can move students beyond teacher performance. | No explicit costly action or teacher-versus-student action-value disagreement. |
| [Student-Informed Teacher Training](https://arxiv.org/abs/2412.09149) | A privileged teacher can choose behavior a partially observed student cannot imitate. | Changes the teacher to be imitable; the mismatch is observability, not same-action, capability-conditioned utility. |
| [Distilling Realizable Students from Unrealizable Teachers](https://arxiv.org/abs/2505.09546) | Privileged expert policies can be unrealizable for students. | Again concerns information asymmetry and recoverability rather than a teacher-optimal action that is executable but student-suboptimal. |
| [Agent Distillation with Retrieval and Code Tools](https://arxiv.org/abs/2505.17612) | Full retrieval/code trajectories transfer across scale. | Uses a fixed tool regime and does not evaluate model-specific action boundaries or price. |
| [Rational Metareasoning](https://arxiv.org/abs/2410.05563) | Computation should be selected by expected value under cost. | No cross-scale teacher or distillation. |
| [Agents Should Invoke Tools Only When Epistemically Necessary](https://arxiv.org/abs/2506.00886) | Knowledge boundaries and appropriate effort allocation are model-specific. | A position/theory paper; it does not causally audit cross-scale policy transfer. |
| [[llm-specific-utility]] | The same retrieved passage has non-transferable utility across readers. | No teacher-policy distillation, action cost, or cross-scale policy regret. |

As of 2026-07-17, the focused primary-source search found no paper jointly
estimating teacher and student forced-action advantages, pricing the action,
and measuring the regret induced by distilling the teacher's boundary. This is
a provisional novelty statement, not a priority claim; it requires another
venue-date search before submission.

## Research questions

### RQ1 — Does a nontrivial disagreement set exist?

How often do `A_T` and `A_S` imply different optimal actions, across tasks,
prices, model families, and teacher–student scale gaps?

Hypothesis: the strongest asymmetry will often be `teacher=internal,
student=external`, but retrieval/evidence integration will also produce the
reverse regime.

### RQ2 — What transfers across scale?

Compare transfer of:

- teacher action labels;
- teacher free-choice probabilities;
- teacher forced-action advantage `A_T`;
- teacher advantage ranking over items;
- teacher rationale or skill-conditioned features;
- simple task difficulty and surface features.

Compare the teacher's self-policy against explicitly target-conditioned advice.
For the latter, give the teacher a compact capability card estimated only from
the student's training split. This tests whether the problem is naive policy
export or a deeper inability to predict another model's action value.

Hypothesis: ordinal task structure transfers better than a common action
threshold. If so, a small student calibration set should outperform direct
teacher-action imitation.

### RQ3 — Does standard distillation import the wrong boundary?

On the predeclared disagreement set, do trace SFT, conditional SFT, or OPD move
the student toward teacher actions while reducing the student's own utility?

The decisive pattern is not lower aggregate accuracy. It is **higher teacher
agreement together with higher student regret** on `D`.

### RQ4 — Can teacher information save student interventions?

At matched numbers of forced student rollouts, does a teacher-informed value
predictor or calibrator estimate `A_S` better than a student-only predictor?

This is the economic reason to call the procedure distillation. If estimating
`A_S` from scratch is equally data-efficient, the teacher adds nothing.

### RQ5 — Does free-choice behavior match each model's own optimum?

After the forced-action audit, expose prices in ordinary tool-use prompts and
measure each model's own necessity-to-action or value-to-action gap. This is a
separate metacognitive result and should not define the causal labels.

### RQ6 — Does the phenomenon extend from deterministic tools to evidence?

Use a fixed evidence-set action to test whether policy transport failure also
appears when value depends on reader conversion, distraction, and evidence-set
quality.

## Discovery-first study; paper shape follows the evidence

The contribution does **not** need to be a new agent architecture or loss. A
clean study first builds an empirical causal map of how external-action value
changes across model capability. Do not decide in advance that the output must
be a metric, scaling/transfer law, or calibration setup. Those are possible
findings if the data support them, alongside a taxonomy, benchmark, negative
result, or sharply delimited safety condition.

The earlier Legal-RAG reader crossover is only a hypothesis generator. It must
not be treated as evidence for the cross-scale claim or used to choose the
confirmatory tasks, prices, or model pairs. The primary study should be freshly
generated and benchmark-agnostic; legal evidence acquisition can be a later
external-validity test.

### Useful pre-specified causal summary: teacher-following value

For a teacher policy `pi_T→S` and target student `S`, define

`TFCV_T→S(lambda) = E[(2*pi_T→S(x,lambda)-1) * V_S(x,lambda)]`.

This is the target student's expected utility from following the teacher rather
than taking the opposite action. Positive values favor obedience; negative
values favor systematic disobedience in the prespecified population. It is
estimable directly from randomized forced actions and does not require turning
noisy item effects into hard oracle labels.

Always compare it with the adaptive gain over the better constant policy:

`J_S(pi_T→S) - max(J_S(always external), J_S(always internal))`.

Otherwise a teacher that always recommends the globally better action can look
adaptive without transporting any item-specific information.

### Optional normalized summary: advantage-weighted transport

A normalized secondary score is

`AWTS_T→S = E[(2*pi_T→S-1)V_S] / E[|V_S|]`.

It ranges from `-1` to `1` and weights action mistakes by their target-student
consequence rather than counting all disagreements equally. Report raw TFCV as
primary because the absolute-effect denominator can be noise-inflated; estimate
AWTS with split-sample signs and magnitudes or posterior integration.

For an oracle source boundary, the equivalent price-integrated action-value
transport regret is

`ITR_T→S = E_(x,lambda)[|V_S| * 1[pi_T* != pi_S*]]`.

This is mathematically advantage-weighted policy regret, not a new species of
CATE or regret. The novelty claim must rest on constructing it from repeated,
model-indexed forced actions and using it to audit cross-scale policy transfer.

### One empirical pattern to test, not optimize toward

One interpretable candidate is:

> **Action-value rankings partially transfer across model capability, while
> cardinal values and utility-maximizing action thresholds do not.**

Separate three empirical questions:

1. **Ordinal transfer:** do teacher and student action advantages have stable
   held-out rank correlation?
2. **Zero-shot threshold transfer:** how much student regret comes from using
   the teacher's uncalibrated boundary?
3. **Calibrated transfer:** can a small affine or isotonic map from teacher
   advantage to student advantage remove that regret using fewer student
   interventions than student-only learning?

Possible outcomes include partial ordinal transfer with threshold displacement,
near-perfect policy transfer, no useful relationship, non-monotonic reversals,
or family/task-specific structure. The experiment should distinguish these
rather than treating one as the desired result.

A stronger capability-normalized law can be tested by first fitting unaided
item response as

`logit P(Y_mi^internal=1) = theta_m - b_i`

and asking whether external-action gain collapses onto relative difficulty
`z = b_i - theta_m`:

`Delta_mia = g(z, action_strength, payload_quality)`.

The substantive hypothesis is a **zone of actionable difficulty**: little gain
when an item is already easy, increasing gain when external help can rescue the
model, and declining or negative gain when the model cannot convert the payload.
This curve should be fit flexibly and must predict held-out model sizes and a
held-out family; it should not be assumed in advance.

### Fresh confirmatory matrix

A claim about scale needs at least four sizes, more than one family, more than
one task/action type, and held-out predictive validation. A clean staged matrix
is:

- one dense instruction-tuned family at four or more scales for the primary
  ladder;
- a second family with at least three scales as the architecture holdout;
- fresh exact-scored task generators with three external-action families:
  deterministic compute, fixed information acquisition, and verification;
- a common `{none, cheap, rich}` action menu with deterministic payloads;
- randomized forced action for every `(reader configuration, item, action)`
  cell, repeated generations, and cross-fitting of policy construction versus
  evaluation;
- difficulty, payload quality, and price chosen without inspecting the earlier
  Legal-RAG effects.

Useful diagnostic views—not a predetermined final figure set—include:

1. a directed teacher-by-target TFCV or transport-regret heatmap;
2. transport versus measured capability gap, including directionality;
3. teacher-advantage quantile versus target advantage for every target scale;
4. oracle action-switch displacement over hypothetical price;
5. teacher-informed versus student-only regret as a function of the number of
   student forced-action labels.

Call any result a scaling law only if a discovered capability-normalized curve
predicts untouched sizes and transfers across families. Otherwise describe the
observed heterogeneity without forcing that label.

## Experiment program

## Phase A — forced-action audit before any training

Phase A is a question-generating and validity-establishing panel. Its schema
should support analyses beyond the first hypothesis while preserving a clean
held-out test for relationships discovered during exploration.

### A1. Smallest clean environment

- One same-family teacher/student pair, initially Qwen 8B-class to 1.5B–4B-class.
- Two procedurally distinct, exact-scored arithmetic/algorithmic families.
- One canonical result from a deterministic Python/calculator computation,
  injected identically through the harness.
- Two causal conditions per item: forced internal and forced canonical payload.
  A separate free-choice arm may measure metacognition; autonomous tool-call
  generation is a later tool-access/execution estimand.
- Matched task prompt, output-token cap, decoding policy, payload
  format/position, evaluation metric, and verifier across readers and arms.
- Frozen train/calibration/test generator seeds and held-out templates.

This two-model study is only a measurement pilot. It can reveal failure modes,
variance, and promising questions, but cannot establish a general cross-model
claim. Any relationship mined from it needs a fresh multi-scale,
multi-family confirmation.

The forced arms should not display a price. Estimate action outcomes and
incremental costs once and analytically rescore them over price vectors. This
is an oracle cost-sensitivity curve, not behavioral price response.
Price-conditioned free choice with displayed randomized prices is a separate
behavioral intervention.

### A2. Repeated outcomes

Use repeated stochastic generations for every
`(item, reader configuration, forced action)`.
An initial 4–8 repeats per cell is a screening range, not a final power claim.
Allocate additional repeats to near-boundary or high-uncertainty items using a
predeclared rule. Fit a hierarchical/binomial model or use empirical-Bayes
shrinkage rather than turning one Bernoulli draw into an item oracle.

Even 20 samples per arm leave roughly 16 percentage points of worst-case
standard error for an individual difference near `p=0.5`. Therefore do not
declare item-level signs from a handful of rolls. Make `Gamma`, policy value,
task/reader conditional effects, and weighted harmful-advice mass primary;
use posterior abstention near zero and treat hard per-item oracle agreement as
secondary.

The statistical unit remains the item. Confidence intervals should cluster by
item and generator family; model/action interactions and disagreement rates
must be reported per family before pooling.

### A3. Price construction

For binary success and a unit-cost single call, prices at or above 1 are
mechanically dominated. Do not reuse the earlier `{0, .25, 1, 4, 8}` grid.

Choose evaluation prices from quantiles of the observed training-side
advantage distribution and keep them inside the nontrivial switching region.
Because forced outcomes can be rescored analytically, report a dense oracle
cost-sensitivity curve rather than a few arbitrary points. Reserve “price
response” for the separate displayed-price free-choice experiment.

### A4. Phase-A outputs

- distribution of `A_T`, `A_S`, and `A_S - A_T`;
- rank correlation and calibrated mapping between teacher and student value;
- sign/action disagreement rate versus price;
- all four action-value regimes;
- student regret from teacher-oracle obedience;
- causal value `Gamma` of teacher self-policy and target-conditioned advice;
- weighted harmful-advice mass and advice calibration;
- teacher and student free-choice regret to their own optima;
- heterogeneity by task family, difficulty, and surface features.

## Phase B — characterize whatever transfer structure emerges

Run Phase B only if Phase A reveals reproducible structure worth predicting.
Treat it as a characterization and sample-efficiency study, not yet an
LLM-training paper. Lightweight routers, value heads, affine maps, and isotonic
maps are probes of the structure—not predetermined methods the project must
produce.

### Candidate comparison menu

1. always external and always internal;
2. task-family, prompt-length, and declared-difficulty heuristics;
3. student no-tool consistency/confidence;
4. model-adaptive no-tool necessity label;
5. teacher free-choice action imitation;
6. teacher self-policy versus explicitly student-targeted advice;
7. targeted advice with a training-only student capability card;
8. teacher-oracle action imitation;
9. direct thresholding of `A_T`;
10. one global offset or affine map from `A_T` to `A_S`;
11. isotonic/rank calibration;
12. question-only student value predictor;
13. student-only predictor trained on the same number of student forced pairs;
14. teacher-informed predictor with the same student-label budget.

### Core evaluation views

Retain ordinary target success, action rate, cost, policy value, regret over a
predeclared price distribution, teacher agreement, and uncertainty. No single
new metric has to be the contribution. Teacher agreement must not substitute
for target outcomes: a policy can agree with the teacher more often and still
be worse for the student.

### Sample-efficiency curve

If prediction is supported by Phase A, evaluate teacher-informed and
student-only predictors at increasing fractions of student counterfactual
labels. One possible positive result would be:

> Teacher advantage supplies transferable ordering, while a small number of
> student interventions relocates the threshold and reaches a given regret
> with materially fewer student rollouts.

## Phase C — distillation only if the measurements raise a training question

If Phases A and B reveal a substantial phenomenon that static analysis cannot
answer, compare:

1. base student with prompt-only/free-choice tool access;
2. teacher-action or teacher-trajectory SFT;
3. conditional trace SFT with price;
4. direct student task-reward training;
5. vanilla OPD as a failure diagnostic;
6. Reward-Gated OPD or another correctness-aware selective baseline;
7. a student-calibrated action router or supervision mask.

Do not begin by inventing another scalar OPD gate. RG-OPD, CRAFT, SDAR,
Token-Teachability, RL-aware distillation, and recent OPD-regulation work make
that method space crowded. A new training loss is justified only if the causal
audit exposes a replicated failure that value calibration and direct task RL
cannot fix.

The student's action value can change during training. Re-estimate or audit
`A_S` at checkpoints; do not assume labels from the base student remain the
post-training optimum.

## Clean connection to the three-dial program

This idea should not become a fourth dial. It is a sharp diagnostic of **reader
conversion**, the second dial in [[three-dial]].

For a fixed evidence set `E`, reader configuration `r`, and metric `q`, define

`A_(r,q)(x,E) = P(correct | r,x,E,q) - P(correct | r,x,no evidence,q)`.

Expansion and selection determine which evidence set is available. Reader
conversion determines `A_(r,q)`. Effort/cost determines whether that value exceeds
the retrieval/context price. A teacher can therefore transfer useful search
structure while still supplying the wrong deployment action:

- a strong teacher may say “do not retrieve” because it already knows the
  answer, while the student needs evidence;
- a teacher may say “retrieve” because it can integrate a complex statute or
  conflicting set, while the student is distracted or misapplies it.

The existing July 2 reader-size crossover is a strong hypothesis generator,
not yet causal proof:

- BarExam: best evidence versus no evidence was `-2.5pp` for the 70B reader but
  `+11.8pp` for the 8B reader;
- Housing: it was `+11.4pp` for the 70B reader but `-2.8pp` for the 8B reader.

See [[judge-answer-conversion]] and the source-gated summary in
[[research-state-2026-07-17]]. These are single-generation paired subset
results. The Housing subset differs from the full-N baseline, and the apparent
per-question oracle in [[offline-bandit-v0]] is noise-inflated. Repetition is
mandatory.

### Smallest legal-RAG stress test

1. Freeze one identical evidence set per question.
2. Restrict the decision to `{no evidence, receive this set}`.
3. Run repeated outcomes for 70B and 8B readers on BarExam and Housing.
4. Ask the 70B reader both what it would do itself and what a specified 8B
   reader should do; optionally supply a capability card built only from the
   8B training split.
5. Treat the 70B oracle action, self-policy, and target-conditioned advice as
   distinct teacher policies and estimate their randomized value for 8B.
6. Compare teacher imitation and direct student utility prediction; add simple
   teacher-value calibration only if the main pilot shows transferable teacher
   structure.
7. Stratify by gold present/absent, evidence-set quality, and context length.

This connects the projects elegantly without importing tree search, conflict
arbitration, variable `k`, or a multidimensional controller into the first
study.

## Decision gates after the pilot

### Continue the broader study if one or more robust questions emerge

Examples include:

- stable cross-model action-value heterogeneity or directional reversals;
- teacher self-policy and target-conditioned advice having different target
  value;
- teacher signals predicting target advantage beyond simple item difficulty;
- utility and teachability separating in consequential cases;
- a simple or complex cross-model relationship surviving untouched data;
- a clear null boundary showing when policy transfer is safe;
- a training intervention changing teacher agreement and target utility in
  different directions.

Partial rank transfer with threshold displacement is only one interesting
regime. Near-identical values, unrelated values, family-specific patterns, and
stable null effects answer different scientific questions and should not be
treated as failed attempts to obtain the preferred story.

### Kill or sharply downgrade if

- teacher/student optimal actions almost always agree;
- disagreements vanish with repeated sampling;
- task family, prompt length, or declared difficulty explains the result;
- post-hoc scalar calibration solves everything and the paper claims a complex
  neural method;
- no model, task, advice, or training comparison yields structure beyond
  sampling noise and simple item covariates;
- the effect exists only in synthetic arithmetic;
- only observed free-choice actions, rather than forced potential outcomes,
  show a difference;
- a current paper is found that already measures student-specific forced-action
  value and cross-scale distillation regret.

## Possible contribution forms, selected after observing robust results

- **Measurement resource:** a repeated, model-indexed forced-action panel.
- **Empirical phenomenon or taxonomy:** robust regimes of helpful, harmful,
  irrelevant, or target-dependent teacher advice.
- **Null/safety boundary:** conditions under which teacher policy transfer is
  already reliable.
- **Metric or evaluation protocol:** only if existing policy-value summaries
  fail to capture an important repeated pattern.
- **Predictive or scaling relation:** only if it forecasts untouched sizes,
  families, or tasks.
- **Simple calibration or method:** only if teacher structure exists and target
  feedback exploits it more efficiently than student-only learning.
- **Training result:** only if standard distillation creates a replicated
  failure that simpler controls do not answer.

## Relationship to optimized skill artifacts

The separate [[skill-lifecycle-research-snapshot-2026-07-17]] asks whether a
skill optimized for source-model contextual execution is also good context or
teaching material for a target student. That question intersects this program
only when the skill encodes a costly action policy such as search, retrieval,
verification, or tool use.

In that case, source-side SkillOpt may encode a boundary like
`external iff A_T(x) > price`, while the target-optimal boundary depends on
`A_S(x)`. Distilling or directly copying the artifact can increase agreement
with the teacher while increasing target-student regret. This gives the skill
pipeline a sharp diagnostic target, but it does not make the pipeline itself
novel: [[opcd]] already covers cross-size context distillation and
[[seed-self-evolving-opd]] already covers self-evolving, gap-gated skill OPD.

Keep the programs separable:

1. establish repeated student-specific forced-action values first;
2. if stable teacher/target reversals exist, test whether source-optimized
   skills import the wrong boundary;
3. only then ask whether a small target-student calibration set can retarget
   the artifact more efficiently than direct target learning.

If the forced-action audit yields no causal heterogeneity, do not use the
skill chain to manufacture complexity. If it yields robust reversals, the
action-value question remains the primary science and skill placement becomes
a controlled mechanism/ablation.

This must be distinguished from established student-friendly distillation.
[[lgtm-student-level-kd]], [[promptkd]],
[[personalized-teacher-selection]], [[distillation-traps-guards]], and
[[token-teachability]] already show that teacher quality, compatibility, and
learnability differ. They do not estimate the acting target's repeated
forced-action potential outcomes or regret from obeying the teacher's
utility-maximizing action. The causal action-value estimand—not the slogan that
teachers differ in teachability—is what remains distinctive here.

Possible later title directions, only if the corresponding finding emerges:

- **Whose Tool Boundary? Causal Auditing of Cross-Scale Agent Distillation**
- **Distill Values, Not Calls: Student-Calibrated Tool Policy Transfer**
- **The Teacher's Policy Is Not the Student's Policy**
- **Capability-Calibrated Agent Distillation**
- **Causal Action-Value Transport Across Language Models: Rankings Transfer,
  Decision Boundaries Do Not**

## Recommendation

This idea is worth pursuing as a gated research program. The next scientific
step, when authorized, is not OPD training and not a large price sweep. It is a
small forced-action audit designed to reveal how teacher and target action
values relate, including the possibility that no simple transport structure
exists.

Do not build the main claim on the earlier BarExam/Housing reader crossover.
Those observations can motivate reader-conditioned utility and later serve as
an external-validity test, but this project is sufficiently different that its
primary evidence should come from the fresh forced-action scale ladder.

When experiments are authorized, begin with a small measurement pilot on
deterministic actions. Collect rich outcomes, behavior, execution, and cost
telemetry; distinguish planned analyses from exploratory mining; then use a
fresh multi-scale, multi-family matrix to confirm whatever patterns merit
follow-up. Add fixed evidence acquisition as a distinct action family rather
than as the evidentiary foundation.

For the three-dial paper, use student disobedience as a reader-conditioning
analysis. For a separate domain-general distillation paper, use Python as the
clean first environment and fixed evidence acquisition as the strongest
second action family.
