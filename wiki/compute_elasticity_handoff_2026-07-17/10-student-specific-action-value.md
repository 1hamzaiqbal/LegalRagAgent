---
title: Student-Specific Action Value — When Teacher Policies Should Not Transfer
type: concept
tags: [distillation, tool-use, causal-utility, policy-transfer, three-dial]
created: 2026-07-17
updated: 2026-07-17
status: research candidate — literature-audited, no experiment launched
---

# Student-specific action value

## Bottom line

There is a promising paper here, but it is narrower than the slogan “when
should a student disobey its teacher?” Generic teacher disagreement, selective
distillation, and students surpassing teachers are already crowded topics. The
defensible object is:

> **A strong teacher's action can be correct for the teacher and still be wrong
> for the student, because the value of an external action depends on the model
> that must execute and use it.**

The strongest paper shape is therefore **measurement first, method second**:

1. causally estimate the value of a fixed external action for both teacher and
   student using repeated forced-action outcomes;
2. identify the inputs and prices where their utility-maximizing actions differ;
3. measure whether trace SFT or OPD imports the teacher's wrong-for-the-student
   boundary;
4. test whether teacher information can reduce the number of student
   counterfactual trials needed to learn the student's own boundary.

The clean thesis is:

> **Distill values or task structure, not action labels. A costly action boundary
> must be calibrated to the model that will act.**

No experiment was launched in this research pass.

## Important title and novelty warning

The exact rhetorical territory is not new. Nagarajan et al.'s NeurIPS 2023
paper is titled [“On student-teacher deviations in distillation: does it pay to
disobey?”](https://arxiv.org/abs/2301.12923). It studies confidence and implicit
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

The candidate opening survives only at their intersection: **cross-scale,
causal, cost-sensitive action-value transport**.

## Formal object

Start with one binary external action, such as one Python call or receiving one
fixed evidence set. For model `m`, item `x`, and action `a` in `{internal,
external}`, define the forced-action success probability

`p_m^a(x) = E[Y_m | do(a), x]`.

The model-specific external-action advantage is

`A_m(x) = p_m^external(x) - p_m^internal(x)`.

For one action with cost `c` and deployment price `lambda`, the transparent
utility-optimal policy is

`a_m*(x, lambda) = external iff A_m(x) > lambda * c`.

The teacher–student disagreement set is

`D(x, lambda) = 1[a_T*(x, lambda) != a_S*(x, lambda)]`.

If the student obeys the teacher's oracle action, its per-item regret is

`R_T->S(x, lambda) = |A_S(x) - lambda*c| * D(x, lambda)`.

There are actually two teacher policies worth separating:

- `a_T^self`: what the teacher would do for itself;
- `T_→S`: what the teacher recommends for a named student after receiving only
  training-side information about that student's capabilities.

The first is a demonstration; the second is advice. A teacher should not be
expected to export its own policy unchanged if it can model the target student.

Because item-level advantages are noisy, the primary empirical estimand should
be the randomized causal value of the advice policy, not accuracy against hard
per-item oracle labels. Let `V_S(x, lambda) = A_S(x) - lambda*c`. Then

`Gamma_T→S(lambda) = E[(2*T_→S(x)-1) * V_S(x, lambda)]`.

This is the student-utility difference between following the binary teacher
recommendation and taking the opposite action in the prespecified population.
Negative `Gamma` means disobedience is beneficial on average. Also report the
weighted harmful-advice mass

`E[|V_S| * 1[T_→S != a_S*]]`,

which distinguishes consequential disagreement from harmless near-ties.

This exposes why action imitation is the wrong default. The teacher's action is
a thresholded statement about `A_T`; the deployment decision requires `A_S`.
With unit cost and both advantages inside the evaluated nonnegative price
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
| [Model-Adaptive Tool Necessity](https://arxiv.org/abs/2605.14038) | Tool necessity and call behavior differ by model; large knowing–doing gaps exist. | Necessity is derived from repeated no-tool reliability. It does not estimate forced tool benefit, cost, or distillation regret. |
| [Tool-Call Boundary Drift](https://arxiv.org/abs/2607.07050) | Multi-teacher OPD shifts the call/no-call boundary and can induce over-calling. | Uses fixed dataset `should-call` labels rather than the acting student's causal utility. |
| [Reward-Gated OPD](https://arxiv.org/abs/2607.04037) | Correctness can identify teacher signals that should be ignored. | Gates realized trajectories; it does not compare teacher- and student-specific counterfactual action values. |
| [CRAFT](https://arxiv.org/abs/2606.29476) | Signed counterfactual credit can push toward or away from teacher-preferred tokens. | Self-distilled token credit over sibling rollouts, not cross-scale external-action policy transport. |
| [Token Teachability](https://arxiv.org/abs/2605.26844) | Teacher disagreement can be learnable or incompatible with student support. | Measures distributional learnability, not whether the teacher's action maximizes student utility. |
| [Demystifying OPD](https://arxiv.org/abs/2607.13399) | A stronger teacher can produce counterproductive guidance under student–teacher mismatch. | Studies signal and reasoning-distribution mismatch, not agent-indexed action payoffs. |
| [Learning Beyond Teacher / G-OPD](https://arxiv.org/abs/2602.12125) | Reward extrapolation can move students beyond teacher performance. | No explicit costly action or teacher-versus-student action-value disagreement. |
| [Student-Informed Teacher Training](https://arxiv.org/abs/2412.09149) | A privileged teacher can choose behavior a partially observed student cannot imitate. | Changes the teacher to be imitable; the mismatch is observability, not same-action, capability-conditioned utility. |
| [Distilling Realizable Students from Unrealizable Teachers](https://arxiv.org/abs/2505.09546) | Privileged expert policies can be unrealizable for students. | Again concerns information asymmetry and recoverability rather than a teacher-optimal action that is executable but student-suboptimal. |
| [Agent Distillation with Retrieval and Code Tools](https://arxiv.org/abs/2505.17612) | Full retrieval/code trajectories transfer across scale. | Uses a fixed tool regime and does not evaluate model-specific action boundaries or price. |
| [Rational Metareasoning](https://arxiv.org/abs/2410.05563) | Computation should be selected by expected value under cost. | No cross-scale teacher or distillation. |
| [Agents Should Invoke Tools Only When Epistemically Necessary](https://arxiv.org/abs/2506.00886) | Knowledge boundaries and appropriate effort allocation are model-specific. | A position/theory paper; it does not causally audit cross-scale policy transfer. |
| [LLM-Specific Utility](https://arxiv.org/abs/2510.11358) | The same retrieved passage has non-transferable utility across readers. | No teacher-policy distillation, action cost, or cross-scale policy regret. |

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

## Preferred architecture-free paper shape

The contribution does **not** need to be a new agent architecture or loss. A
stronger and cleaner paper may be an empirical causal map of how external-action
value changes across model capability, together with a metric and a predictive
transport law.

The earlier Legal-RAG reader crossover is only a hypothesis generator. It must
not be treated as evidence for the cross-scale claim or used to choose the
confirmatory tasks, prices, or model pairs. The primary study should be freshly
generated and benchmark-agnostic; legal evidence acquisition can be a later
external-validity test.

### Primary causal quantity: teacher-following causal value

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

### Cross-task summary: advantage-weighted transport score

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

### Empirical law to test

The most compelling candidate is:

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

The desired pattern is high enough rank transfer to make the teacher useful,
poor enough threshold transfer to make action imitation wrong, and rapid
recovery after a small target-student calibration set. If either the first or
third condition fails, the distillation story is weak.

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
- randomized forced action for every `(model, item, action)` cell, repeated
  generations, and cross-fitting of policy construction versus evaluation;
- difficulty, payload quality, and price chosen without inspecting the earlier
  Legal-RAG effects.

The minimal decisive plots would be:

1. a directed teacher-by-target TFCV or transport-regret heatmap;
2. transport versus measured capability gap, including directionality;
3. teacher-advantage quantile versus target advantage for every target scale;
4. action-switch displacement over price;
5. teacher-informed versus student-only regret as a function of the number of
   student forced-action labels.

Call the result a scaling law only if the capability-normalized curve predicts
held-out sizes and transfers across families. Otherwise, call it a causal
transport audit.

## Experiment program

## Phase A — forced-action audit before any training

### A1. Smallest clean environment

- One same-family teacher/student pair, initially Qwen 8B-class to 1.5B–4B-class.
- Two procedurally distinct, exact-scored arithmetic/algorithmic families.
- One external Python/calculator action.
- Three conditions per item: forced internal, forced exactly one tool action,
  and free choice.
- Matched prompt, output-token cap, decoding policy, tool interface, and
  verifier across models and arms.
- Frozen train/calibration/test generator seeds and held-out templates.

This two-model study is only a measurement pilot. It cannot establish the
architecture-free paper claim above; a confirmatory run must use the fresh
multi-scale, multi-family matrix.

The forced arms should not display a price. Estimate action outcomes once and
analytically rescore them over all prices. Price-conditioned free choice is a
separate behavioral intervention.

### A2. Repeated outcomes

Use repeated stochastic generations for every `(item, model, forced action)`.
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
Because forced outcomes can be rescored analytically, report a dense price
curve rather than a few arbitrary points.

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

## Phase B — value transfer with a small student calibration set

Treat Phase B as a sample-efficiency experiment, not yet an LLM-training paper.
Train a lightweight router/value head on frozen item/model signals.

### Required baselines

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

### Primary metric

Use **student utility regret integrated over a predeclared price distribution**.
Report action accuracy and teacher agreement only as diagnostics. A policy can
agree with the teacher more often and still be worse for the student.

### Sample-efficiency curve

Evaluate teacher-informed and student-only predictors at increasing fractions
of student counterfactual labels. The important positive result would be:

> Teacher advantage supplies transferable ordering, while a small number of
> student interventions relocates the threshold and reaches a given regret
> with materially fewer student rollouts.

## Phase C — distillation only after the phenomenon passes

If Phases A and B establish a substantial, learnable transport gap, compare:

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

For a fixed evidence set `E`, define

`A_m(x, E) = P(correct | m, x, E) - P(correct | m, x, no evidence)`.

Expansion and selection determine which evidence set is available. Reader
conversion determines `A_m`. Effort/cost determines whether that value exceeds
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
6. Compare teacher imitation, direct student utility prediction, and a
   teacher-value-plus-student-calibration rule.
7. Stratify by gold present/absent, evidence-set quality, and context length.

This connects the projects elegantly without importing tree search, conflict
arbitration, variable `k`, or a multidimensional controller into the first
study.

## Go/no-go gates

### Continue to a paper if

1. `A_T` and `A_S` differ enough to create a stable disagreement set across at
   least two task families and more than one model pair or external action;
2. teacher signals predict student advantage beyond question-only difficulty;
3. teacher self-policy or target-conditioned advice causes measurable student
   regret on the disagreement set;
4. teacher-informed calibration beats a student-only predictor at a matched
   student-intervention budget;
5. the result survives held-out templates, repeated outcomes, and a strong
   scalar-threshold baseline.

The ideal transfer regime is partial: teacher and student advantages share
structure, but not a common threshold. If they are identical, disobedience is
unnecessary. If they are unrelated, the teacher is useless.

### Kill or sharply downgrade if

- teacher/student optimal actions almost always agree;
- disagreements vanish with repeated sampling;
- task family, prompt length, or declared difficulty explains the result;
- post-hoc scalar calibration solves everything and the paper claims a complex
  neural method;
- teacher signals do not reduce student counterfactual sample requirements;
- the effect exists only in synthetic arithmetic;
- only observed free-choice actions, rather than forced potential outcomes,
  show a difference;
- a current paper is found that already measures student-specific forced-action
  value and cross-scale distillation regret.

## Plausible paper contributions, in order

1. **Evaluation/estimand:** a forced-action, model-indexed audit of cross-scale
   tool-policy transport under cost.
2. **Empirical finding:** teacher policy fidelity can be anti-correlated with
   student utility on a stable disagreement set.
3. **Transfer law:** teacher value rankings partially transfer, but action
   thresholds do not.
4. **Simple method:** a few student counterfactuals calibrate teacher value more
   efficiently than learning the student's router from scratch.
5. **LLM-training method:** only if standard distillation exhibits an unresolved
   replicated failure after the simple controls.

Candidate title directions that avoid the NeurIPS 2023 collision:

- **Whose Tool Boundary? Causal Auditing of Cross-Scale Agent Distillation**
- **Distill Values, Not Calls: Student-Calibrated Tool Policy Transfer**
- **The Teacher's Policy Is Not the Student's Policy**
- **Capability-Calibrated Agent Distillation**
- **Causal Action-Value Transport Across Language Models: Rankings Transfer,
  Decision Boundaries Do Not**

## Recommendation

This idea is worth pursuing as a gated research program. The next scientific
step, when authorized, is not OPD training and not a large price sweep. It is a
small forced-action audit that tells us whether teacher advantage contains
transferable structure while teacher action labels create student regret.

Do not build the main claim on the earlier BarExam/Housing reader crossover.
Those observations can motivate reader-conditioned utility and later serve as
an external-validity test, but this project is sufficiently different that its
primary evidence should come from the fresh forced-action scale ladder.

When experiments are authorized, begin with a small measurement pilot on
deterministic actions, then run the multi-scale, multi-family confirmatory
matrix. Add fixed evidence acquisition as a distinct action family rather than
as the evidentiary foundation.

For the three-dial paper, use student disobedience as a reader-conditioning
analysis. For a separate domain-general distillation paper, use Python as the
clean first environment and fixed evidence acquisition as the strongest
second action family.
