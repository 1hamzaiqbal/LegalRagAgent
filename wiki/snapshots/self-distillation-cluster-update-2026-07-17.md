---
title: Self-Distillation Cluster Update - OPSD, SDFT, and SDPO
type: decision
tags: [self-distillation, opd, continual-learning, rich-feedback, action-value, skills]
created: 2026-07-17
updated: 2026-07-17
status: literature-integrated; no experiment launched
---

# Self-distillation cluster update - 2026-07-17

## Bottom line

The three January 2026 papers materially strengthen one conclusion and leave
the primary research recommendation intact:

1. **Generic context-to-weights self-distillation is decisively occupied.**
   Verified solutions, instance-specific demonstrations, environment feedback,
   and interaction history have all been used as privileged context for a
   same-model teacher whose behavior is distilled into an unconditioned
   student.
2. **Self-teaching has a capability boundary.** OPSD says privileged teaching
   fails beyond the model's comprehension frontier; SDFT loses to SFT at 3B
   but wins at 7B/14B; SDPO loses or ties at the smallest Qwen2.5 sizes and
   improves with scale.
3. **A contextual probability shift is not task value.** All three optimize
   token-distribution agreement. None estimates repeated task outcomes under
   forced internal versus external actions, action cost, or target regret from
   following another model's action rule.
4. **The primary pilot remains student-specific forced-action value.** The
   self-distillation papers become downstream baselines and diagnostics only
   after the no-training causal panel shows a stable phenomenon.
5. **The secondary skill question survives in narrower form.** The open object
   is not whether context can be baked into weights. It is whether several
   exact fixed procedural artifacts preserve their ordering between one
   target's runtime context and that same target's matched post-withdrawal
   acquisition.

No experiment was launched in this literature pass.

## The three papers are different mechanisms

| Paper | Privileged view | Teacher/student relation | Main learning signal | What it establishes |
|---|---|---|---|---|
| [[opsd-self-distilled-reasoner]] | Verified solution or reasoning trace | Same checkpoint; main teacher frozen at initialization | Full-vocabulary clipped forward KL on student rollouts | Gold-supported same-model reasoning can be distilled on-policy without decoding a teacher trajectory. |
| [[sdft-continual-learning]] | Instance-specific demonstration, article, answer, or tool call | Same model with an EMA teacher | Empirically forward KL on student prefixes, despite reverse-KL paper text | Demonstration-conditioned self-distillation can acquire skills/facts with much less forgetting than SFT in the tested settings. |
| [[sdpo-rich-feedback]] | Runtime/unit-test feedback and/or a successful peer rollout | Same current policy with EMA/trust-region regularization | Feedback-conditioned logit/token pseudo-advantages | Rich feedback can provide dense hindsight credit and accelerate training/discovery, especially for stronger models. |

They share a template:

1. obtain a student trajectory from the unprivileged policy;
2. expose a second contextual view of the same model to privileged information;
3. re-score the student's prefixes under that view;
4. move the unprivileged policy toward the privileged distribution; and
5. deploy without the privileged context.

That template is now prior art. Variants in context source, divergence,
teacher update, feedback construction, and task reward remain engineering or
ablation choices unless tied to a new measured phenomenon.

## A crucial three-object separation

These papers make it even more important not to collapse three quantities:

### 1. Acting utility

Does an external intervention improve this reader's task outcome **now**?

For reader configuration `r`, item `x`, and a fixed intervention, this is the
repeated forced-outcome contrast already defined in
[[research-question-recommendation-2026-07-17]]:

`A_r(x) = E[Y | do(external), r, x] - E[Y | do(internal), r, x]`.

This can be positive, zero, or negative and becomes net value only after
incremental cost is included.

### 2. Privileged-view teacher quality

Does conditioning a model on a solution, demonstration, skill, or feedback
actually create a more correct teacher for the relevant prefixes/items?

A large teacher-student KL is not enough. OPSD finds style tokens can dominate
the divergence; SDPO finds that including the failed attempt can reduce
entropy and performance; SDFT's 3B result shows that privileged conditioning
need not yield a useful learning signal for a weak reader.

Teacher quality should therefore be measured in task terms and with the
teacher context removed/varied, not inferred from distribution distance.

### 3. Acquisition utility

Does training from that privileged view improve no-context held-out behavior
relative to matched direct training, and what does it forget or distort?

This is the secondary skill snapshot's `U_int`. It depends on the training
algorithm, teacher update, divergence, budget, seed, and prior capabilities.
It is not implied by acting utility or contextual teacher quality.

The useful conceptual result is a **three-gate decomposition**, not a new
metric:

1. the intervention must help the named target when used;
2. the privileged view must produce trustworthy guidance for that target; and
3. the learning procedure must convert that guidance into durable behavior
   without unacceptable forgetting or boundary drift.

The first gate is the primary causal pilot. The latter two belong to a later
training phase.

## Empirical patterns worth preserving

### Capacity-relative self-teaching boundary

- OPSD's authors explicitly warn that privileged solutions stop being useful
  when problems exceed the model's comprehension threshold.
- SDFT's Science gap versus SFT changes from -3.3 at 3B to +4.0 at 7B and +6.9
  at 14B.
- SDPO loses at Qwen2.5-1.5B, roughly ties at 3B, and wins at 7B; the SDPO-GRPO
  gap also grows over Qwen3 0.6B, 1.7B, 4B, and 8B.
- A mostly-GRPO mixture helps SDPO's weakest reader but slightly hurts stronger
  readers.

This is suggestive of a capability-relative threshold, but it is not yet a
scaling law. Each paper uses a different task, objective, teacher context, and
evaluation protocol. A law claim would require prediction of untouched model
sizes and a held-out family.

### Feedback/context composition matters

- SDFT's article-plus-answer teacher reaches 89% strict knowledge accuracy,
  compared with 75% for article only and 37% for answer only.
- SDPO's environment feedback and peer solution are complementary; adding the
  student's failed attempt to both lowers trained accuracy from 48.3 to 44.5
  and lowers entropy.
- OPSD's student-thinking-off/teacher-thinking-on setup creates large
  divergence, but stylistic-token contributions exceed math-token
  contributions and require clipping.

“More privileged context” is therefore not a monotone intervention. Exact
bytes, ordering, renderer, and whether a failed attempt is included are causal
factors, not incidental prompt details.

### Teacher update and objective are regime-dependent

- OPSD's main teacher is frozen at initialization.
- SDFT and SDPO prefer regularized moving/EMA teachers in their settings.
- OPSD reports forward KL better than reverse KL/JSD for its reasoning setup.
- SDPO changes divergence by regime.
- SDFT's paper says reverse KL, but the official code states that every
  headline result used forward KL.

There is no evidence-backed universal rule such as “always freeze,” “always
synchronize,” or “always use reverse KL.” Cross these choices or pre-specify a
single baseline; do not elevate one into the contribution.

### Self-distillation reduces but does not erase forgetting

SDFT and SDPO both preserve prior capabilities better than their off-policy
comparisons, but neither establishes zero forgetting. SDFT's sequential claim
rests on one three-task order and normalized curves rather than raw standard
continual-learning endpoints. SDPO's holdout average after training is 42.4
versus a 43.5 base. Use “mitigates” or “substantially reduces,” not “solves.”

## Effect on the primary action-value direction

The primary question remains:

> When does a strong teacher's optimal costly external action become
> suboptimal for the student that must use it, and what structure in that
> action value transfers across model scale and family?

The trio does not answer it. In particular:

- OPSD always trusts the privileged teacher and has no abstention or negative
  credit.
- SDFT predicts/copies correct task behavior, not the causal benefit of forcing
  a tool/evidence action on a reader.
- SDPO's “advantage” is a feedback-conditioned token log-ratio, not a repeated
  task-reward contrast and not a price-sensitive value.
- None transfers a forced-outcome action oracle from one reader to another and
  evaluates held-out target regret.

These papers do suggest later analyses if the forced-action pilot succeeds:

1. Does the immediate student-specific action advantage predict whether
   privileged self-distillation helps or harms?
2. Does the self-teacher's task-quality gap predict acquisition better than
   raw teacher-student KL, target likelihood, [[token-teachability]], or
   [[informative-alignment-rsr]]?
3. Can value/reward gating prevent a teacher action boundary from being baked
   into a student for whom that boundary has negative regret?
4. Does distillation increase teacher-policy agreement while worsening target
   utility on the disagreement items?

Those are downstream questions. They are not reasons to add training to the
initial causal pilot.

## Effect on the secondary skill direction

The broad SkillOpt -> SKILL0 -> OPD story is even less defensible as novelty.
SDFT and SDPO explicitly demonstrate privileged-context/interaction-history
compression into weights, and OPSD supplies the verified-solution
self-distillation baseline.

The surviving secondary question remains:

> For one fixed target reader and several exact versioned procedural skills,
> does the target's contextual utility ordering predict the same target's
> no-context acquisition ordering after reset-from-base matched training?

This trio changes the baseline set and the interpretation:

- SDFT is a mandatory same-model demonstration/skill-conditioned
  internalization baseline.
- OPSD is a mandatory unconditional privileged-context OPD baseline.
- SDPO is a mandatory feedback/skill-conditioned self-teaching baseline if the
  task provides rich feedback.
- Every artifact arm must still be compared with matched direct SFT/RL and one
  pre-specified withdrawal method from the same base checkpoint.
- Independent training seeds, raw old-task endpoints, context-marker leakage,
  and compute amortization are required.

The contribution cannot be “self-distillation internalizes skills.” It would
have to be a replicated placement-ordering phenomenon, a target-specific
selection failure, or a strong cost-matched null.

## Three-dial interpretation

The cluster fits the existing three-dial lens without forcing a fourth dial:

1. **Expansion/exposure:** which privileged material is supplied - verified
   solution, demonstration, skill, peer success, runtime feedback, or a
   combination.
2. **Selection/credit:** which parts of that material or which token shifts are
   trusted, clipped, gated, or mixed with outcome reward.
3. **Conversion/reader:** whether the named checkpoint/harness can use the
   privileged view now and convert it into no-context behavior without harmful
   drift.

Cost remains attached across all three: context tokens and calls at inference,
teacher rescoring and rollout generation during training, and downstream
deployment reuse. The papers reinforce the three-dial framework as an analysis
language; they do not by themselves make the framework a contribution.

## Questions worth collecting data for - without precommitting to a paper form

If and only if the no-training pilot reveals stable action-value heterogeneity,
overcollect enough to ask:

- Where does the privileged self-teacher become outcome-better than the
  unprivileged reader, and is that boundary aligned with immediate action
  utility?
- Do sign/order reversals occur between acting utility, contextual teacher
  quality, and acquisition utility?
- Is model size merely a proxy for target ICL lift, or does the relationship
  hold within size after conditioning on teacher contextual performance?
- Which token-level signals predict useful training: forward/reverse KL,
  entropy change, style/task-token divergence, teacher correctness, RSR,
  student NLL, or forced action value?
- Does a reward/value gate help only weak readers, echoing SDPO's
  scale-dependent GRPO mixture?
- Are apparent continual-learning gains actually reduced distribution shift,
  shorter outputs, checkpoint selection, or task-order effects?
- Under matched lifecycle cost, is external context still preferable to weight
  internalization at realistic reuse/update rates?

Let those measurements determine whether the eventual paper is a causal
atlas, empirical threshold/rank regularity, calibration result, training
intervention, lifecycle analysis, or strong null. Do not name a metric or law
in advance.

## Decision update

The research ordering does not change:

1. run the architecture-free forced-action measurement pilot when ready;
2. characterize any real cross-reader disagreement or transport structure;
3. only then test self-distillation as a mechanism or intervention; and
4. keep fixed-artifact context-versus-acquisition as an independent secondary
   study, not a required stage of the primary project.

The most valuable new insight is a gate, not an architecture: **privileged
context is teachable only when the named reader can turn it into a better
teacher distribution, and even that does not imply that the underlying action
is useful for another reader.**

## Primary-source custody

| Source | PDF hash | Pinned official repository |
|---|---|---|
| [[opsd-self-distilled-reasoner]] | `e521ed6c2939dc612daec5fe6fb47bef554a291228c02cafae07102b9729bad8` | `OPSD` at `7448751f307a9cdbcc1246dd1565a1a605b443df` |
| [[sdft-continual-learning]] | `c27949d1b7888b128bf40f6b031d4e84a82b8ba40d6e03d671491f8e414a516b` | `Self-Distillation` at `d77573212fa0a3ae2eeb64b9b44db1c251f75e3e` |
| [[sdpo-rich-feedback]] | `2714e7734c43cf849c2e7c49fc95cf57a6533c0833465a01983b87eb7f72d190` | `SDPO` at `7c457fc1b1f636ae794eb0362ba37d4743b06fbc` |

All PDFs and repositories are in the persistent EIT literature vault. The
SDFT page records the paper/code objective mismatch; a pinned repository is a
custody anchor, not a claim that all paper artifacts are reproducible.

## Links

[[research-question-recommendation-2026-07-17]] ·
[[skill-lifecycle-research-snapshot-2026-07-17]] ·
[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] ·
[[opd-distillation]] · [[literature/index]]
