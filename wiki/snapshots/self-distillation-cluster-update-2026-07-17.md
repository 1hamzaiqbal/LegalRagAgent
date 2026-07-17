---
title: Self-Distillation Cluster Update - OPSD, SDFT, and SDPO
type: decision
tags: [self-distillation, opd, continual-learning, rich-feedback, action-value]
created: 2026-07-17
updated: 2026-07-17
status: literature-integrated; no experiment launched
---

# Self-distillation cluster update - 2026-07-17

## Bottom line

The three papers change the OPD lane's baseline set but do not justify starting
a new training run:

1. [[opsd-self-distilled-reasoner]], [[sdft-continual-learning]], and
   [[sdpo-rich-feedback]] jointly occupy generic privileged-context-to-weights
   self-distillation for verified solutions, demonstrations, environment
   feedback, and interaction history.
2. They do not estimate repeated task outcomes under forced internal versus
   external actions, price those actions, cross the same payload over readers,
   or measure a target student's regret from following another model's action
   policy.
3. Their scale results show that self-teaching can fail for weak readers. A
   privileged context or nonzero KL is not evidence that the resulting teacher
   signal is useful.
4. The primary [[three-dial]] question remains student-specific,
   cost-sensitive action value. The narrower secondary question is whether
   several exact procedural artifacts preserve their ordering between one
   target's runtime context and that same target's matched post-withdrawal
   acquisition.

No experiment was launched in this literature pass.

## What the three methods establish

| Paper | Privileged view | Learning signal | Operational consequence |
|---|---|---|---|
| [[opsd-self-distilled-reasoner]] | Verified solution or reasoning trace | Full-vocabulary clipped forward KL on student rollouts | Mandatory unconditional privileged-context OPD baseline. |
| [[sdft-continual-learning]] | Instance-specific demonstration, article, answer, or tool call | Empirically forward KL on student prefixes with an EMA teacher | Mandatory same-model skill/fact internalization and forgetting baseline. |
| [[sdpo-rich-feedback]] | Runtime feedback and/or a successful peer rollout | Feedback-conditioned token/logit pseudo-advantages | Mandatory rich-feedback self-teaching baseline when a verifier exists. |

All three obtain a student trajectory from an unprivileged policy, expose a
second contextual view of the same model to privileged information, re-score
the student's prefixes, and update the unprivileged policy. That template is
prior art. Changing context source, divergence, or teacher update is an
engineering choice unless it reveals a new measured phenomenon.

## Keep three quantities separate

1. **Acting utility:** does an external intervention improve this reader's
   task outcome now, net of its incremental cost?
2. **Privileged-view teacher quality:** does adding a solution,
   demonstration, skill, or feedback produce a more correct teacher on the
   relevant task and prefixes?
3. **Acquisition utility:** does training from that view create durable
   no-context gain over matched direct training without harmful forgetting or
   boundary drift?

The first is the primary causal object in [[three-dial]]. The latter two are
training-stage gates for [[opd-distillation]]. A probability shift, KL, or
SDPO log-ratio is not a substitute for repeated forced-action task outcomes.

## Capability boundary

- OPSD warns that privileged solutions cease to teach beyond the model's
  comprehension frontier.
- SDFT's Science gap versus SFT is -3.3 at 3B, +4.0 at 7B, and +6.9 at 14B.
- SDPO loses at Qwen2.5-1.5B, approximately ties at 3B, and wins at 7B; a
  mostly-GRPO mixture helps its weakest tested reader.

This is suggestive of a capability-relative threshold, not a scaling law.
The papers use different tasks, objectives, contexts, and protocols. A law
claim would need untouched model sizes and a held-out family.

## Method-custody warnings

SDFT's manuscript describes reverse KL, while its official repository says
that all headline results used student-prefix/on-policy sampling followed by
full-vocabulary forward-KL matching to a demonstration-conditioned EMA
teacher. The code supports the repository description. The paper's knowledge
ablation also contains an internal contradiction: Figure 7 shows answer-only
37%, article-only 75%, and both 89%, while the prose says answer-only
outperforms article-only. Preserve the plotted values and do not inherit the
prose interpretation.

OPSD's main teacher is frozen; SDFT and SDPO favor EMA or regularized moving
teachers in their regimes. The evidence does not support a universal teacher-
update or KL-direction rule.

## Consequences for this branch

Before any E3-style allocation:

1. measure a held-out target-specific teacher-context gap in task terms;
2. compare direct target SFT/RL with the applicable OPSD, SDFT, or SDPO arm;
3. preserve weak-reader failures and stratify by model capability;
4. report task reward, action cost, retention, and target regret rather than
   imitation loss alone; and
5. keep bare OPD as a diagnostic, consistent with [[sdar]], not as evidence
   that the student learned a useful policy.

The architecture-free forced-action panel still comes first. Training becomes
scientifically relevant only if teacher-side information predicts a stable
target-specific disagreement or improves target-data efficiency.

## Primary-source custody

| Source | PDF SHA-256 | Pinned official repository |
|---|---|---|
| [[opsd-self-distilled-reasoner]] | `e521ed6c2939dc612daec5fe6fb47bef554a291228c02cafae07102b9729bad8` | `OPSD` at `7448751f307a9cdbcc1246dd1565a1a605b443df` |
| [[sdft-continual-learning]] | `c27949d1b7888b128bf40f6b031d4e84a82b8ba40d6e03d671491f8e414a516b` | `Self-Distillation` at `d77573212fa0a3ae2eeb64b9b44db1c251f75e3e` |
| [[sdpo-rich-feedback]] | `2714e7734c43cf849c2e7c49fc95cf57a6533c0833465a01983b87eb7f72d190` | `SDPO` at `7c457fc1b1f636ae794eb0362ba37d4743b06fbc` |

The PDFs and repositories live in the persistent shared EIT literature vault.
A pinned checkout is a custody anchor, not a claim of complete reproduction.

## Links

[[opd-distillation]] · [[three-dial]] · [[research-state-2026-07-17]] ·
[[literature/index]] · [[sdar]]
