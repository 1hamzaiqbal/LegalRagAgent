---
title: Can a Language Model Learn Facts Continually in Its Weights?
type: source
tags: [context-distillation, continual-learning, weights, context, forgetting]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2607.11020
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2607.11020.pdf
authors: Charles O'Neill
year: 2026
---

# Can a Language Model Learn Facts Continually in Its Weights?

## TL;DR

O'Neill compares facts kept in context with facts written into Qwen3 weights
through bare-statement SFT, broad “study” data, and offline/online context
distillation. Broad prompt coverage creates more usable knowledge, but even the
best weight methods lag context on composition and lose access under later
writes. The paper explicitly says whether procedural skills behave similarly
is untested.

It is therefore a measurement template for
[[skill-lifecycle-research-snapshot-2026-07-17]], not evidence that the same
failure already holds for `SKILL.md` files.

## Setup and estimands

- Qwen3-4B is the main model; an 8B replication covers the primary
  recitation-to-use gap.
- The primary evaluation uses 247 invented facts and five question types:
  recall, paraphrase, application, composition, and counterfactual use.
- “Study” data contain 24 diverse paraphrases, questions, implications, and
  contrasts; bare training repeats the statement in two trivial framings.
- Context distillation uses the original model plus the fact as teacher and a
  student without the fact in its prompt.
- Measurements include strict accuracy, a lenient-minus-strict entailment gap,
  retention after later writes, retained statement-log-probability lift,
  general capability, and KL drift.

## Main evidence

- Diverse recitation, even without explicit derived conclusions, reduces the
  entailment gap from 27.4 to 5.4 points.
- At 96 steps, study data improve application by 21 points, composition by 18,
  and counterfactual use by 29 relative to bare statements.
- Online context distillation reaches 77–78% strict five-type accuracy at its
  tested operating point, versus 71–72% for study/offline distillation and 61%
  for bare training. The best weight method still scores 70% composition
  against 83% with the fact in context.
- After 20 sequential writes, bare-statement facts retain 1%, study facts 46%,
  online context-distilled facts 27–32%, and offline-distilled facts 14%. The
  moving-teacher distillation runs also develop 35–46% looping/capped output.
- A frozen original-model teacher yields +2 points of held-out capability, KL
  0.48, and 54% retention after 20 writes. A teacher based on accumulated
  merges yields -31 points, KL 1.70, and 21% retention, or 34% after excluding
  capped loops.
- Behavioral forgetting is mostly lost access rather than erasure under the
  paper's probe. Forgotten facts retain 57–67% of drift-corrected statement
  log-probability lift. Two written facts score 32% on joint-use questions
  versus 91% when both are in context, and supplying forgotten study facts in
  context restores 77–80% accuracy.

## Interpretation for skills

The valuable tension is **compact execution context versus broad learning
curriculum**. [[skillopt]] finds compact context artifacts. This paper finds
that broad prompt coverage, not more repetitions of a narrow statement, is
what produces usable weight knowledge. A compact skill may therefore be a
good runtime instruction but a poor training dataset unless it is expanded
into diverse tasks, phrasings, failures, and contrasts.

The teacher result is not a universal rule. [[skill-sd]] reports that a
dynamically synchronized teacher beats a frozen teacher during joint agent RL.
The difference in update regime, task, and objective makes frozen versus moving
teacher an experimental factor, not a decision to settle from either paper.

## Limits

One model family, invented facts, mostly LoRA writes, small access/causal
subsets, generated data, and model judging limit generalization. The
100-write extension is descriptive. The paper does not measure skill use,
retrieval/routing cost, context-token cost, direct task RL, or amortized reuse.

## Code and data custody

The PDF advertises https://github.com/basetenlabs/cortex and
https://huggingface.co/datasets/baseten/cortex. On 2026-07-17 the GitHub URL
returned 404 and the dataset returned 401, so neither was archived as a usable
artifact.

- PDF SHA-256:
  `3d7fde31817d293afd64066174293fe6a925ce2650331550d66c38af3d6b3702`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillopt]] · [[skill-sd]] ·
[[opd-distillation]]
