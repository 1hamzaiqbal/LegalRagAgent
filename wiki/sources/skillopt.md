---
title: SkillOpt — Executive Strategy for Self-Evolving Agent Skills
type: source
tags: [skills, optimization, agents, context-adaptation, verifier]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.23904
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.23904.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/SkillOpt
authors: Yang et al.
year: 2026
---

# SkillOpt

## TL;DR

SkillOpt treats one natural-language skill document as the trainable external
state of a frozen agent. A separate optimizer model converts scored rollouts
into bounded add/delete/replace edits; a candidate replaces the current skill
only after a strict held-out improvement, with ties rejected. A rejected-edit
buffer and a protected epoch-level slow/meta update provide short- and
long-horizon optimizer state.

This is the strongest direct precedent for “gradient descent over a
`SKILL.md`,” but it does **not** train model weights. The paper itself names
self-distilling an optimized skill into model weights as future work, so
[[skill-lifecycle-research-snapshot-2026-07-17|SkillOpt → skill
internalization]] is an anticipated extension, not a sufficient novelty claim.

## Method

The target model and execution harness are frozen. SkillOpt repeatedly:

1. executes a rollout batch with the current skill;
2. has an optimizer model separately reflect on successes and failures;
3. aggregates and ranks atomic edit proposals under a textual learning-rate
   cap;
4. applies the candidate edits without allowing step-level edits to overwrite
   the protected slow-update region;
5. accepts the candidate only if its selection-split score is strictly higher;
6. exports only the best accepted `best_skill.md`.

The final test split is locked until reporting. The optimizer-only meta skill
is not shipped with the deployed skill.

## Main evidence

- Six benchmarks, seven target models, and direct-chat, Codex, and Claude Code
  harnesses produce 52 reported cells. The paper reports SkillOpt as best or
  tied-best in every cell against its measured prompt/skill baselines.
- In the six GPT-5.5-target/GPT-5.5-optimizer case runs of Table 6, final skills
  contain 379–1,995 tokens, with median 921, and require 1–4 accepted bounded
  updates. This is a six-run result, not a property established over all 52
  cells.
- A Codex-optimized SpreadsheetBench skill transfers to Claude Code from 22.1
  without a skill to 81.8, a +59.7-point gain. This is one model/benchmark/
  direction; the three other cross-harness gains are +1.6, +12.8, and +43.6.
- Cross-model transfer is positive in the four reported GPT-5.4 to
  GPT-5.4-mini/nano rows, but usually below a skill optimized directly for the
  target.
- Training cost is substantial even though deployment uses no optimizer call:
  Table 6 reports 20.8M–213.8M training tokens per case and 0.6M–46.4M tokens
  per absolute test-point gain. The deployed skill still has context-token,
  latency, and attention cost.

## Audit of the social-post claims

| Claim | Primary-source verdict |
|---|---|
| Held-out gate; strict improvements; ties rejected | Supported by the method, protocol, and current code. |
| “The validation gate is the only thing that matters” | Not established. There is no gate-off ablation, and the rejected buffer and slow/meta update have separate effects. |
| Final skills need 1–4 edits | Supported only for six Table 6 case runs. Each accepted update can itself contain several atomic edits. |
| Four to eight edits per step is the universal sweet spot | Overstated. The sweep covers caps 1, 2, 4, 8, and 16; all are competitive and the winner varies by benchmark. |
| Removing the budget collapses performance | False characterization. The no-learning-rate row drops 2.5, 1.8, and 4.0 points on the three ablation tasks. |
| Median final skill is about 920 tokens | Supported for the six Table 6 runs. |
| Codex-to-Claude transfer is +59.7 | Supported for one GPT-5.5 SpreadsheetBench cell, without uncertainty or repeated-seed estimates. |
| The protected section is worth about 22 points | Misattributed. The 22.5-point SpreadsheetBench drop removes both the meta skill and slow update; protection is not isolated. |
| GPT-5.4-nano plus a skill reaches frontier behavior | A derived six-task average is close to unadapted GPT-5.5, but per-task differences are large; the paper does not establish a scale law. |
| It is cheaper than directly training the small model | Untested. There is no direct SFT/RL/LoRA cost-matched baseline. |
| Description/body routing mismatch is a SkillOpt result | No. SkillOpt deploys one selected skill and does not study activation descriptions or multi-skill routing. |

## Limitations that matter here

- Scores are point estimates without confidence intervals or independent
  training/model-generation seeds.
- The automatic verifier and stable held-out selection split are core
  dependencies; open-ended writing, design, and strategy are not validated.
- There is no weight-training baseline and no post-withdrawal evaluation.
- The optimizer searches for **target-model contextual performance**, not for
  how well another model can learn the artifact.
- A skill accepted for a strong teacher is not thereby validated for a smaller
  student. This reader-conditioned mismatch is the useful opening.

## Repository observations

The official repository is modular: `skillopt/engine/trainer.py` orchestrates
rollout/reflection/edit/gate state; `skillopt/evaluation/gate.py` implements the
strict gate; optimizer modules implement clipping, schedules, protected
regions, and slow/meta state. The current checkout is newer than the paper.
Paper-faithful runs must explicitly set
`slow_update_gate_with_selection: true`; current `main` defaults it to false.
The newer `skillopt_sleep/` loop is post-paper preview code, not evidence for
the paper's claims.

## Bearing

SkillOpt is a mandatory external-context baseline and a useful source of
versioned, inspectable skill candidates. The open question is not whether it
can be concatenated with [[skill0]], [[skillc]], [[skill-sd]], or OPD. It is
whether the skill ranking induced by source-model contextual utility matches
the ranking induced by target-model use, acquisition, withdrawal, composition,
and retention.

## Code custody

- PDF SHA-256:
  `87f7f0f323b1671e9202b3ebb1596e909e507c71ecd1b360b0075a5ee1727fe3`.
- Official repository: https://github.com/microsoft/SkillOpt.
- EIT checkout pinned at
  `8a50db33124009772eb68d2e27a115bec819935e` on 2026-07-17.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skill0]] · [[skill-sd]] ·
[[skillc]] · [[skill-zero-five]] · [[opd-distillation]]
