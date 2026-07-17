---
title: OPSD - Self-Distilled Reasoner
type: source
tags: [self-distillation, on-policy-distillation, privileged-context, reasoning, scale]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2601.18734
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2601.18734.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/OPSD
authors: Zhao et al.
year: 2026
---

# Self-Distilled Reasoner - OPSD

## TL;DR

On-Policy Self-Distillation (OPSD) uses one model in two contextual roles. The
student generates a rollout from the question alone; a stop-gradient teacher
with the same underlying checkpoint re-scores that rollout while conditioned
on a verified solution or reasoning trace. Full-vocabulary token-distribution
matching then trains the unprivileged student.

This is strong prior art for **privileged context to no-context weights through
same-model on-policy soft distillation**. It rules out presenting a
skill-conditioned self-teacher or verified-solution-conditioned OPD pipeline as
the contribution. It does not estimate the student's immediate causal benefit
from an external action, cross a canonical action payload over different
readers, price the action, or measure target regret from a transferred teacher
policy.

## Method

- Student: `p_S(. | question)`, sampled on-policy.
- Teacher: the same model conditioned on the question and privileged verified
  solution, evaluating the student's existing prefixes rather than generating
  a separate trajectory.
- Main teacher weights: frozen at the initial policy. The released code also
  contains optional EMA-teacher support, but that is not the main reported
  setup.
- Main loss: full-vocabulary forward KL with gradients through the student
  only. Reverse KL, generalized JSD, and a sampled-token policy-gradient form
  are ablated.
- Stabilizer: pointwise clipping of individual vocabulary-entry divergence
  contributions, motivated by style-token divergence dominating mathematical
  tokens.
- Data: up to 30,000 OpenThoughts mathematical problem-solution pairs.
- Readers: Qwen3-1.7B, Qwen3-4B, and Qwen3-8B Instruct.
- Evaluation: AIME 2024, AIME 2025, and HMMT 2025 with 12 stochastic samples
  per item.

The privileged teacher is a conditional distribution, not an independently
verified policy. The method assumes that the same model can rationalize the
solution well enough for that distribution to be useful.

## Main evidence

| Reader | Base average | SFT | GRPO | OPSD | OPSD - base |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 37.1 | 35.8 | 37.7 | 43.4 | +6.3 |
| Qwen3-4B | 61.2 | 58.6 | 62.7 | 63.6 | +2.4 |
| Qwen3-8B | 61.8 | 59.8 | 64.0 | 64.8 | +3.0 |

The averages are the paper's three-benchmark Avg@12 values. OPSD reports the
best checkpoint among evaluations every 20 steps through step 100; GRPO reports
its peak through 500 steps. This checkpoint-selection asymmetry and the lack of
independent training-seed uncertainty limit how strongly small differences
should be interpreted.

On Qwen3-1.7B/AIME25, forward KL reaches 43.9 at step 50 from a 36.7 base,
whereas reverse KL reaches 37.5 and JSD 36.9 at the same step. On a separate
Qwen3-4B pass@8 ablation with 2,048-token rollouts, full-vocabulary
distillation scores 84.1/60.0 on AIME25/HMMT25 versus 82.1/57.3 for the
sampled-token objective. These are different evaluation protocols from the
main table and should not be merged into one matrix.

The reported training recipe uses one student rollout capped at 1,024 tokens
for 100 steps, while the GRPO configuration uses eight rollouts capped at
16,000 tokens for up to 500 steps. This supports a token-efficiency result in
the tested setting; caps are not the same as realized generated-token counts.

## Exact boundary for our questions

OPSD occupies:

- same-model privileged-context self-teaching;
- on-policy soft context distillation for reasoning;
- verified reasoning traces as teacher-only information;
- full-logit versus sampled-token objective comparisons; and
- the general claim that deployment context can be compressed into weights.

OPSD does **not** contain:

- forced `do(internal)` versus `do(external)` outcomes;
- one identical canonical external payload crossed over multiple readers;
- action cost or a deployment price;
- cross-scale transport of signed action advantage;
- a teacher action oracle or held-out target-student regret; or
- several fixed procedural artifacts ranked both in target context and after
  matched training/withdrawal.

For [[research-question-recommendation-2026-07-17]], OPSD is therefore a
downstream training baseline, not a collision with the measurement-first
action-value object. For [[skill-lifecycle-research-snapshot-2026-07-17]], it
makes generic context-to-weights internalization non-novel but leaves the
same-fixed-artifact placement-ordering question open.

## Design lessons

1. **Measure the privileged-view gap first.** The paper's own limitation says
   that if a problem exceeds the model's comprehension frontier, the
   privileged teacher cannot provide meaningful supervision.
2. **Do not assume a divergence.** Forward KL dominates reverse KL/JSD in the
   reported reasoning ablation, while other OPD papers use different losses.
3. **Treat style as a failure mode.** High divergence on stylistic tokens can
   dominate task-relevant signal; token-category and output-behavior audits are
   required.
4. **Separate same-model self-teaching from cross-model teaching.** OPSD does
   not test a larger teacher supervising a smaller reader.
5. **Keep task reward as a safety control in agentic settings.** [[sdar]] finds
   that standalone or naively combined OPSD can collapse on agent tasks even
   though OPSD succeeds here on fixed mathematical reasoning.

## Version and code custody

- Audited paper: arXiv v3, 20 March 2026.
- PDF SHA-256:
  `e521ed6c2939dc612daec5fe6fb47bef554a291228c02cafae07102b9729bad8`.
- Official repository: https://github.com/siyan-zhao/OPSD.
- EIT checkout pinned at
  `7448751f307a9cdbcc1246dd1565a1a605b443df` on 2026-07-17.
- The repository records that chat-template and ZeRO-2 bugs were fixed and the
  paper/results rerun before v3. Older numeric summaries should not be mixed
  with the v3 table.
- No license file was present at the pinned repository commit; code custody is
  not the same as permission for unrestricted reuse.

## Links

[[self-distillation-cluster-update-2026-07-17]] ·
[[sdft-continual-learning]] · [[sdpo-rich-feedback]] · [[sdar]] ·
[[opcd]] · [[research-question-recommendation-2026-07-17]]
