---
title: Diagnosing and Calibrating Tool-Call Boundary Drift in Multi-Teacher On-Policy Distillation
type: source
aliases: [Behavior Leverage Imbalance in Multi-Teacher On-Policy Distillation]
tags: [opd, tool-use, boundary-calibration, multi-teacher, soft-clamp]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2607.07050
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2607.07050.pdf
code: not yet released as of 2026-07-17
authors: Jiabin Shen, Guang Chen, Chengjun Mao
year: 2026
---

# Tool-Call Boundary Drift

## Version caution

Version 1 was titled *Behavior Leverage Imbalance in Multi-Teacher On-Policy
Distillation*. The current v2 PDF is titled *Diagnosing and Calibrating
Tool-Call Boundary Drift in Multi-Teacher On-Policy Distillation*. The arXiv
abstract page still exposes the older `13.7% → 9.0%` summary in some views;
the v2 PDF abstract and multi-seed table report `14.2±2.1% → 9.0±0.2%`.
Numerical claims here use the v2 PDF.

## TL;DR

Multi-teacher GKD can improve should-call recall while shifting the student
toward over-calling on fixed should-respond examples. Aggregate exposure and
sequence loss miss this boundary shift. Soft Clamp locally compresses extreme
token JSD and reduces over-calling, although a global reweighting baseline is
competitive and no method dominates across sizes or metrics. The paper audits
behavior against fixed target-type labels, not the target student's causal
tool benefit or deployment price.

## Method

Training routes examples to a tool-call teacher or a direct-response teacher.
GKD uses a top-32-support approximation to token Jensen–Shannon divergence.
Student rollouts supply 80% of trajectories; dataset trajectories supply 20%
and receive an SFT format anchor of weight `0.3`.

Soft Clamp sets a detached batch threshold
`C = k * mean(token_JSD)`, with `k=3`. Divergences below `C` are unchanged;
above it, the forward contribution is capped while the gradient remains
nonzero and is scaled by `C / divergence`. Comparators are hard clipping,
global batch-relative reweighting, and a validation-tuned inference-time logit
bias on tool entry.

Diagnostics include first-token tool-call probability, tool-entry margin,
signed teacher pressure, token-level JSD concentration, and full checkpoint
boundary trajectories.

## Evidence

- Tool and response teachers: Qwen3.5-9B specialists.
- Students: Qwen3.5-9B and a Qwen3.5-4B replication.
- Training: 15,419 tool-call and 15,420 response examples derived from
  APIGen-MT with conversation-disjoint splits.
- Primary comparisons use three seeds.
- Evaluation includes a balanced 4,000-example APIGen decision set, BFCL,
  When2Call, and an 800-task/3,136-turn BFCL multi-turn diagnostic.

Main 9B APIGen results:

| Method | Decision accuracy | Over-call | Call recall | Respond recall |
|---|---:|---:|---:|---:|
| Base SFT | 85.3 | 4.9 | 75.5 | 95.1 |
| Vanilla GKD | 88.6±0.3 | 14.2±2.1 | 91.5±1.7 | 85.8±2.1 |
| Global Reweight | 88.6±1.3 | 9.7±0.8 | 86.9±3.3 | 90.3±0.8 |
| Soft Clamp | 88.7±0.7 | 9.0±0.2 | 86.5±1.4 | 91.0±0.2 |

Soft Clamp trades about five call-recall points relative to vanilla GKD for
lower over-calling. Generalization is mixed:

- on BFCL overall, Base SFT scores `82.9`, Vanilla GKD `79.0±1.7`, and Soft
  Clamp `80.8±0.1`;
- on When2Call, Base is `72.8`, Base SFT `71.6`, and all GKD variants fall in
  `64.3–67.1`;
- Global Reweight is slightly better than Soft Clamp on several 4B multi-turn
  means;
- boundary AUC barely changes (`0.9692±0.0023 → 0.9710±0.0011`), while a
  validation-tuned scalar entry bias reproduces much of the call-frequency
  and loop reduction;
- pure OPD without the format anchor catastrophically over-calls (`83.2%`) and
  only `16.1%` of emitted calls are valid.

## Limitations

- “Should call” is a fixed target-type label, not a model-specific utility
  decision.
- There are no paired forced tool/no-tool outcomes and no action price.
- Primary decision accuracy detects a `<tool call>` marker, not valid schema,
  successful execution, or answer correctness.
- Multi-turn “non-tool final” measures termination, not correct task outcome.
- One family, two student sizes, a fixed teacher pair, and three seeds limit
  transport claims.
- Mechanism evidence is diagnostic rather than causal; teacher JSD is
  truncated to the returned top-32 support.
- The pure-OPD ablation changes both rollout mixture and SFT anchoring, so it is
  not an anchor-only test.
- Nearly unchanged AUC plus the strong scalar-bias baseline suggests much of
  the effect is operating-point movement rather than a new decision
  representation.

## Bearing on our work

This work occupies OPD-induced tool-boundary drift, over-calling under
multi-teacher OPD, and local divergence calibration. It is the required
training-dynamics baseline, but not a target-student utility baseline.

The useful hypothesis is that **rankings may transport while thresholds do
not**. Compare direct action imitation, teacher-value regression,
affine/isotonic student calibration, and student-only outcome learning. Rescore
each policy across an analytic price grid, report advantage-weighted regret,
and include the scalar entry-bias baseline before proposing a new loss.

At every checkpoint, distinguish teacher pressure, target-student causal
value, schema validity, execution success, and task reward. Increased teacher
agreement is not evidence of successful transfer if student utility falls.

## Code custody

No repository was available on 2026-07-17. Section 6 of the v2 PDF says the
training code, analysis scripts, and evaluation harness are planned for
release after internal review. Record this as **not yet released**, and check
again before submission.

## Raw source

EIT PDF `papers/arxiv_2607.07050.pdf`.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] ·
[[action-value-transport-reading-packet-2026-07-17]] · [[reward-gated-opd]] ·
[[turnopd]] · [[rethinking-opd]]
