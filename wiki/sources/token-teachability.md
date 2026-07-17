---
title: Not All Disagreement Is Learnable — Token Teachability in On-Policy Distillation
type: source
tags: [opd, teachability, student-support, token-selection, distillation]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.26844
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.26844.pdf
code: https://github.com/wyy-code/TA-OPD
code_commit: ccdf21d2066466f3d616f63cd867cc49119c45e6
authors: Anonymous authors
year: 2026
---

# Token Teachability

## TL;DR

Large teacher–student disagreement is not automatically useful supervision.
Token Teachability argues that disagreement is more learnable when the
teacher's preferred probability mass lies among alternatives the student
already ranks highly. TA-OPD selects high-scoring tokens and often matches or
beats full OPD with 3–10% of response tokens. This is a measure of local
**absorbability**, not whether the teacher's desired behavior improves the
student's task utility.

## Formal object

The diagnostic freezes a bank of student-generated prefixes and measures how
much same-context teacher KL falls after training:

`G_fix = KL(p_T || p_student_before) - KL(p_T || p_student_after)`.

For the union of student and teacher top-`K` token sets, it computes a
renormalized forward-KL disagreement `D_t` and teacher mass `C_t` on the
student's top-`K` set. After within-batch percentile normalization:

- learnable disagreement: `D_L = D_tilde * C_tilde`;
- incompatible disagreement: `D_I = D_tilde * (1 - C_tilde)`.

TA-OPD retains only the top budgeted fraction of response positions by
`D_L`, then applies an otherwise standard reverse-KL OPD objective.

## Evidence

Four teacher/student pairs are evaluated:

1. Qwen3-4B → Qwen3-1.7B;
2. Qwen3-8B-GRPO → Qwen3-4B;
3. Qwen3-14B → Qwen3-4B;
4. DeepSeek-R1-Distill-Qwen-14B → Qwen2.5-3B.

Training prompts come from DAPO; evaluation covers AIME24/25, GPQA-Diamond,
HumanEval, IFEval, and MATH-500. The fixed-context analysis covers 300
contexts and 57,600 token positions. Across `K={8,16,32}`, learnable
disagreement has approximately twice the standardized coefficient of
incompatible disagreement (`0.086–0.087` versus `0.043–0.045`).

At a 10% supervision-token budget, TA-OPD has the highest reported macro
average in all four blocks. The margins vary dramatically:

- 4B→1.7B: `44.89` versus `42.37` for full OPD;
- 8B→4B: `56.87` versus `56.81` for TIP;
- 14B→4B: `54.65` versus `54.64` for full OPD;
- cross-backbone: `30.62` versus `29.98` for the base model and `28.76` for
  full OPD.

The 0.01 and 0.06 “wins” are too small to treat as established without
independent training variance or statistical tests. Token-budget sweeps are
non-monotonic; 3–5% is often best. A supervision-token fraction is not the
same as proportional wall-clock, memory, or teacher-serving savings.

## Limitations

- The student's top-`K` set is a proxy for support, not literal zero/nonzero
  support or a proven learnability condition.
- The diagnostic target is teacher agreement, not task reward. Harmful teacher
  behavior may be highly teachable.
- Diagnosis uses forward KL while training uses reverse KL.
- Within-batch normalization makes scores difficult to compare across batches,
  students, or scales.
- Evaluation seeds sample one trained checkpoint; training variance is not
  measured.
- The study is Qwen-heavy and math-heavy, with one cross-backbone pair and no
  tools, retrieval actions, or external-action prices.
- Appendix Table 12 appears mislabeled: its heading names the 8B-GRPO→4B pair
  while its `54.65` TA-OPD average matches the 14B→4B main block.

## Bearing on our work

This paper closes “not every teacher disagreement is learnable” as a novelty
claim. It gives us a separate axis that must not be conflated with
[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]]:

1. Is the external action valuable to the target student?
2. Can the student exploit the action's payload?
3. Is the teacher's token-level signal locally imitable?

A signal can be easy to imitate and wrong for the student. Conversely, an
external action can have high task value even if the teacher's exact trace is
distributionally incompatible. A strong analysis is therefore a 2×2 of
signed target-student action value and high/low teachability. Compare raw
teacher advantage, teacher advantage plus compatibility features, and direct
student estimates on held-out families.

## Code custody

The official `wyy-code/TA-OPD` checkout is pinned in EIT at commit
`ccdf21d2066466f3d616f63cd867cc49119c45e6`. It includes a modified `slime`
training tree, launch scripts, a compact patch, and fixed-context diagnostics,
including:

- `tools/export_fixed_context_bank.py`;
- `tools/eval_fixed_context_bank.py`;
- `tools/analyze_fixed_context_gain.py`;
- `tools/matched_fixed_context_topn.py`;
- `tools/support_definition_robustness.py`.

It does not include result JSONL/CSV artifacts or checkpoints sufficient to
reconstruct the published tables. The reported campaign used 64 H800 GPUs;
source availability should not be mistaken for cheap reproduction.

## Raw source

EIT PDF `papers/arxiv_2605.26844.pdf`; EIT repository checkout recorded in
`wiki/literature/manifests/eit_repos.tsv`.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] · [[action-value-transport-reading-packet-2026-07-17]] ·
[[reward-gated-opd]] · [[craft-counterfactual-credit]] · [[rethinking-opd]]
