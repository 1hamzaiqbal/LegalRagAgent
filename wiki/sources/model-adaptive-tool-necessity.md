---
title: Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use
type: source
tags: [tool-use, model-specific-necessity, metacognition, hidden-state-probes]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.14038
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.14038.pdf
code: https://github.com/chengez/Tool-Cognition-Action
code_commit: b200505f16ca00dc1a23af932a99aa6e1e45c6c7
dataset: https://huggingface.co/datasets/yizecheng/model-adaptive-tool-necessity
dataset_commit: cc92815d77444dee391bdd2709d2935c72ffdfb6
authors: Yize Cheng et al.
year: 2026
---

# Model-Adaptive Tool Necessity

## TL;DR

The paper establishes that “should this model call a tool?” cannot be assigned
globally: model-specific no-tool reliability and observed call behavior differ
substantially. It then uses hidden-state probes to decompose recognizing need
from executing a call. Its necessity label is one-sided—**any failure among ten
no-tool samples**—and never observes whether the tool helps. It therefore
measures imperfect no-tool reliability, not causal or cost-sensitive tool
value.

## Formal object

For model `f` and item `x`, the paper labels the tool unnecessary only when all
`N=10` no-tool samples are correct; otherwise it labels the tool necessary.
Sampling uses temperature `0.7`.

The conceptual pipeline is `x → cognition z_f(x) → action a_f(z_f(x))`.
Logistic probes decode the necessity label and the observed call action from
hidden states, with MCC used under class imbalance. The paper compares probe
directions by cosine similarity and measures a normalized probability of
entering tool mode.

## Evidence

Models:

- Qwen3-8B and Qwen3-4B;
- Llama-3.1-8B-Instruct and Llama-3.2-3B-Instruct.

Tasks and actions:

- 4,000 generated arithmetic questions from 13 templates with a calculator;
- 817 TruthfulQA questions with web search.

Observed necessity/action mismatch is large and changes direction across
models and domains:

| Model | Arithmetic | TruthfulQA |
|---|---:|---:|
| Qwen3-8B | 41.7% | 31.1% |
| Qwen3-4B | 26.5% | 41.8% |
| Llama-3.1-8B | 38.5% | 30.8% |
| Llama-3.2-3B | 54.0% | 32.8% |

Necessity is moderately linearly separable for arithmetic and less so for
TruthfulQA; action is broadly separable. Necessity and action probe directions
become nearly orthogonal at late layers. A Sankey-style decomposition assigns
most mismatch to the inferred cognition-to-action transition. Explicitly
asking whether a tool is needed changes calls on `18.2–49.3%` of arithmetic
items and `20.9–34.6%` of TruthfulQA items, yet the prompted answer has poor
necessity MCC.

## The central estimand problem

If the true no-tool success probability is `p`, the probability that the
paper labels the item necessary is `1 - p^10`. Thus:

- `p=.95` still yields a `40.1%` necessity-label probability;
- `p=.99` still yields a `9.6%` probability.

The label is better described as “failed a ten-draw perfect-reliability test.”
It can call a tool necessary when the tool is useless or harmful and can call
it unnecessary when a free tool would further improve reliability. It has no
action cost and never estimates
`P(success | do(tool)) - P(success | do(no tool))`.

## Other limitations

- Four models cover only 3B–8B and two families.
- Only arithmetic and TruthfulQA are evaluated.
- Results are sensitive in principle to `N` and temperature, but only one
  setting is studied.
- Linear decodability does not prove a metacognitive representation or causal
  role. Random item splits may let arithmetic probes exploit visible template
  families.
- The cognition-to-action attribution is observational; the probe direction is
  not intervened on.
- End-task tool-assisted improvement, harm, latency, and price are absent.

## Bearing on our work

This work occupies model-adaptive necessity, cross-model need variation, a
knowing/doing decomposition, and hidden-state probing. It does **not** occupy
causal tool advantage or teacher-to-student action-value transport.

The sharpest first experiment for
[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] is a
three-label audit on the same items:

1. `not 10/10 correct without tool` (this paper);
2. a fixed benchmark `should-call` label;
3. `P_S(success | do(tool)) - P_S(success | do(no tool)) > price`.

Then vary the number and temperature of no-tool samples and measure which
label predicts realized target-student utility. If a representation analysis
is included, hold out entire arithmetic templates and decode continuous
advantage rather than the binary any-failure label.

## Code and dataset custody

The official `Tool-Cognition-Action` repository is pinned in EIT at commit
`b200505f16ca00dc1a23af932a99aa6e1e45c6c7`; the official Hugging Face dataset
revision observed during intake is
`cc92815d77444dee391bdd2709d2935c72ffdfb6`.

The checkout provides raw necessity/action splits, generations, Qwen/Llama
tool handlers, prompt formatting, batched hidden-state extraction, linear
probes, and stored probe directions. It is approximately 218 MiB because it
includes roughly 10,000 probe-weight files. Important caveats:

- no license file or GitHub license metadata was detected, so inspect and cite
  it but do not copy code into this repository without permission;
- the original ten-run necessity-labeling pipeline and arithmetic generator do
  not appear to be included;
- probe evaluation uses a random 70/30 item split rather than template-family
  holdout.

The first data-heavy checkout attempt was preserved outside the vault at
`/engrfs/tmp/jacobsn/hiqbal_legalrag/vault-repair/Tool-Cognition-Action.incomplete-20260717`.
It is not counted or referenced as a source and is a safe candidate for a later
scratch-cleanup pass once no recovery need remains.

## Raw source

EIT PDF `papers/arxiv_2605.14038.pdf`; official code checkout is recorded in
`wiki/literature/manifests/eit_repos.tsv`.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] ·
[[action-value-transport-reading-packet-2026-07-17]] ·
[[effort-conditioned-resource-allocation]] · [[rational-metareasoning]]
