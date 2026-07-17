---
title: LLM-Specific Utility for Retrieval-Augmented Generation
type: source
tags: [rag, evidence-utility, reader-specific, retrieval, causal-evaluation]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2510.11358
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2510.11358.pdf
code: https://anonymous.4open.science/r/LLM_specific_utility-4260/README.md
authors: Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei Yin, Xueqi Cheng
year: 2026
---

# LLM-Specific Utility for Retrieval-Augmented Generation

## TL;DR

This is the closest direct precursor to the three-dial reader-conversion
framing. The same passage can be useful to one LLM and not another, and each
of four evaluated models performs best with its own constructed “utilitarian”
passages. The label is, however, a single deterministic binary `incorrect →
correct` flip chosen and evaluated on the same outcomes. It does not estimate
signed harm, uncertainty, evidence-set marginal value, price, or
teacher-to-student policy regret.

## Formal object

For model `L`, question `q`, and candidate passage `d`, the paper labels

`u(L,q,d) = 1[Acc(L(q,d)) > Acc(L(q,empty))]`.

Because accuracy is binary and temperature is zero, positive utility is
exactly a deterministic incorrect-without-evidence to correct-with-evidence
flip. A model that already answers correctly cannot receive a positive label;
evidence for known questions is collapsed into neutral or harmful but those
two cases are not separated.

The benchmark evaluates set F1, NDCG@5, and downstream RAG accuracy for
utility-aware passage selection and ranking.

## Evidence

- Models: Llama3.1-8B-Instruct and Qwen3-8B/14B/32B, with Qwen thinking
  disabled.
- Core datasets: Natural Questions, TriviaQA, and MS MARCO-FQA.
- Candidate pools combine BGE-M3 top-20 retrieval with human-gold passages.
- Utility-judgment evaluation additionally uses HotpotQA, 2WikiQA, and FEVER.
- On NQ, each model's own selected evidence beats human gold: `80.30` versus
  `70.46` for Llama3.1-8B; `78.60` versus `67.32` for Qwen3-8B; `80.68`
  versus `71.55` for Qwen3-14B; and `82.16` versus `73.60` for Qwen3-32B.
- Cross-model transfer matrices show diagonal dominance: each model generally
  does best with its own utilitarian set. Same-family Qwen sets overlap more
  than Qwen and Llama sets.
- Roughly half of human-gold evidence overlaps model-specific gold on NQ and
  MS MARCO-FQA. Selected passages also have lower model perplexity than
  excluded human-gold passages.
- Listwise verbal selection generally beats pointwise selection; pseudo-
  answers usually help. All methods over-select when the true utilitarian set
  is empty.

## Central validity caveat

The strongest diagonal result is partly mechanical. Model-specific gold is
selected using the same model's deterministic outcome and then evaluated by
showing that the same model does best on passages chosen because they made it
correct. There is no repeated generation, independent outcome sample, or
cross-fitted label/evaluation split. The cross-model pattern is suggestive,
but its magnitude may include selection-on-noise and cannot be read as a
repeated causal effect.

Other limitations:

- one temperature-zero generation per model/question/passage condition;
- binary labels discard magnitude and signed harm;
- pointwise labels ignore redundancy and multi-document synergy;
- empty-gold questions are excluded from some selection metrics;
- answer-string accuracy can produce false positives;
- the three Qwen scales plus one Llama-8B do not identify a family-robust
  scaling law;
- perplexity is correlational and may proxy answer-token overlap;
- there is no retrieval price, latency, context cost, or stopping decision.

## Bearing on our work

The paper closes the claims that evidence utility is LLM-specific, that the
same passage is not universally transferable, and that human-gold evidence
need not be optimal for an LLM. It should be the primary deterministic
baseline for [[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]].

Our candidate earns a distinct contribution only by adding:

- repeated paired forced-action outcomes;
- an independently cross-fitted evaluation sample;
- signed advantage, including evidence harm;
- explicit action cost and price thresholds;
- evidence-set marginal value, redundancy, and synergy;
- a teacher×target action-value matrix and transport regret;
- prediction at held-out scale and held-out family.

The most direct replication is to rebuild the paper's transfer matrix under
repeated outcomes and separate label/evaluation repeats, then measure how much
diagonal dominance survives. Controlled paraphrases of identical factual
content can test whether readability/perplexity causally mediates
reader-specific utility rather than merely correlating with it.

## Code and artifact custody

The official artifact is an anonymous 4open.science snapshot, not a Git
repository. It exposes scripts such as:

- `src/pointwise_performance_answerem.py`;
- `src/without_passage_answer.py`;
- `evaluation/gold_utility_computing.py`;
- `evaluation/ranking-evaluation.py`;
- `evaluation/set-evaluation.py`.

As of 2026-07-17, the README says retrieval results, gold-utility labels, and
judgment outputs will be uploaded after anonymity. It therefore supports code
inspection but not full table reconstruction. The persistent vault records a
checksummed snapshot separately from pinned Git checkouts.

## Raw source

EIT PDF `papers/arxiv_2510.11358.pdf`; anonymous code snapshot recorded in the
literature manifests.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] · [[action-value-transport-reading-packet-2026-07-17]] ·
[[predicting-retrieval-utility]] · [[cue-r]] · [[effort-conditioned-resource-allocation]]
