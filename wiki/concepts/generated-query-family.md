---
title: Generated-Query Family (HyDE-family expansion)
type: concept
tags: [expansion, hyde-family, taxonomy]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# The generated-query family — a taxonomy

LLM-generated text replaces or augments the retrieval query. Members differ on
four axes that matter for novelty claims (C3/C6/C7):

| Method | Conditions on | Keeps raw query? | Uses corpus text? | Answer-conditioned? |
|---|---|---|---|---|
| [[hyde]] (Gao '23) | question | yes (query vec averaged in) | no | implicitly (doc answers q) |
| [[query2doc]] (Wang '23) | question | yes (concat) | no | implicitly |
| GAR (Mao '20, [[lamer-gar]]) | question | yes (expansion) | no | yes (generated answer/title) |
| LameR (Shen '23, [[lamer-gar]]) | question + candidate answers | yes | pseudo (answers from prompt) | yes (explicit) |
| [[csqe]] (Lei '24) | question + retrieved sentences | yes | **yes** (real sentences) | no |
| ParSeR ([[koblex-parser]], '25) | question (multi-hop, statute-style) | (verify) | no (parametric provisions) | (verify) |
| GuRE ([[gure]], '25) | question (trained rewriter) | replaced | trained on corpus pairs | no |
| [[scope]] (ours) | question + **private snap answer** | **no (discarded)** | no (optional 1-shot exemplar) | yes, but discarded pre-answer |

Lessons the taxonomy makes obvious:
- Answer-conditioning is *not* novel (GAR/LameR); privacy+discard of the draft
  is the only structural novelty left to SCOPE — and Query2doc's ablation says
  discarding the raw query usually *hurts*, so the design needs its own
  ablation (C12) rather than assertion.
- The family splits on *where grounding comes from*: parametric (HyDE, SCOPE,
  ParSeR) vs corpus-steered (CSQE, GuRE-trained). Our crossover result:
  parametric wins on weak-query regimes (nothing real to steer with), corpus-
  steered wins on strong. That regime split ([[weak-vs-strong-query-regime]])
  is a family-level finding, bigger than any single method.
- Trained (GuRE) vs zero-shot (rest): the missing baseline in our paper (C8).

## Links
[[scope]] · [[hyde]] · [[query2doc]] · [[lamer-gar]] · [[csqe]] ·
[[koblex-parser]] · [[gure]] · [[weak-vs-strong-query-regime]] ·
[[query-drift]]
