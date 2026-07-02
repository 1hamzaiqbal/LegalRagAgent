---
title: Can QPP Choose the Right Query Variant? (SIGIR 2026)
type: source
tags: [qpp, query-variant-selection, rag, retrieval-answer-gap, query-reformulation]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2604.22661
local: references/emami-qpp-variant.pdf
authors: Arabzadeh et al. (Negar Arabzadeh, Andrew Drozdov, Michael Bendersky, Matei Zaharia)
year: 2026
venue: SIGIR 2026 (arXiv 2604.22661, 24 Apr 2026)
code: https://github.com/Narabzad/QPP-4-RAG
---

> **Bibkey note**: our internal notes call this "Emami et al." — that is wrong. The paper at arXiv 2604.22661 is Arabzadeh et al. (UC Berkeley / Databricks). Keeping bibkey `emami-qpp-variant` for continuity; cite as Arabzadeh et al. 2026.

## TL;DR

Studies whether label-free QPP can pick the best of 31 query variations (original + 30 GPT-4o reformulations from six QueryGym methods, incl. Query2Doc) per information need, *before* paying retrieval+generation cost, on TREC-RAG 2024 (56 queries, MS MARCO v2.1, BM25 + Cohere dense, nugget-judged RAG). Headline: QPP selection reliably beats the original query and any single fixed reformulator on end-to-end answer utility, BUT retrieval-optimal and answer-optimal variants are *structurally misaligned* — even the retrieval-metric oracle underperforms cheap practical QPP on answer quality. They explicitly punt "generation-aware predictors" to future work.

## Key claims / numbers

- **Setup**: intra-topic variant selection (argmax predicted score among 31 variants), 13 pre-retrieval QPP methods (IDF/ICTF/SCQ/SCS/QL/DM/QSD) and 13 post-retrieval (Clarity, WIG, NQC, SMV, RSD, σ-based, BERT-QPP, QSD-post); metrics nDCG@5, Recall@100 vs nugget-based Nugget-All/Nugget-Strict with human judgments. *our-relevance:* this is the same NQC/WIG/SMV predictor family we used in our QPP routing experiment, applied to selection instead of routing — the near-twin for [[regime-routing]].
- **RQ1**: pre-retrieval QPP selection lifts BM25 Nugget-All 0.273→0.398 (IDF_max, +45.8%) and Nugget-Strict 0.227→0.377 (+66.1%) over the original query, while often *failing to improve ranking metrics*, especially dense. *our-relevance:* answers can improve without ranking metrics moving — the inverse face of our [[answer-conversion-gap]] (SCOPE's 8x retrieval lift moving answers 72.3→72.9).
- **RQ3 (utility gap, the money result)**: the retrieval-metric oracle is answer-suboptimal — BM25 Oracle-nDCG@5 hits nDCG 0.644 but only 0.344 Nugget-Strict, *below* practical IDF_max (0.377) and NQC (0.355); Oracle-Nugget-Strict reaches 0.536/0.569 (BM25/dense). Perfect retrieval prediction would not give optimal RAG. *our-relevance:* independent, human-judged confirmation that optimizing retrieval exposure is the wrong objective — directly grounds C9's "answers driven by parametric knowledge, not retrieval" and our Hit@5-up/accuracy-flat pattern.
- **RQ4 (correlation disconnect, verified)**: dense-retriever NQC correlates r=0.3286 with nDCG@5 and r=0.4206 with Recall@100 but −0.0038 (Nugget-All) / −0.0145 (Nugget-Strict) with answer quality; conversely SCQ_max on BM25 gets r=0.2436 with nuggets but r=0.0447 with nDCG@5. (The "NQC r=0.33 with nDCG@5, ~0 with answer quality" claim from our notes checks out — Table 2, dense column.) *our-relevance:* this is our "retrieval-generation gap" measured with a real QPP suite; also a methodology lesson for C11 — they argue selection-accuracy evaluation over global correlation.
- **RQ2**: post-retrieval QPP wins on ranking metrics (dense RSD nDCG@5 0.557→0.601, +7.9%) but Nugget-All gains only +2.7%; lightweight pre-retrieval predictors match or beat post-retrieval on end-to-end utility at lower latency. *our-relevance:* supports our conservative framing that cheap pre-retrieval signals suffice for routing; expensive geometry-after-retrieval must justify itself on the answer metric.
- **Variant selection > single fixed reformulator**: best single reformulator (MuGI, BM25) Nugget-All 0.371 vs QPP-selected 0.398; consistent but smaller gain in dense (~0.4013→0.4152). *our-relevance:* per-query selection among a [[generated-query-family]] beats committing to one method — the published version of what our regime-routing does at dataset level; also mildly pre-empts "just always run SCOPE".
- **Explicit future-work punt**: "develop generation-aware predictors that treat retrieval and generation as a coupled system, directly estimating answer-grounding potential rather than ranking quality alone"; oracle gap framed as attainable within the existing variant pool. *our-relevance:* the exact open slot our CE-affinity-movement mechanism (Spearman ~0.44, [[geometry-vs-factuality]]) is positioned to fill — they name the problem, we have a candidate signal.
- **Scale caveat**: 56 information needs (TREC-RAG 2024), GPT-4o as sole reformulator, general web domain, no significance tests visible in the main tables. *our-relevance:* their N is far below our per-dataset N (1195/6853); C11-style rigor complaints would apply to them too — CIs are not standard in this subfield either.

## Bearing on the review

- **C9 (weakest-baseline framing / parametric-knowledge answers)**: this paper makes the retrieval-vs-answer misalignment a published, human-judged, oracle-proven fact. A revised SCOPE paper must cite it and reframe: stop selling retrieval Hit@5 lift as the headline and instead present the answer-conversion gap as a known structural problem ("utility gap") that we characterize mechanistically in a domain-specific weak-query regime.
- **C7/C5 (marginal gains, no significance tests)**: their evaluation philosophy — selection accuracy against oracle upper bounds, improvement-over-original — gives us a rigor template: report oracle-Nugget-style upper bounds and gap-to-oracle rather than only pairwise deltas. Adopting their decision-oriented evaluation would directly answer the "marginal gains" charge with a ceiling analysis.
- **C11 (rigor)**: they publish all code/variants/configs (QPP-4-RAG repo). A revision should match that reproducibility bar and can also note their evaluation lacks CIs at N=56, i.e., our N is a strength.
- **Related-work obligation**: any revised paper's QPP/routing section must cite Arabzadeh et al. 2026 as the closest selection-among-LLM-variants work, alongside Scells et al. QVPP (CLEF 2017 TAR) which they identify as the Boolean-retrieval ancestor of variant-level prediction.

## Differentiation

Honest position: we are **pre-empted on the framing** that (a) QPP can select among LLM query variants label-free, (b) retrieval-optimal ≠ answer-optimal, and (c) correlation-based QPP evaluation is the wrong yardstick for RAG. We cannot claim any of those as novel. What they do **not** do: (1) no per-query *mechanism* for why an expansion helps — no geometric/CE-affinity analysis, no factuality falsification; their predictors are classic term/score statistics, not answer-grounding signals, and they explicitly flag generation-aware prediction as open; (2) no query-difficulty *regime* analysis — TREC-RAG queries are general web questions; nothing like our weak-query (BarExamQA Hit@5 1.4%) vs strong-query crossover, vocabulary gap, or query drift on strong queries; (3) no domain-specialized corpora (legal/medical) where expansion vs collapse behavior diverges (our CSQE-collapse finding); (4) selection among 30 pre-generated variants is costlier upstream than our route-then-generate (they still pay 30 generation calls per query before QPP). Our defensible remaining lane: mechanistic, regime-conditioned explanation of *when a single generative expansion converts to answers*, in expert domains — which slots into the "generation-aware predictor" gap they leave open.

## Links

[[scope]] · [[hyde]] · [[query2doc]] (one of their six variant generators) · [[qpp]] · [[answer-conversion-gap]] · [[regime-routing]] · [[generated-query-family]] · [[query-drift]] · [[weak-vs-strong-query-regime]] · [[geometry-vs-factuality]] · [[vocabulary-gap]] · [[icml-ai4law-2026-rejection]] · siblings: [[gure]], [[koblex-parser]], [[scope-paper-2026]]

## Raw source

- `references/emami-qpp-variant.pdf` (arXiv 2604.22661v1, 11 pp., read in full)
