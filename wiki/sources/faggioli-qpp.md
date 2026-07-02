---
title: QPP for Neural IR — Are We There Yet? (ECIR 2023) + QPP Ad-hoc-to-Conversational (SIGIR 2023)
type: source
tags: [qpp, dense-retrieval, evaluation-protocol, query-rewriting, routing, neural-ir]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2302.09947
local: references/faggioli-qpp.pdf, references/faggioli-qpp-survey.pdf
authors: Faggioli et al. (primary); Meng et al. (companion)
year: 2023
venue: ECIR 2023 (primary); SIGIR 2023 (companion)
code: https://github.com/guglielmof/ECIR2023-QPP (primary); https://github.com/ChuanMeng/QPP4CS (companion)
---

**Provenance note**: the assignment labels both PDFs "Faggioli QPP." The primary PDF (2302.09947) is Faggioli, Formal, Marchesin, Clinchant, Ferro, Piwowarski, "Query Performance Prediction for Neural IR: Are We There Yet?" (ECIR 2023). The companion PDF saved as `faggioli-qpp-survey.pdf` (2305.10923) is actually **Meng, Arabzadeh, Aliannejadi, de Rijke, "Query Performance Prediction: From Ad-hoc to Conversational Search" (SIGIR 2023)** — a reproducibility study, not a Faggioli survey. Both were read in full.

## TL;DR

Faggioli et al. test 19 classic and supervised QPP predictors (incl. NQC/WIG/SMV/Clarity/BERT-QPP) on 7 lexical and 7 BERT-based first-stage retrievers over Robust'04 and Deep Learning'19, and find QPP largely breaks on neural retrieval: pre-retrieval predictors collapse, post-retrieval predictors lose most of their signal exactly where semantic matching matters, and — critically — QPP fails worst on the queries where lexical and neural systems differ most, i.e., where prediction would actually be used for routing/selection. Meng et al. extend the protocol to rewritten queries and a dense retriever (ConvDR) and show the failure is not universal: score-based unsupervised QPP (NQC/WIG) can work well on a dense retriever when its score distribution has high variance, and supervised QPP only wins with large training data. Together they define what a credible QPP claim must look like and why per-query routing signals are expected to be weak.

## Key claims / numbers

- Faggioli: pre-retrieval predictors are near-useless on neural retrievers — Robust'04 mean Pearson r 6.2% on NIR vs 25.6% on TIR; DL'19 NIR 5.4%. *our-relevance:* grounds why our perplexity/vocabulary-gap pre-screen failed per-query (memory: corr ~0) — this is the literature norm, not our bug.
- Faggioli: post-retrieval predictors hold up on Robust'04 (32.3% NIR vs 34.5% TIR) but drop hard on DL'19 passage retrieval (NIR overall 13.1%, post-retrieval 19.9%, vs TIR ~38.1%); NQC, the best TIR predictor, is ~10% worse on NIR. *our-relevance:* positions our WIG/NQC/SMV-on-CE-scores router-negative result (best WIG-CE τ≈−0.11) as consistent with known dense-score degradation.
- Faggioli: supervised BERT-QPP does not rescue neural QPP — DL'19 NIR mean correlation 4.5% vs 23.8% for other post-retrieval predictors. *our-relevance:* "just train a predictor" is not a free fix for our router.
- Faggioli: on the 62 "semantically defined" Robust'04 topics (top-quartile |TIR−NIR| nDCG gap), the run-type effect on QPP error grows 6× (ω² 0.11%→0.67%) — QPP fails most exactly where model selection needs it. *our-relevance:* directly analogous to our routing problem: the queries where SCOPE vs raw diverge are the ones where per-query QPP is least reliable; supports our regime-level (dataset-level) routing framing over per-query routing.
- Faggioli ANOVA: topic effect (ω² 22.5–24%) and topic×predictor interaction (17–23%) dwarf predictor choice (ω² ~2%). *our-relevance:* per-query variance dominance mirrors our finding that dataset-level regime separation works while per-query prediction fails.
- Faggioli protocol: Pearson r for comparability, plus sARE (scaled Absolute Rank Error, sARE(q)=|R^e_q − R^p_q|/|Q|) / sMARE for per-query distributional analysis, plus ANOVA with ω² strength-of-association. *our-relevance:* the evaluation kit a revised paper must adopt for any routing/confidence claim (C11).
- Meng: score-based unsupervised QPP can be *strong* on a dense retriever — NQC Pearson ρ 0.431 on ConvDR CAsT-19, often better than on BM25 — because ConvDR's retrieval-score distribution has higher variance; effectiveness "relies on the retrieval score distribution of a specific retriever, regardless of whether they assess a lexical-based or a neural-based retriever." *our-relevance:* our negative QPP-routing result cannot be blamed on "dense scores break NQC" without checking our CE/embedding score-distribution variance — a concrete diagnostic to add.
- Meng: supervised QPP wins only with large training data (NQA-QPP ρ 0.781 on OR-QuAC) and collapses few-shot (ρ 0.001 on CAsT-20 without warm-up). *our-relevance:* at our N (hundreds of queries per legal benchmark) trained routers are unlikely to beat unsupervised signals.
- Meng: QPP over *rewritten* queries is a studied setting — feeding T5/QuReTeC rewrites into QPP works, and QPP effectiveness degrades with worse rewrites (human > T5). *our-relevance:* closest published analogue to predicting quality of SCOPE-generated queries; rewrite quality sensitivity echoes our [[query-drift]] finding.
- Meng: all QPP methods correlate better with deeper metrics (nDCG@100/Recall@100) than nDCG@3. *our-relevance:* our Hit@5-level routing target is the harder end of QPP; expected weak correlations should be stated up front.

## Bearing on the review

- **C11 (rigor: no CIs/significance)**: both papers model the required protocol — correlation coefficients (Pearson r, Kendall τ, Spearman ρ) with significance tests (Meng: t-test p<0.05, non-significant values typeset distinctly), sARE/sMARE, ANOVA/ω². Any revised routing/confidence claim must report at least correlation + significance, ideally sMARE.
- **C2 (insufficient literature)**: our "confidence-gated" and router work must be positioned as QPP (post-retrieval, unsupervised, NQC/WIG/SMV lineage) citing Faggioli et al. and Meng et al.; not doing so is exactly the gap the reviewers flagged.
- **C9/C7 adjacent (framing weak results honestly)**: our per-query router-negative result becomes a *confirmatory* finding — "consistent with Faggioli et al., per-query QPP on dense pipelines fails where methods diverge; we therefore route at regime level" — rather than an unexplained failure. This is the honest repositioning of the routing negative.
- **C5 (marginal gains)**: indirectly — Faggioli's ANOVA shows topic variance dominates method variance in QPP; the same variance-decomposition style applied to our answer-accuracy matrix would make our small-delta claims more defensible.

## Differentiation

We do not propose a QPP predictor; we *consume* QPP-style signals to decide whether to apply generative query expansion (SCOPE) per query or per dataset. Neither paper studies QPP for gating query expansion, legal corpora, LLM-generated pseudo-document queries, or downstream answer accuracy — their target is retrieval quality of a fixed system (Faggioli) or of a rewrite+retrieve pipeline (Meng). Meng et al. is the closest prior art to "QPP over generated queries" and must be cited as such; we are pre-empted on the general idea that QPP can be computed on a rewritten query, but not on routing between raw and expanded queries by regime. Conversely, we cannot claim novelty for observing that NQC/WIG degrade on neural scores — that is Faggioli's headline — and our stronger asset is the gold-affinity geometric mechanism ([[scope-mechanism]], Spearman ~0.44), which is a *with-gold* diagnostic, a different object from no-gold QPP. Honest caveat: Meng et al. shows score-based QPP can succeed on dense retrievers, so our router-negative result is a property of our score distributions and task, not a law; a revised paper should report score-distribution variance before generalizing.

## Links

- [[qpp]] — this pair is the canonical grounding for the concept page
- [[scope]], [[hyde]], [[generated-query-family]] — Meng's rewrite-fed QPP is the nearest published analogue
- [[query-drift]] — rewrite-quality sensitivity of QPP (Meng §4.2.2) parallels drift
- [[regime-routing]] — Faggioli's "fails where systems differ most" is the strongest argument for dataset-level over per-query routing
- [[weak-vs-strong-query-regime]], [[vocabulary-gap]] — pre-retrieval predictor collapse constrains what vocabulary-gap metrics can do per-query
- [[geometry-vs-factuality]] — our with-gold geometry signal vs their no-gold predictors
- [[icml-ai4law-2026-rejection]] — grounds C2/C11 responses
- Sibling sources: [[emami-qpp-variant]], [[tian-right-track]], [[weller-drift]], [[gure]], [[koblex-parser]]

## Raw source

- `references/faggioli-qpp.pdf` — Faggioli et al., ECIR 2023, arXiv:2302.09947 (18 pp., read in full)
- `references/faggioli-qpp-survey.pdf` — Meng et al., SIGIR 2023, arXiv:2305.10923 (11 pp., read in full; note authorship correction above)
