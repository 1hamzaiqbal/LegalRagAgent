---
title: Limitations of QPP for Selective Query Processing (arXiv 2025)
type: source
tags: [qpp, selective-query-processing, ir-evaluation, kendall-tau, dense-retrieval]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2504.01101
local: references/datta-qpp-reliability.pdf
authors: Chifu et al. (Adrian-Gabriel Chifu, Sebastien Dejean, Josiane Mothe, Moncef Garouani, Diego Ortiz, Md Zia Ullah)
year: 2025
venue: arXiv preprint (cs.IR), 2504.01101
code: https://anonymous.4open.science/r/UncoveringTheLimitationsofQPP-346C/README.md (release promised on acceptance)
---

**Authorship correction (important).** Our internal task list labeled this "Datta et al. 2025" and the bibkey is frozen as `datta-qpp-reliability`, but arXiv 2504.01101 ("Uncovering the Limitations of Query Performance Prediction: Failures, Insights, and Implications for Selective Query Processing", v1 Apr 1 2025) is by **Chifu, Dejean, Mothe, Garouani, Ortiz, and Ullah**. Suchana Datta appears only as their reference [6] (selective relevance feedback, ECIR 2024). Cite as Chifu et al. 2025 in any paper text.

## TL;DR

A comprehensive stress test of QPP (NQC, UQC, WIG, QF, summarized LETOR features, and BERT bi/cross-encoder predictors) across four TREC collections (ROBUST, GOV2, WT10G, MS MARCO DL) and sparse/dense rankers (BM25, DFree, +Bo2 QE, SPLADE, ColBERTv2). Predictor-performance correlations are modest at best (best-case Kendall tau ~0.32-0.37, on ROBUST only), collapse across collections (near zero or negative on MS MARCO for BM25), and QPP-driven selective query processing yields only marginal, mostly non-significant downstream gains even in-domain. Collection identity, not ranker, is the dominant factor in predictor accuracy (ANOVA).

## Key claims / numbers

- Best individual post-retrieval predictor: UQC, Pearson r=.407 / Kendall tau=.322 (BM25, ROBUST) and r=.439 / tau=.328 (SPLADE, ROBUST); NQC is weaker (tau .285 BM25-ROBUST). *our-relevance:* these are the field's healthy-case ceilings against which our no-gold QPP router (best |tau|~0.11 for WIG-CE) should be framed - our negative is within-regime for QPP, not an implementation failure (C11, and grounds the QPP-routing framing demanded by C2).
- Combined-feature ceiling: linear regression over the 4 SOTA predictors reaches tau=.347 (BM25, ROBUST) and tau=.374 (SPLADE, ROBUST) - i.e. the observed **~0.37 tau ceiling**; Pearson r hovers ~0.48-0.53. *our-relevance:* this is the strongest number to cite for "even mature QPP tops out well below decision-grade reliability" when motivating regime-level rather than per-query routing ([[regime-routing]]).
- **Verification note on the "tau >= 0.5 bar":** the paper does NOT state an explicit Kendall tau >= 0.5 reliability threshold anywhere we could find. The only 0.5-adjacent sentence ("The correlations however are above 0.5", Sec. 3.1 re Table 4) reads as Pearson-r commentary and only approximately matches the SPLADE/ROBUST column. If our paper needs a formal tau>=0.5 bar, it must be sourced elsewhere (or stated as our own operational threshold); cite Chifu et al. for the *observed ~0.37 ceiling* and for *selective processing gains being marginal even at that ceiling*. *our-relevance:* prevents us from mis-citing the yardstick behind our tau~0.11 router negative (C11 rigor).
- Generalization failure across collections: on MS MARCO with BM25, no predictor correlates with NDCG (UQC r=-.123, NQC r=-.010, WIG r=.027); ROBUST/GOV2 behave, WT10G/MS MARCO do not. ANOVA: collection factor F=137.6 (p<2e-16) vs ranker factor F=3.021 (p=0.0106). *our-relevance:* directly parallels our dataset-level finding that QPP separates regimes (BarExam vs Housing) but fails per-query - the field's own evidence says QPP signal is collection/regime-scoped ([[qpp]], [[weak-vs-strong-query-regime]]).
- BERT-based predictors (B_bi, B_cross) are the weakest (e.g. B_cross r=.069 / tau=.034 on ROBUST-BM25); dense-tailored features do not fix dense-retrieval QPP. *our-relevance:* supports our choice to build geometric CE-affinity mechanisms rather than off-the-shelf neural QPP ([[geometry-vs-factuality]]).
- Downstream selective query processing (Sec. 5.2): threshold-based ranker selection "seldom outperforms the individual system", best observed NDCG increase "modest, about 4%"; the direct predicted-value comparison approach degenerated (same configuration chosen for all queries); trained SVM selection reached 0.5322 NDCG vs 0.5322 for the best standalone (BM25 QE) with oracle at 0.5393, and 0.5121 vs 0.5106 (BM25 vs SPLADE, oracle 0.5476) - "not statistically significant". *our-relevance:* this is the exact task template for our SCOPE-vs-raw router; their marginal-gain conclusion makes our conservative regime-routing claim (avoid ~14% of Housing dilution, keep BarExam wins, held-out still pending) the honest and field-consistent framing (C5, C10).
- Statistical reporting discipline: two-fold cross-validation, two-tailed paired t-tests with Bonferroni correction, and explicit critical values (with ~102 queries, r=0.195 is significant at 0.05; 0.254 at 0.01); conclusion recommends plotting correlations even when significant. *our-relevance:* a model for the CIs/significance reporting our rejected paper lacked (C11, C7).

## Bearing on the review

- **C11 (rigor gaps):** This paper shows the reporting standard for prediction-quality claims: paired significance tests, oracle upper bounds, and per-collection breakdowns. A revised SCOPE paper that includes any routing/QPP claim must report tau/r with significance, an oracle-selection bound, and best-standalone comparisons - exactly the Table 12 pattern.
- **C2 (insufficient literature grounding):** Citing Chifu et al. (plus [[faggioli-qpp]] and [[emami-qpp-variant]]) situates our "when does generated-query expansion help" question inside the established QPP/selective-query-processing literature instead of appearing ad hoc.
- **C5 / C10 (marginal gains, regression framed as parity):** Their finding that even in-domain QPP-driven selection is marginal and often not significant lets us frame our router honestly: per-query routing is currently out of reach for everyone; regime-level routing is the defensible claim, and Housing regression must be reported as regression.
- **C7 (no significance test for SCOPE-vs-HyDE):** Their explicit critical-value discipline (r=0.195 at n~102) is a reminder that our +0.5-1.2pp Hit@5 deltas need McNemar/bootstrap CIs or must be called indistinguishable.

## Differentiation

We are not pre-empted: Chifu et al. evaluate QPP for choosing among *rankers/query-expansion configs* on TREC newswire/web collections with short keyword topics; we study QPP-style signals for routing *LLM-generated query expansion (SCOPE/HyDE-family)* on legal/professional QA with long natural-language questions, and we tie routing to a per-query geometric mechanism (CE-affinity movement toward gold, Spearman ~0.44) rather than to score-distribution predictors. Where we overlap, we land on the same side: per-query QPP is too weak to route on (their best tau ~0.37 in-domain, near zero cross-collection; our best no-gold |tau| ~0.11), which is corroboration, not novelty for us. What we must NOT claim: that this paper defines a tau>=0.5 reliability bar (it does not), or that our router negative is surprising (it is the expected outcome given their ceilings). Our genuine additions relative to this paper: the weak-vs-strong query regime taxonomy, the mechanism-level (geometry) predictor, and the regime-level routing result - none of which they attempt.

## Links

- Concepts: [[qpp]], [[regime-routing]], [[weak-vs-strong-query-regime]], [[query-drift]], [[geometry-vs-factuality]], [[generated-query-family]], [[scope]], [[hyde]], [[vocabulary-gap]], [[icml-ai4law-2026-rejection]]
- Sibling sources: [[faggioli-qpp]] (neural-IR QPP "are we there yet"), [[emami-qpp-variant]] (near-twin QPP work), [[weller-drift]] (query drift on strong queries), [[gure]] (legal generative query rewriting), [[koblex-parser]] (closest legal-RAG prior art), [[scope-paper-2026]]

## Raw source

- references/datta-qpp-reliability.pdf (arXiv 2504.01101v1, 17 pp., read pages 1-17; all main tables 1-12 and Figures 1-4 legible)
