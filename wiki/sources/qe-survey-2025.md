---
title: Query Expansion in the PLM/LLM Era — Comprehensive Survey (arXiv 2025)
type: source
tags: [query-expansion, survey, taxonomy, hyde-family, query-drift, qpp]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2509.07794
local: references/qe-survey-2025.pdf
authors: Li et al. (Minghan Li, Xinxuan Lv, Junjie Zou, Tongna Chen, Chao Zhang, Suchao An, Ercong Nie, Guodong Zhou; Soochow University + LMU Munich)
year: 2025
venue: arXiv cs.IR (2509.07794, v1 Sep 2025, v3 7 May 2026)
code: https://github.com/lmh0921/QueryExpansion-PLM-LLM-Survey-paperList
---

# Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey

## TL;DR

The first dedicated, IR-focused survey of query expansion in the PLM/LLM era (42 pp., 145 refs). It organizes modern QE along four axes — point of injection (implicit embedding-level vs explicit text), grounding & interaction (zero-grounding / grounding-only / interactive), learning & alignment, and KG augmentation — plus a deployment blueprint (Table 6) and open challenges. Its category names give us the exact vocabulary to position SCOPE, and its deployment guidance ("use zero-grounding generative QE selectively for hard or underspecified queries, not as an always-on default") is our regime-routing thesis stated as received wisdom, with the per-query predictor left as an open gap.

## Key claims / numbers

- **The organizing taxonomy (Fig. 1, §4)**: four complementary axes — (1) *Point of injection*: implicit embedding-based QE (ANCE-PRF, ColBERT-PRF, Eclipse, QB-PRF, LLM-VPRF) vs selection-based explicit QE (CEQE, SQET, BERT-QE, CQED, PQEWC); (2) *Grounding & interaction*; (3) *Learning & alignment* (SoftQE, RADCoT, ExpandR, AQE); (4) *KG-augmented* (KGQE, CL-KGQE; hybrid KAR, QSKG). *our-relevance:* these are the canonical names for positioning; SCOPE is an explicit, text-level, generated-expansion method, so implicit/embedding QE and selection-based QE are the sibling families a related-work section must delimit against (C2, C3).
- **Zero-Grounding, Non-Interactive QE (§4.2.1)** = LLM-driven single-stage expansion with no corpus evidence: Query2Doc, CoT-QE, GAR, GRF, **HyDE**, Exp4Fuse, contextual clue sampling, SEAL, SU-RankFusion, Two-LLM QE. Table 2 takeaway: "Useful for sparse evidence, but risks hallucination or drift." *our-relevance:* this is SCOPE's precise taxonomic home — the answer to C3 ("just HyDE") is that SCOPE is a member of this named family and our contribution is the per-query mechanism for when the family helps, not the family itself.
- **Grounding-Only, Non-Interactive QE (§4.2.2)** = corpus-evidence-anchored single pass: MILL, AGR, EAR, GenPRF, **CSQE**, MUGI, PromptPRF, FGQE. CSQE described as combining LLM-judged relevant sentences from first-pass results with an ungrounded hypothetical document; Table 6 lists this family's main risk as "inherits errors, omissions, and bias from first-pass evidence; effectiveness depends on grounding depth and filtering." *our-relevance:* names the exact mechanism behind our CSQE-collapses-on-weak-BarExam result — grounding-only QE presupposes a usable first pass, which weak-query legal retrieval does not supply ([[weak-vs-strong-query-regime]]).
- **Deployment guidance (Table 6)**: zero-grounding single-round generative QE — main risks "unsupported additions, topic drift, and model/version sensitivity"; practical guidance "**Best used selectively for hard or underspecified queries, rather than as an always-on default**." *our-relevance:* our regime thesis (expansion net-negative on strong queries, wins on weak) is the mechanism-level substantiation of exactly this one-line folklore; the survey offers no per-query decision rule, which is the gap our CE gold-affinity margin fills ([[regime-routing]]).
- **Exp4Fuse note (§4.2.1)**: "regressions on already-strong queries suggest invoking it selectively using simple query-quality predictors." *our-relevance:* the only place the survey names QPP-style routing — one sentence, no predictor evaluated. Our result that no standard no-gold QPP clears τ≥0.5 per-query but regime-level routing works is a direct deepening of this line (C6-adjacent positioning).
- **Query2Doc effect sizes**: reliably improves BM25 on MS MARCO/TREC DL, "up to +15.2% nDCG@10 on DL'20"; trade-offs are single-shot hallucinations and decoding latency. HyDE's stated main risks: "drift from fanciful generations and added encoding cost." *our-relevance:* the survey's headline effect sizes come from web/ODQA corpora with strong lexical anchoring; none of its evidence covers legal corpora, where we show the sign of the effect flips by query regime.
- **Learning & alignment axis (§4.3)**: SoftQE, RADCoT, ExpandR, AQE align generation with retrieval utility via SFT/preference/distillation; EAR trains a reranker to pick the candidate expansion that most raises gold rank. *our-relevance:* GuRE (our C8 prior art) slots here as the trained tier; EAR is the nearest survey ancestor of per-candidate expansion selection, and our Tinker-trained relevance judge is the retrieval-side sibling (trained judge over pooled candidates rather than over expansions) ([[judge-pilot-v0-results]]).
- **Open challenges (§7.1)**: field needs "low-cost faithfulness checks that go beyond expensive LLM-as-a-judge pipelines," "selective verification that is triggered only for uncertain cases," and "monotonicity-oriented testing" where an expansion is accepted only if it does not harm retrieval under counterfactual checks. *our-relevance:* our geometry-vs-factuality falsification (geometry AUC 0.79-0.94 vs real-factuality AUC ≈0.55-0.58, marginal lift of factuality ≈0) answers the §7.1 agenda with a twist: for retrieval gain, cheap geometric checks are not merely a proxy for faithfulness — faithfulness is largely irrelevant ([[geometry-vs-factuality]]).
- **Evaluation guidance (§6.3)**: aggregate benchmark scores "obscure practically important differences"; results should be stratified "by query type, ambiguity, domain specificity, or downstream use case"; QE should be judged with the downstream pipeline role, not a single end-task number. *our-relevance:* license for our per-query, per-regime reporting style, and for treating the [[answer-conversion-gap]] as a first-class outcome separate from retrieval exposure.
- **Coverage gap — no legal IR**: application domains covered are web search, biomedical, e-commerce, cross-lingual, ODQA, RAG, conversational, and code search; legal appears only in a passing mention of domain ontologies. *our-relevance:* the authoritative QE survey has no legal application section — our legal weak-query lane is genuinely uncovered territory in this taxonomy, which both supports novelty and explains why legal-NLP reviewers (C2, C8) and IR reviewers read different literatures.

## Bearing on our thesis

- **Strengthens**: gives us canonical family names (zero-grounding generative QE; grounding-only QE; learning/alignment QE) so the mechanism paper can be positioned as "a per-query predictive account of when zero-grounding generative QE helps," rather than a new method — precisely the reframe the rejection demanded. Table 6's "selectively, not always-on" guidance plus the Exp4Fuse QPP sentence show the field *believes* our conclusion but has never measured it; we supply the measurement (7 datasets × 3 retrievers, Spearman ≥0.30 pooled everywhere).
- **Strengthens (judge)**: §7.1's call for verification/selection layers and the EAR/learning-alignment axis make the trained-judge pilot a legible next step in taxonomy terms: moving from prompt-only generation to learned selection is the survey's own predicted trajectory.
- **Threatens / must handle**: (1) family completeness — MUGI (multi-pseudo-doc integration) and contextual clue sampling are published multi-candidate expansion methods; our 3SCOPE "diversity adds nothing" result must be positioned against MUGI's claimed gains (different regime: MUGI's gains are on TREC DL/BEIR strong-query corpora with calibration, ours is legal weak-query). (2) SEAL shows corpus-constrained generation (FM-index-valid substrings) sidesteps hallucination entirely — a structural alternative to our gating that a reviewer could name. (3) Survey attributes generative-QE failures to drift/faithfulness; our falsification says geometry, not factuality — we contradict the survey's implicit causal story and should say so explicitly.

## Differentiation

- The survey is a map, not a result; it contains no per-query analysis, no predictor evaluation, and no legal benchmark. Our contribution occupies its two self-declared gaps: (a) "simple query-quality predictors" for selective invocation (named, never tested) and (b) low-cost checks beyond LLM-as-a-judge (named as future work).
- Honest caveat: with this taxonomy in hand, "SCOPE" cannot be presented as method novelty — it is a two-call zero-grounding generative QE variant. The paper spine must be mechanism + regime + judge, with SCOPE as the instrument.
- Citation hygiene: use their exact axis names when classifying HyDE/Query2Doc/CSQE/GuRE ([[generated-query-family]] page should adopt this vocabulary), and cite Exp4Fuse and EAR when introducing routing and judge respectively — otherwise we recreate C6 (uncited near-twins) inside the IR literature.

## Links

- [[scope]] / [[hyde]] — both members of the survey's zero-grounding, non-interactive generative QE family.
- [[generated-query-family]] — should be renamed/reorganized around the survey's four axes.
- [[vocabulary-gap]] — the survey's opening framing ("vocabulary mismatch" as QE's core target).
- [[weak-vs-strong-query-regime]] — Table 6 "selectively for hard or underspecified queries" is the folklore version.
- [[query-drift]] — the survey's named main risk for our family; ties to Weller'24.
- [[regime-routing]] — Exp4Fuse's "simple query-quality predictors" sentence is the survey's only routing pointer.
- [[geometry-vs-factuality]] — §7.1 faithfulness-check agenda; our falsification refines it.
- [[answer-conversion-gap]] — §6.3 downstream-role evaluation guidance.
- [[judge-pilot-v0-results]] — learning/alignment axis (EAR) is the nearest ancestor.
- [[legal-rag-benchmarks]] — legal IR is absent from the survey's application domains.
- [[icml-ai4law-2026-rejection]] — supplies the C3 reframe and the C2/C6 citation vocabulary.

## Raw source

- `references/qe-survey-2025.pdf` (arXiv 2509.07794v3, 42 pp.; read pp. 1-19 and 30-37 — taxonomy, method families, deployment blueprint, open challenges).
