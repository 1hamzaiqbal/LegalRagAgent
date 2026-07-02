---
title: Hallucination-Free? AI Legal Research Tools (JELS 2025)
type: source
tags: [legal-rag, hallucination, expert-evaluation, legal-research-process, groundedness]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2405.20362
local: references/magesh2024hallucinationfree.pdf
authors: Magesh et al. (Magesh, Surani, Dahl, Suzgun, Manning, Ho; Stanford RegLab)
year: 2024
venue: arXiv 2024; Journal of Empirical Legal Studies 2025
code: none (dataset released with publication)
---

# TL;DR

The first preregistered, lawyer-graded evaluation of commercial legal RAG (Lexis+ AI, Westlaw AI-Assisted Research, Ask Practical Law AI, vs closed-book GPT-4) on 202 real legal-research queries. RAG reduces hallucination relative to GPT-4 but does not eliminate it (17-33% hallucination), and the paper contributes the correctness-x-groundedness typology plus a failure taxonomy (misreading holdings, confusing legal actors, violating authority hierarchy, fabrication) that is the legal-NLP community's standard frame for what legal RAG must not do. This is the practitioner-process study C1/C2 expect a legal RAG paper to engage.

# Key claims / numbers

- 202 preregistered queries in four categories modeled on real legal research: general legal research 80 (39.6%), jurisdiction/time-specific 70 (34.7%), false premise 22 (10.9%), factual recall 30 (14.9%); 20 queries taken verbatim from LegalBench rule_qa and 20 from BARBRI bar-prep. *our-relevance:* the community's notion of a legal query is an open-ended research question with jurisdiction/time structure — our BarExamQA/HousingQA MC/yes-no setup should be positioned as the discrete-answer, retrieval-measurable slice of this space (C2), not as a stand-in for legal research generally.
- Headline rates: hallucination Lexis+ AI ~17%, Westlaw AI-AR ~33%, Ask Practical Law ~17% (with 63% incomplete), GPT-4 ~43%; accuracy 65/42/19-20/49%. Conditional on giving a responsive answer, Westlaw/Practical Law are not significantly more trustworthy than GPT-4. *our-relevance:* even production-grade legal RAG with premium corpora leaves a large answer-side failure rate — external evidence for our core finding that retrieval lift does not convert to answer lift ([[answer-conversion-gap]]), and a counter to reading C5's "marginal gains" as a SCOPE-specific defect.
- Typology: correctness (correct/incorrect/refusal) x groundedness (grounded/misgrounded/ungrounded); hallucination = incorrect OR misgrounded; groundedness is defined *legally* (a citation to jurisdiction-inapplicable law counts as misgrounded even if faithful to the retrieved text). Inter-rater kappa 0.77, 85.4% agreement, hand-graded by law-trained authors. *our-relevance:* C4 asks about fabricated legal content in SCOPE's pseudo-documents; this typology gives the measurement instrument a revision could apply to p (is p misgrounded?) and to final answers, replacing our current accuracy-only reporting (also C11's rigor complaint).
- Failure taxonomy from hand analysis: misunderstanding holdings (citing CaseHOLD-style holding confusion), failing to distinguish litigant arguments from court statements, violating the hierarchy/order of authority (e.g., asserting a state court reversed SCOTUS), citation suppression of red-flagged overruled cases, and outright fabricated provisions. *our-relevance:* these are retrieval-and-use failures a lawyer cares about; our geometric mechanism story ([[geometry-vs-factuality]]) explains *when* expansion moves retrieval toward gold but says nothing about these use-side failures — honest scope line for the revision.
- Section 3.2 argues legal retrieval is intrinsically hard: queries often lack a single answering document, relevance is jurisdiction- and time-conditioned rather than text-similarity-conditioned, and models may trust training-data priors over retrieved context. *our-relevance:* independent articulation of the weak-vs-strong-query problem and the jurisdiction-filter effect we found on HousingQA (state filter 2.8%->36.9%); citable motivation for [[weak-vs-strong-query-regime]] and metadata filtering as first-class, addressing C8's corpus/jurisdiction-analysis complaint in spirit.

# Bearing on the review

Grounds C4 most directly, plus C1/C2 (practitioner-process framing), C5/C9 (how to frame small answer deltas), and C11 (evaluation rigor norms: preregistration, CIs, IRR). A revised paper must: (1) cite this as the reference point for what legal RAG failure means to practitioners and adopt correctness/groundedness language when discussing SCOPE outputs; (2) report a groundedness-style audit of pseudo-documents p (even sampled), because asserting the a0-discard guardrail without measuring p's factuality is exactly the C4+C12 gap; (3) reframe the HousingQA regression and near-zero LLM-only deltas using their finding that answer-side reliability is the binding constraint industry-wide, while being explicit that our systems are research prototypes, not comparable to Lexis/Westlaw.

# Differentiation

Different object of study: they evaluate closed commercial end-to-end products with human grading on open-ended queries; we evaluate an open, controlled query-generation intervention with exact-match scoring and measured retrieval exposure (Hit@k/MRR against gold passages), which their black-box setting cannot do. We are complementary — mechanism vs audit — and not pre-empted. Where we are exposed: they measure hallucination directly with legal experts; we inferred (in post-submission work) that geometry, not judged factuality, predicts expansion benefit, and an LLM judge is not a lawyer — our factuality falsification claims should be caveated against their hand-grading standard rather than presented as equivalent evidence.

# Links

[[scope]], [[hyde]], [[legal-rag-benchmarks]], [[answer-conversion-gap]], [[geometry-vs-factuality]], [[weak-vs-strong-query-regime]], [[vocabulary-gap]], [[expert-judgment-replication]], [[icml-ai4law-2026-rejection]]; siblings: [[guha2023legalbench]], [[jiang2023syllogism]].

# Raw source

references/magesh2024hallucinationfree.pdf (read pp. 1-18 of 38; results section 6.3 onward and appendices skimmed via abstract/figures only).
