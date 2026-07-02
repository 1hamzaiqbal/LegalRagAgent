---
title: Reliable Retrieval for Large Legal Datasets — DRM & Summary-Augmented Chunking (NLLP 2025)
type: source
tags: [legal-ir, chunking, indexing-side, vocabulary-gap, retrieval-failure, legalbench-rag]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2510.06999
local: references/reuter2025sac.pdf
authors: Reuter et al. (Markus Reuter, Tobias Lingenberg, Rūta Liepiņa, Francesca Lagioia, Marco Lippi, Giovanni Sartor, Andrea Passerini, Burcu Sayin; TU Darmstadt / Florence / Bologna / Trento / EUI)
year: 2025
venue: NLLP Workshop 2025 (ACL Anthology 2025.nllp-1.3; arXiv 2510.06999)
code: https://github.com/DevelopedByMarkus/summary-augmented-chunking
---

# Towards Reliable Retrieval in RAG Systems for Large Legal Datasets

## TL;DR

Defines and quantifies **Document-Level Retrieval Mismatch (DRM)** — the fraction of top-k chunks retrieved from entirely the wrong source document — on LegalBench-RAG (CUAD, MAUD, ContractNLI, PrivacyQA), where boilerplate-heavy legal corpora push standard dense RAG to DRM around 46% at k=1 (over 95% on ContractNLI in a 362-doc pool). Their fix, **Summary-Augmented Chunking (SAC)**, prepends a single ~150-char LLM-generated document summary to every 500-char chunk before embedding, roughly halving DRM and multiplying text-level precision/recall. Counter-intuitively, a *generic* summarization prompt beats an expert-guided legal prompt. This is the corpus-side (indexing-side) attack on the same context/vocabulary gap our query-side generation attacks — and their named future work ("query transformation, expansion, or routing" plus a stronger reranking step) is literally our lane.

## Key claims / numbers

- **DRM definition and severity**: DRM = proportion of top-k retrieved chunks not originating from the ground-truth document. Standard recursive-chunking dense RAG (gte-large, 500-char chunks, FAISS cosine) shows weighted-average DRM ≈46% at k=1 rising to ≈80% at k=64; ContractNLI exceeds 95% DRM in a pool of just 362 NDAs, attributed to boilerplate uniformity. *our-relevance:* an independent, corpus-side quantification of gold burial by structurally-similar distractors — the document-level analog of the distractor geometry our CE gold-affinity margin measures per query ([[vocabulary-gap]]).
- **SAC method**: one LLM call per document (gpt-4o-mini) produces a ~150-char "document fingerprint" summary, prepended to every chunk; chunks re-embedded and indexed as normal. Reported as "effectively halving the mismatch rate" across all k and seeds (Fig. 2), with weighted DRM dropping to roughly 23-38%. *our-relevance:* a one-call-per-*document* offline dual of our one-call-per-*query* online expansion; the same global-context signal can be injected on either side of the embedding similarity, and their side is amortizable across queries.
- **Text-level gains follow**: SAC substantially raises character-level precision and recall over the baseline at all k (Fig. 3; e.g., precision at k=1 rises from ≈0.04 to ≈0.20; recall at k=64 from ≈0.31 to ≈0.62). Selected config (Appendix A): chunk 500 / summary 150 gives averaged Prec 11.03% / Rec 41.80% / DRM 19.29% over k∈{1..64}. *our-relevance:* corpus-side context injection moves both document- and span-level retrieval, so any head-to-head with SCOPE must control the index; our BarExam/Housing indexes are plain-chunk and thus live in their "baseline" condition.
- **Generic beats expert-guided summarization**: a legal-expert "meta-prompt" (structured NDA/privacy-policy templates, Appendix D, built with two legal experts) does *not* outperform the generic prompt on retrieval (Fig. 3); expert summaries retrieve the right document but surface irrelevant boilerplate snippets in their case study. Hypothesized causes: expert cues overfit narrow features; dense structured summaries strain small embedding models. *our-relevance:* a clean legal-domain datapoint that generic generated context beats domain-engineered context — rhyming with our exemplar-SCOPE Phase A null (fixed-medoid legal exemplars didn't help) and cautioning against over-engineering prompt legality (C2 response should cite this, not just claim it).
- **Hybrid BM25 trade-off (Appendix B)**: adding sparse retrieval improves DRM slightly (19.29 → 18.18 at 25/75 semantic/keyword weighting) but *reduces* text-level precision (11.03 → 8.23) and recall; they drop BM25 entirely. *our-relevance:* another instance of pooling/hybrid fusion helping one retrieval level while hurting another — same shape as our raw∪SCOPE pool destroying weak-query gains; fusion is not a free lunch in boilerplate-heavy legal corpora.
- **Retrieval-only scope**: no downstream answer evaluation; DRM→generation impact is explicit future work, as is hierarchical summarization, "applying query optimization methods (e.g., transformation, expansion, or routing) to bridge the semantic gap between user questions and the formal language of legal text chunks," and "adding a reranking step where a more powerful model re-evaluates and re-orders the top-k retrieved chunks." *our-relevance:* their items (ii) and (iii) are respectively our SCOPE/routing lane and our Tinker-trained judge — the two halves of our current story are this group's declared next steps ([[judge-pilot-v0-results]], [[regime-routing]]).
- **Setup details**: thenlper/gte-large embeddings (OpenAI text-embedding-3-large was strongest but rejected for reproducibility/rate limits, Appendix C); English common-law contract/policy documents only; residual mismatch remains significant ("SAC is a valuable component ... but not a complete solution"). *our-relevance:* their corpora are contracts/policies, not statutes or bar-exam doctrine — DRM's severity may differ on our statute corpora (HousingQA's 1.8M statute chunks are boilerplate-heavy too, so a DRM-style audit there is a cheap, citable diagnostic we currently lack).

## Bearing on our thesis

- **Strengthens (mechanism)**: DRM is independent evidence that legal retrieval failure is dominated by *geometric confusability among near-duplicate distractors*, not by missing knowledge — the same conclusion as our P4 falsification (failures geometric, AUC 0.91-0.94) reached from the corpus side. Citing DRM lets us ground the "why legal is the weak-query regime" claim in a published legal-NLP quantity (directly serves C8/C2).
- **Strengthens (judge)**: their observation that retrievers rank textually-similar-but-wrong-document chunks on top is exactly the pooled-gold-burial condition where our trained 9B judge beats the ms-marco CE (Hit@5 20.6% vs 3.8%); their future-work reranking item argues the field wants this component.
- **Threatens**: (1) If a one-off, generic, corpus-side SAC pass halves the mismatch, a reviewer can ask whether query-side expansion is still worth its per-query cost on a SAC-fixed index — we have no SCOPE-on-SAC-index experiment, and the honest framing is "complementary, untested interaction." (2) Their generic-beats-expert result cuts both ways: it supports anti-over-engineering, but also suggests cheap global context (not hypothesized *legal reasoning*) may be the active ingredient — which is compatible with our CE-affinity mechanism but weakens any claim that SCOPE's legal framing per se matters.

## Differentiation

- **Side of the gap**: SAC edits the index (offline, one LLM call per document, amortized); SCOPE edits the query (online, per-query cost, no index rebuild). On a 1.8M-doc corpus like housing_statutes, SAC means 1.8M summarization calls plus re-embedding — non-trivial; per-query expansion scales with query volume instead. Neither paper tests the other's side; the combination is an open experiment.
- **Granularity of failure metric**: DRM is document-level and binary; our CE gold-affinity margin is per-query, continuous, and *predictive* (Spearman ~0.44 with expansion gain). DRM diagnoses the corpus; our margin routes the method.
- **No regime analysis**: SAC is evaluated as always-on; there is no query stratification, no when-does-it-hurt analysis, and no significance testing visible in the main text — our per-query regime framing has no counterpart here.
- **No downstream answers**: like GuRE, retrieval-only; the [[answer-conversion-gap]] is untouched (they defer end-to-end evaluation to future work on Australian Legal QA).

## Links

- [[scope]] — query-side dual of SAC's corpus-side context injection.
- [[hyde]] — they cite Reverse HyDE/QuIM-RAG as the synthetic-question variant of the same chunk-enrichment family.
- [[vocabulary-gap]] — DRM quantifies the corpus-side face of the gap (boilerplate confusability).
- [[weak-vs-strong-query-regime]] — their corpora are weak-signal by structure (near-duplicate documents), complementing our weak-by-query framing.
- [[query-drift]] — expert-guided summaries retrieving boilerplate snippets is drift's indexing-side cousin.
- [[generated-query-family]] — SAC belongs to the generated-*context* family; the wiki family page should note both sides.
- [[regime-routing]] — their future-work "routing" item; also their always-on evaluation is what routing critiques.
- [[geometry-vs-factuality]] — DRM supports the geometric-failure account.
- [[answer-conversion-gap]] — explicitly out of scope for them; our downstream measurements remain differentiating.
- [[judge-pilot-v0-results]] — their future-work reranker = our trained judge.
- [[legal-rag-benchmarks]] — adds LegalBench-RAG (CUAD/MAUD/ContractNLI/PrivacyQA) and the DRM metric to the benchmark map.
- [[icml-ai4law-2026-rejection]] — C2/C8 legal-NLP grounding; a DRM audit of our corpora is a concrete revision item.

## Raw source

- `references/reuter2025sac.pdf` (arXiv 2510.06999v1, 14 pp. incl. appendices; all pages read).
