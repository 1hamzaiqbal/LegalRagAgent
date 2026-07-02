---
title: HyDE - Precise Zero-Shot Dense Retrieval without Relevance Labels (ACL 2023)
type: source
tags: [hyde, dense-retrieval, zero-shot, query-expansion, pseudo-documents, retrieval]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://aclanthology.org/2023.acl-long.99/
local: references/hyde.pdf
authors: Gao et al. (Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan)
year: 2023
venue: ACL 2023 (Long Papers)
code: https://github.com/texttron/hyde
---

# TL;DR

HyDE decomposes zero-shot dense retrieval into (1) an instruction-following LLM (InstructGPT, temp 0.7) that generates a *hypothetical document* answering the query, and (2) an **unsupervised** contrastive encoder (Contriever) that embeds that fake document and searches a document-only embedding space; the encoder's "dense bottleneck" is argued to act as a lossy compressor that filters out hallucinated details. With zero relevance labels it roughly doubles unsupervised Contriever on TREC DL19/20 and beats BM25, approaching fine-tuned retrievers. It is evaluated purely as a retrieval method (mAP/nDCG/recall) — never on downstream answer accuracy.

# Key claims / numbers

- **Exact mechanism**: sample N hypothetical documents d̂_k from `InstructLM(q, INST)`; the search vector is the mean of the document-encoder vectors, and Eq. 8 explicitly **also averages in the query itself** encoded by the document encoder: v̂ = 1/(N+1)[Σ f(d̂_k) + f(q)]. So canonical HyDE = multi-sample averaging + query-vector mixing. N is not stated in the paper body (the repo defaults it). *our-relevance:* SCOPE deviates on both knobs — single sample, pseudo-doc only, no query mixing — and none of these deviations were ablated (feeds C7, C11).
- **Encoder choice is theoretically load-bearing**: HyDE requires an *unsupervised contrastively learned* encoder (Contriever) so that retrieval runs in a document-document similarity space; Table 6 shows gains shrink on fine-tuned encoders (Contriever-ft DL19 mAP 41.7→48.6; GTR-XL 46.7→50.6, with smaller gains "presumably because it has not been contrastively pre-trained to explicitly learn document-document similarity"). *our-relevance:* SCOPE uses supervised gte-large-en-v1.5 + cross-encoder rerank, outside HyDE's motivating regime — consistent with our BEIR finding that expansion is net-negative for strong queries on strong retrievers (query drift), and with C3's demand to state what we actually inherit.
- **Headline retrieval numbers**: DL19 Contriever mAP 24.0 → HyDE 41.8 (nDCG@10 44.5→61.3, Recall@1k 74.6→88.0); DL20 mAP 24.0→38.2. On 7 low-resource BEIR tasks HyDE beats Contriever everywhere on nDCG@10 and loses to BM25 only on TREC-COVID (59.5 vs 59.3). Mr.TyDi: improves mContriever in sw/ko/ja/bn but trails fine-tuned mContriever-ft. *our-relevance:* HyDE's gains are largest exactly where the base retriever is weak — the same weak-query/weak-baseline pattern behind C9's complaint that SCOPE's 8x lift is measured against a floor (raw Hit@5 1.4%).
- **Domain-styled instructions are already canonical HyDE** (Appendix A.1): "write a *scientific paper* passage" (SciFact/TREC-COVID), "a *financial article* passage" (FiQA), "a *news* passage" (TREC-NEWS), "a *Wikipedia* passage to verify the claim" (Climate-Fever), a counter-argument (Arguana). *our-relevance:* SCOPE's "style of formal legal authority" instruction is squarely within this template family; it cannot be presented as legal-expertise novelty (C3), and per-benchmark prompt variation is precedented.
- **Hallucination is acknowledged, not measured**: the generated document "is not real, can contain factual errors" and the dense bottleneck is *asserted* to filter "extra (hallucinated) details"; Limitations adds LLM generation bias may bias search results. No factuality-vs-benefit measurement exists. *our-relevance:* our geometry-vs-factuality falsification (geometry AUC ~0.79-0.94 vs LLM-judged factuality ~0.55-0.58) is the first direct test of this asserted mechanism we know of, and it answers C4's "fabricated content" objection with evidence rather than assertion.
- **Generator quality matters, weaker LMs are unstable**: Table 4 (Flan-T5 < Cohere < InstructGPT on DL19/20); Table 5: 3-shot base GPT-3 "performs less stably." *our-relevance:* grounds our 8B-vs-70B generator axis; small-model SCOPE rows have precedent for instability.
- **No answer-accuracy evaluation anywhere**; HyDE is framed as a bootstrap for the early life of a search system, to be retired once relevance labels accumulate. *our-relevance:* the retrieval-to-answer conversion gap (72.3→72.9 despite 8x retrieval lift) is genuinely outside HyDE's scope — our contribution, but also our burden (C5, C9).

# Bearing on the review

- **C3 ("essentially HyDE applied to the legal domain")**: largely correct on the generation side and must be conceded. A revised paper must (a) cite HyDE as the parent and A.1's domain-styled instructions as precedent, (b) enumerate the exact deltas: snap-answer conditioning, single-sample pseudo-doc-only embedding (no Eq.-8 query mixing, no N-sample averaging), supervised embedder + CE rerank, and downstream answer evaluation, and (c) test which deltas matter.
- **C7**: since snap-conditioning is the only generative delta vs HyDE, the revision needs a significance-tested SCOPE-vs-HyDE comparison *and* ablations of the inherited-but-dropped HyDE knobs (query mixing, multi-sample averaging) — dropping them is itself an untested implementation choice.
- **C4**: cite HyDE's own hallucination framing (fake-document tolerance is the method's stated premise, not our inconsistency) and back it with our factuality-falsification result.
- **C9/C5**: HyDE's gain profile (huge over weak baselines, shrinking under supervision/strong retrieval) predicts exactly the pattern reviewers flagged; the revision should present regime-conditioned claims, not floor-relative headlines.
- **C12**: no analogue in HyDE (no snap answer exists there); the a0-passing ablation is entirely on us.

# Differentiation

SCOPE is honestly a HyDE-family method: pseudo-document generation, domain-styled instructions, and the hallucination-tolerance argument are all pre-empted here. What HyDE does *not* contain: any per-query analysis (only dataset averages), any answer-accuracy evaluation, any failure characterization on strong queries beyond the shrinking-gains observation, snap-answer conditioning, and reranking. Our defensible territory is the mechanism/regime layer — CE-affinity geometry (Spearman ~0.44), the geometry-vs-factuality falsification, weak-vs-strong regime routing, and the answer-conversion gap — not the generation recipe. Note also a precision point for the revision: our implementation is *not* faithful HyDE (single sample, no query-vector averaging, supervised encoder, CE rerank), so "HyDE" baselines in our tables should be labeled as our re-implementation.

# Links

[[scope]], [[generated-query-family]], [[vocabulary-gap]], [[weak-vs-strong-query-regime]], [[query-drift]], [[qpp]], [[answer-conversion-gap]], [[geometry-vs-factuality]], [[regime-routing]], [[legal-rag-benchmarks]], [[icml-ai4law-2026-rejection]], [[koblex-parser]], [[gure]], [[scope-paper-2026]]

# Raw source

- references/hyde.pdf (ACL Anthology 2023.acl-long.99, pp. 1762-1777, all 16 pages read including Appendix A.1 instructions)
