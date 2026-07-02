---
title: Query2doc (EMNLP 2023)
type: source
tags: [query-expansion, pseudo-documents, llm-retrieval, sparse-retrieval, dense-retrieval, hyde-family]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://aclanthology.org/2023.emnlp-main.585/
local: references/query2doc.pdf
authors: Wang et al. (Liang Wang, Nan Yang, Furu Wei, Microsoft Research)
year: 2023
venue: EMNLP 2023 (main)
code: none (generated pseudo-documents released at https://huggingface.co/datasets/intfloat/query2doc_msmarco)
---

# Query2doc: Query Expansion with Large Language Models

## TL;DR

Query2doc few-shot-prompts an LLM ("Write a passage that answers the given query", 4 in-context examples) to generate a pseudo-document, then **concatenates it with the original query** — repeating the query n=5 times for sparse retrieval to balance term weights, or `concat(q, [SEP], d')` for dense. BM25 gains +3% to +15% (nDCG@10 51.2→66.2 on TREC DL19); dense gains are small (+0.4 to +1.4 MRR@10) and shrink as the retriever gets stronger. The pivotal Table 4 ablation shows pseudo-document-ONLY retrieval is far worse than query+doc concatenation on sparse retrieval — even worse than the raw query alone.

## Key claims / numbers

- **Method**: pseudo-document d' from few-shot GPT-3 (text-davinci-003, k=4 examples, temp 1, max 128 tokens); sparse query `concat({q}×5, d')`, dense query `concat(q, [SEP], d')`. *our-relevance:* this is the canonical keep-the-query member of the generated-query family SCOPE belongs to (C3); SCOPE embeds only the pseudo-doc, i.e., the design query2doc explicitly rejected.
- **Table 4 (keep-vs-discard, sparse/BM25)**: TREC DL19 nDCG@10 — query+pseudo-doc concat **66.2**, query only **51.2**, pseudo-doc only **48.7**. TREC DL20 — concat **62.9**, query only **47.7**, pseudo-doc only **44.5**. Pseudo-doc-only is below even the raw query. "The original query and pseudo-documents are complementary." *our-relevance:* strongest published evidence that discarding the raw question (as SCOPE does) can be the weaker design; this is exactly the C12 ablation we never ran. Caveat: this table is BM25/sparse only — the dense keep-vs-discard is not cleanly ablated (HyDE at 61.3/57.9 vs DPR+query2doc 68.7/67.1 uses different underlying retrievers).
- **Sparse main results**: BM25 MRR@10 18.4→21.4 on MS MARCO dev; TREC DL19 51.2→66.2, DL20 47.7→62.9. *our-relevance:* their headline gains come from web/TREC queries, a strong-query regime by our taxonomy; nothing here is a weak-query/vocabulary-gap result (C8).
- **Dense results diminish with retriever strength**: DPR +1.4, SimLM +0.4, E5+KD +0.8 MRR@10; "gains tend to be diminishing when distilling from a strong cross-encoder based re-ranker". *our-relevance:* directly parallels C5 (marginal gains) and our query-drift/strong-query finding — expansion helps least where the base system is already strong.
- **BEIR zero-shot mixed**: DBpedia +5.7 BM25 nDCG@10, but NFCorpus/Scifact show "a minor decrease" for dense (Scifact −2.9 for both SimLM and E5). *our-relevance:* published precedent that LLM expansion regresses on some corpora, matching our HousingQA regression (C10) and BEIR net-negative finding — regressions should be framed as regime effects, not hidden as "parity".
- **Scale is critical (Table 3)**: TREC DL19 BM25+expansion 52.0 (1.3B) → 55.1 (6.7B) → 66.2 (davinci-003) → 69.2 (GPT-4); "small language models only provide marginal improvements". *our-relevance:* our 8B–70B model axis is a defensible design choice this paper legitimizes.
- **Factual errors acknowledged**: Table 5 case study shows a fabricated date in a pseudo-doc; "such errors... pose a significant challenge to building trustworthy systems". GPT-4 self-verification (Appendix B) barely changes outputs (69.2→68.6 DL19). *our-relevance:* C4's fabrication concern is a known, citable property of the family — and our geometry-vs-factuality falsification (AUC 0.79–0.94 vs 0.55–0.58) is a genuine advance over their anecdotal treatment.
- **Rigor practices**: variance over 3 random runs reported (DL19 64.8 ±1.14, DL20 60.9 ±1.63, Table 10); latency table includes the >2000ms LLM call vs 16ms BM25 (Table 6); ~550k API calls ≈ $5k disclosed. *our-relevance:* sets the reporting bar C11 says we missed — error bars, and cost accounting that includes the generation stage.

## Bearing on the review

- **C3 / C7 (HyDE-family novelty, untested snap-answer delta)**: query2doc is a must-cite sibling; its own Related Work section already distinguishes concat-expansion from HyDE's pseudo-doc-only embedding ("HyDE implicitly assumes that the groundtruth document and pseudo-documents express the same semantics... which may not hold for some queries"). A revised paper must position SCOPE on this concat-vs-replace axis explicitly and test the snap-conditioning delta with significance.
- **C12 (guardrail/keep-vs-discard ablation)**: Table 4 is the template. A revised paper needs the 2×2: {raw query only, pseudo-doc only (SCOPE as-is), concat/pool, concat+a0-passed-through} on both benchmarks. Our post-submission raw∪SCOPE pooling result is exactly this ablation for the retrieval side — it belongs in the paper, not the repo.
- **C11 (rigor)**: adopt their practices — multi-run variance, and token/latency accounting that includes first-stage generation.
- **C5 / C10**: their honest "gains diminish with strong retrievers" and BEIR regressions give respectable framing language for marginal deltas and the HousingQA regression.

## Differentiation

Query2doc pre-empts the generic claim "LLM-generated pseudo-documents improve retrieval" — that is settled 2023 work, and on their (strong-query, web) benchmarks the query-keeping concat design beats the pseudo-doc-only design SCOPE uses. We are **not** pre-empted on: (1) the weak-query regime — our BarExamQA result (raw Hit@5 1.4% → 12%) is a regime they never test, and our 3SCOPE+raw pooling experiments show the **opposite** of their Table 4 there (pooling with the raw query destroys the weak-query gain: 3.9% vs 12.0% Hit@5), so their concat-wins conclusion is regime-dependent, which is our regime-routing contribution; (2) the per-query geometric mechanism (CE-affinity movement, Spearman ~0.44); (3) the falsification that factuality, which they only flag anecdotally, does not explain expansion failure. Honest concession: SCOPE's submitted paper cited neither query2doc nor ran its Table 4 ablation, and on strong-query corpora (HousingQA, BEIR) their design likely dominates ours.

## Links

[[scope]], [[hyde]], [[generated-query-family]], [[vocabulary-gap]], [[weak-vs-strong-query-regime]], [[query-drift]], [[regime-routing]], [[geometry-vs-factuality]], [[answer-conversion-gap]], [[icml-ai4law-2026-rejection]], [[gure]], [[koblex-parser]]

## Raw source

- references/query2doc.pdf (ACL Anthology, EMNLP 2023 main, pp. 9414–9423)
