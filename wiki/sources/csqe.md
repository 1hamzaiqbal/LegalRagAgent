---
title: Corpus-Steered Query Expansion (EACL 2024)
type: source
tags: [query-expansion, prf, hallucination, retrieval, hyde-family, knowledge-gap]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2402.18031
local: references/csqe.pdf
authors: Lei et al.
year: 2024
venue: EACL 2024 (short)
code: https://github.com/Yibin-Lei/CSQE
---

# Corpus-Steered Query Expansion with LLMs (CSQE)

## TL;DR

CSQE replaces pure LLM-generated expansion text with **real sentences extracted from the corpus**: BM25 retrieves top-10 documents, an LLM (GPT-3.5-Turbo, one-shot prompt) identifies which are relevant and extracts key sentences, and the query is expanded with those sentences plus a smaller dose of KEQE (HyDE-style hypothetical-answer) generations. The explicit motivation is that purely parametric expansions suffer from **hallucination, outdated information, and long-tail knowledge deficiency** — i.e., a knowledge-gap attribution for expansion failure. Training-free BM25+CSQE beats the supervised Contriever^FT on TREC DL19/20 and six low-resource BEIR datasets, and shines most on NovelEval (post-GPT-4 queries), where the pure-LLM expansion baseline actually falls below BM25.

## Key claims / numbers

- **Method**: BM25 top-10 (128-token truncation; Arguana top-3), one-shot LLM relevance assessment + key-sentence extraction (S = {s1..sn}); final query = repeated original q + extracted sentences + KEQE generations (N=2 corpus-originated + N=2 KEQE, vs N=5 for the KEQE baseline). GPT-3.5-Turbo-0301, temperature 1.0, Pyserini BM25. *Our-relevance:* CSQE is the corpus-grounded pole of the [[generated-query-family]]; its first stage is a retrieval pass, which is exactly the dependency that fails in weak-query regimes (C6-adjacent novelty mapping).
- **Faithfulness of extraction**: in their preliminary study, 830/1000 extracted "key sentences" were verbatim identical to sentences in the initially retrieved documents. *Our-relevance:* CSQE really does inject near-verbatim corpus text, so it is a genuine hallucination-free contrast condition against SCOPE/HyDE pseudo-documents (C4).
- **Web search**: DL19 BM25+CSQE mAP 47.2 / nDCG@10 67.3 vs KEQE 45.0 / 65.9 and Contriever^FT 41.7 / 62.1; DL20 CSQE 46.5 / 66.2. CSQE beats KEQE on 5/6 metrics with fewer LLM generations. *Our-relevance:* on strong-first-pass corpora, steering with real text beats pure generation — consistent with our BEIR finding that ungated generative expansion is net-negative on strong queries (C5, C9 framing discipline).
- **Low-resource BEIR (nDCG@10)**: CSQE improves BM25 on all 6 datasets (avg 43.7 → 49.7); RM3 PRF *hurts* on 5/6 (42.4). Q2D/PRF has the best average (50.1). *Our-relevance:* even in their own tables the corpus-steered method does not uniformly dominate generation-plus-PRF; the win is robustness, mirroring our SCOPE-vs-HyDE robustness result.
- **NovelEval (knowledge-gap testbed)**: on queries published after GPT-4's release, KEQE *drops* nDCG@10 to 62.0 vs BM25 68.4, while CSQE reaches 82.6 (nDCG@1 85.7 vs BM25 61.9), beating a GPT-3.5 reranker (75.7). This is their headline causal evidence that "reduction of hallucination leads to the performance improvements." *Our-relevance:* this is the cleanest published statement of the hallucination/knowledge-gap attribution that our [[geometry-vs-factuality]] falsification tests — we find geometry (AUC 0.79–0.94), not judged factuality (0.55–0.58), predicts expansion failure.
- **Their own failure precondition, stated in passing**: "if LLMs find no relevant documents in the initially retrieved set, they will yield no expansions" (framed as an advantage over Q2D/PRF noise). *Our-relevance:* this is precisely the mechanism of the collapse we measured on BarExamQA — raw Hit@5 ~1.4% means there is nothing real to steer with; our runs show CSQE Hit@5 2.0% vs SCOPE 12.1% (CE-delta −0.5 vs +3.9) on that weak-query legal regime ([[weak-vs-strong-query-regime]]).
- **Model scaling**: CSQE > KEQE across Llama2-Chat 7B/13B/70B and GPT-3.5; performance rises with model size. Also large gains stacking CSQE on unsupervised Contriever (mAP 24.0 → 44.0 on DL19). *Our-relevance:* method-over-model consistency parallels our three-model / three-retriever generality checks (C11).
- **Limitations section explicitly names legal case retrieval** as a latency-tolerant domain where CSQE "may offer benefits." *Our-relevance:* they gesture at legal IR but never test it; our legal results show where the recipe breaks (C2, C8).

## Bearing on the review

- **C4 (fabricated legal content risk in pseudo-documents)**: CSQE is the field's direct answer to exactly this criticism — ground the expansion in real corpus text instead of parametric generation. A revised SCOPE paper **must cite CSQE**, present it as the corpus-grounded alternative, and then report our measured crossover: CSQE is robust on strong-query BEIR but collapses on weak-query BarExamQA (Hit@5 2.0% vs SCOPE 12.1%), because the corpus-steering step requires a usable first-pass retrieval that the weak regime denies. That turns C4 from an unanswered risk into a tested trade-off: hallucination-safe expansion is available, and it is precisely the regime where SCOPE wins that it cannot serve.
- **C6/C7 (novelty within the expansion family, untested prior art)**: CSQE belongs in the related-work taxonomy alongside HyDE/Query2doc/GRF as the PRF-grounded branch; positioning SCOPE against it (and actually running it, as we did) is the kind of closest-prior-art comparison the reviewers said was missing.
- **C5/C9 (marginal gains, weakest-baseline framing)**: CSQE's NovelEval result shows pure-LLM expansion can go *below* the raw baseline — supporting the revised framing that generative expansion needs gating/routing rather than blanket claims ([[regime-routing]], [[query-drift]]).
- **Falsification headline support**: CSQE operationalizes the knowledge-gap attribution (NovelEval as "LLMs can only hallucinate" testbed). Our factuality-judge experiments are a direct test of that attribution on legal/BEIR data and find geometry dominates; citing CSQE makes clear we are engaging the strongest published version of the story, not a strawman.

## Differentiation

- **Where they pre-empt us**: the observation that pure parametric expansions fail from hallucination/staleness/long-tail gaps, and that corpus grounding fixes this on knowledge-gap queries, is theirs (2024). Any SCOPE claim about expansion-failure attribution must be framed relative to CSQE, not as a new observation. They also already note the "no relevant initial documents → no expansion" behavior, so our collapse mechanism has a one-line antecedent in their paper (they framed it as a *feature*).
- **Where we differ**: (1) they never evaluate a regime where first-pass retrieval is near-zero — all their corpora give BM25 a workable starting set; our BarExamQA result shows the method degenerates there, inverting their robustness story. (2) They attribute failure to knowledge/hallucination and validate only on NovelEval; we test the attribution per-query with LLM-judged factuality vs retrieval geometry and find factuality is a weak predictor — a mechanism-level correction, not a replication. (3) They optimize sparse BM25 expansion; SCOPE targets dense+CE reranking in a domain-mismatched (legal) corpus, where the [[vocabulary-gap]] rather than knowledge recency is the binding constraint. (4) No answer-stage evaluation in CSQE — retrieval metrics only; our [[answer-conversion-gap]] finding has no counterpart there.

## Links

- [[scope]], [[hyde]], [[query2doc]] — the parametric-expansion methods CSQE steers/contrasts
- [[generated-query-family]] — CSQE = corpus-grounded PRF branch
- [[weak-vs-strong-query-regime]], [[regime-routing]] — our crossover: CSQE robust on strong, collapses on weak
- [[geometry-vs-factuality]] — our falsification of CSQE's knowledge-gap attribution
- [[query-drift]], [[weller-drift]] — CSQE cites Weller et al. 2023 for expansion degradation
- [[vocabulary-gap]], [[legal-rag-benchmarks]], [[gure]] — legal-IR context CSQE mentions but never tests
- [[icml-ai4law-2026-rejection]] — bears on C4, C5, C6, C7, C9

## Raw source

- `references/csqe.pdf` (arXiv:2402.18031v1, 28 Feb 2024; 9 pp incl. appendix — full text read, including prompts and dataset statistics tables)
