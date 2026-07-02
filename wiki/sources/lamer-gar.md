---
title: LameR (arXiv 2023) + GAR (ACL 2021) — answer-conditioned query expansion
type: source
tags: [query-expansion, answer-augmentation, zero-shot-retrieval, bm25, open-domain-qa, prior-art]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2304.14233 ; https://arxiv.org/abs/2009.08553
local: references/lamer.pdf ; references/gar.pdf
authors: Shen et al. (LameR); Mao et al. (GAR)
year: 2023 (LameR); 2021 (GAR)
venue: arXiv preprint (LameR, "Under review"); ACL 2021 (GAR)
code: https://github.com/morningmoni/GAR (GAR); none found for LameR
---

# LameR + GAR: answer-conditioned expansion prior art

## TL;DR

Two papers that together bound the novelty of SCOPE's "snap-answer conditioning." **GAR** (Mao et al., ACL 2021) expands a question with a *trained* BART generator's outputs — the answer itself, the sentence containing it, and the passage title — appends them to the question, and retrieves with BM25; it matches or beats DPR on NQ/TriviaQA. **LameR** (Shen et al., arXiv 2304.14233) makes this zero-shot: it prompts an LLM with the query plus top-10 BM25-retrieved candidates to generate 5 potential *answers*, concatenates query+answers, and retrieves with BM25, beating HyDE substantially on DL19/DL20. Both explicitly condition expansion on answer generation and both keep the original query in the final retrieval string — the design space SCOPE occupies was mapped by 2023.

## Key claims / numbers

- **GAR method**: supervised seq2seq (BART-large) generates three query contexts from the question — (1) the answer (the "default target"), (2) the sentence containing the answer, (3) the title of a passage containing the answer — then *appends* them to the question ("does not rewrite the query but expands it") and retrieves with BM25; results from the three context types are fused. *our-relevance:* generating an answer and using it to steer retrieval is GAR's default target from 2021, so SCOPE's snap answer a0 is not a new primitive (C6, C7).
- **GAR verified claim on answer-alone retrieval** (Sec 3.3, quoted): "conducting retrieval with the generated contexts (e.g., answers) alone as queries instead of concatenation is ineffective because (1) some of the generated answers are rather irrelevant, and (2) a query consisting of the correct answer alone (without the question) may retrieve false positive passages with unrelated contexts that happen to contain the answer." **Verified present; it is a stated observation with no supporting table** — neither the main text nor Appendix A gives numbers for the answer-alone condition. *our-relevance:* this is the closest published rationale for NOT retrieving on the answer alone; SCOPE's opposite design (embed only the pseudo-doc p, drop the raw question entirely from retrieval) contradicts GAR's advice and works anyway in the weak-query regime — a real, arguable difference (C7, C9).
- **GAR numbers (NQ test, top-k retrieval accuracy)**: BM25 43.6/62.9/78.1 (top-5/20/100); BM25+RM3 44.6/64.2/79.6; DPR 68.3/80.1/86.1; GAR 60.9/74.4/85.3; GAR+DPR fusion 70.7/81.6/88.9. TriviaQA: GAR 73.1 top-5 beats DPR 72.7. End-to-end extractive EM 41.8 NQ / 62.7 Trivia (SOTA extractive at the time). *our-relevance:* GAR's retrieval lifts translate to answer-EM lifts because the extractive reader is retrieval-bound; SCOPE's 8x retrieval lift moving answers only 72.3→72.9 is the opposite — our [[answer-conversion-gap]] is the story, and GAR shows the coupling is task-dependent, not automatic (C5, C9).
- **GAR vocabulary-gap quantification**: ROUGE-1/2/L F1 overlap between query and ground-truth passage rises from 6.00/2.36/5.01 (original question) to 13.21/6.99/10.27 (answer-sentence-augmented). *our-relevance:* a 2021 corpus-level lexical-overlap analysis of exactly the kind C8 says we lack.
- **GAR hallucination stance** (Sec 3.2): generated contexts "involve unfaithful or nonfactual information due to hallucination... they are beneficial rather than harmful overall," and fusing 3 context types "alleviates the distraction of hallucinated content." *our-relevance:* pre-figures our [[geometry-vs-factuality]] falsification (factuality of the expansion is not what determines retrieval benefit) and gives C4 a citable counter-argument (C4).
- **LameR method**: zero-shot; retrieve top-M=10 candidates with BM25, prompt gpt-3.5-turbo with query + candidates ("most of these passages are wrong... please write a correct answering passage"), sample N=5 answers, form q̄ = Concat(q, a1, q, a2, ..., q, aN) — the query is *repeated* alongside every answer — retrieve with BM25 at K=1000. *our-relevance:* LameR is answer-conditioned AND evidence-conditioned expansion; it keeps the query in the retrieval string, where SCOPE keeps neither the query nor the answer (only the pseudo-doc p) (C6, C7).
- **LameR numbers**: DL19 nDCG@10 — BM25 50.6, HyDE 61.3, LameR 69.1; DL20 — BM25 48.0, HyDE 57.9, LameR 64.8. BEIR low-resource: LameR best on 4/6 (Scifact 73.5, TREC-COVID 75.8, DBPedia 39.0, TREC-NEWS 50.3); HyDE wins ArguAna (46.6 vs 40.2) and FiQA (27.3 vs 25.8). *our-relevance:* answer-conditioning beat HyDE by ~8pp nDCG@10 on web search in 2023, so "conditioning generation on an answer attempt helps over plain HyDE" was already demonstrated at much larger effect sizes than SCOPE's +0.5–1.2pp Hit@5 (C7).
- **LameR ablations**: M=0 (no retrieved candidates, pure answer generation) is *already better than HyDE*; performance rises with M up to 10; demo-passage start-index sweep shows a U-shaped hard-negative curve; random passages from the whole collection work surprisingly well (they convey domain/format, not relevance). A 2nd-round LameR *drops* 1.0 nDCG@10 — "the query augmented by an LLM (in the 1st round) is prone to return spurious passages that especially confuse the LLM... resulting in wrong answers to poison BM25." *our-relevance:* the 2nd-round poisoning result is an early observation of expansion-induced [[query-drift]]/error reinforcement, consonant with our strong-query-regime failures (C6, related-work depth for C2).
- **LameR retriever-bottleneck argument**: self-supervised dense retrievers bottleneck LLM augmentation; BM25 "takes the outputs of LLMs in a transparent mode." *our-relevance:* retriever-dependence of expansion benefit supports our three-retriever generality analysis framing.

## Bearing on the review

- **C7 (snap-answer conditioning shows no measured benefit)**: this pair is the direct pre-emption surface. GAR made the generated answer the *default* expansion target in 2021; LameR made answer-conditioning zero-shot and beat HyDE by ~8pp nDCG@10 in 2023. A revised paper cannot present "condition the pseudo-doc on a private answer attempt" as novel per se. It must (a) cite both, (b) state precisely what differs (see Differentiation), and (c) run the significance-tested SCOPE-vs-HyDE comparison plus the a0-ablation (C12) — otherwise the one novel knob is both pre-empted and unmeasured.
- **C6 (uncited closest prior art)**: LameR and GAR belong in the same related-work paragraph as [[koblex-parser]]; the "generated answer/passage as retrieval scaffold" family is at least GAR→HyDE→LameR→ParSeR, and SCOPE must be positioned inside it, not beside it.
- **C4 (fabricated pseudo-doc content risk)**: GAR's explicit "beneficial rather than harmful overall" hallucination stance plus our geometry-vs-factuality result gives a two-source answer: the field has known since 2021 that expansion utility is not gated on factuality; our contribution is *measuring* that dissociation (AUC 0.79–0.94 geometry vs ~0.55 judged factuality).
- **C8 (no corpus-level distributional analysis)**: GAR's ROUGE query-passage overlap table is the template; a revision should report the same overlap statistics for raw question vs SCOPE pseudo-doc vs gold passage on BarExamQA/HousingQA.
- **C9 (weak baseline framing)**: GAR reports against BM25, RM3, and DPR and wins end-to-end; LameR reports against BM25, Contriever, HyDE, Q2D, and supervised DPR/ANCE. Both establish the norm of multi-baseline comparison that reviewers expected of us.
- **C5 (marginal gains)**: the honest contrast — LameR's answer-augmentation buys ~18pp nDCG@10 over BM25 on DL19; SCOPE's answer-conditioning buys +0.5–1.2pp Hit@5 over HyDE. The revision either finds a regime where conditioning matters measurably or drops the novelty claim and leads with the regime/mechanism story.

## Differentiation

Honest position: **SCOPE is not the first answer-conditioned expansion — it is a different disposal policy for the answer.** GAR conditions retrieval on the generated answer directly (appended to the question); LameR conditions on multiple sampled answers plus retrieved in-domain candidates (query repeated in the augmented string). SCOPE generates an answer a0, uses it only as private conditioning for a pseudo-document p, embeds *only p* (no raw question, no a0) for dense retrieval, and discards a0 before the answer call. So the residual novelty is narrow: (1) answer-as-latent-conditioning rather than answer-as-query-text, (2) discarding a0 as a confirmation-bias guardrail — asserted, never ablated (C12), and (3) dense+CE-rerank in a legal corpus rather than BM25. Point (1) directly contradicts GAR's Sec 3.3 warning against retrieving without the question, which is a defensible design argument only in the weak-query regime where the raw question is lexically useless (BarExamQA raw Hit@5 1.4%) — that regime-conditional reversal of GAR's advice is our most defensible framing, and it connects to [[weak-vs-strong-query-regime]] and [[regime-routing]]. Where we are simply pre-empted: answer generation improving retrieval (GAR), answer-conditioning beating HyDE (LameR), and tolerance of hallucinated expansions (GAR). Neither paper touches legal corpora, per-query mechanism prediction ([[qpp]], CE-affinity geometry), or the retrieval→answer conversion failure — those remain ours.

## Links

- [[scope]], [[hyde]], [[query2doc]] — the generated-expansion family this pair anchors: [[generated-query-family]]
- [[vocabulary-gap]] — GAR's ROUGE overlap table is the earliest quantification we have on file
- [[weak-vs-strong-query-regime]], [[regime-routing]] — where dropping the raw question (contra GAR) is defensible
- [[query-drift]] — LameR's 2nd-round error-reinforcement drop; see [[weller-drift]]
- [[geometry-vs-factuality]], [[answer-conversion-gap]] — our mechanism results that neither paper has
- [[qpp]] — per-query prediction absent from both
- [[koblex-parser]], [[gure]] — the other C6/C8 prior-art pages
- [[legal-rag-benchmarks]], [[icml-ai4law-2026-rejection]], [[scope-paper-2026]]

## Raw source

- `references/lamer.pdf` (arXiv 2304.14233v2, 2 Aug 2023, 15 pp; pages 1–10 read: abstract, related work, observations, method, main tables 2–6, ablations, limitations)
- `references/gar.pdf` (arXiv 2009.08553v4, 6 Aug 2021, ACL 2021, 12 pp; all substantive pages read including Appendix A)
