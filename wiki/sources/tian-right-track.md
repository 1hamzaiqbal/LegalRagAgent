---
title: Am I on the Right Track? QPP for Agentic RAG (IR-RAG @ SIGIR 2025)
type: source
tags: [qpp, agentic-rag, rag, query-quality, retrieval-evaluation]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2507.10411
local: references/tian-right-track.pdf
authors: Tian et al. (Fangzheng Tian, Jinyuan Fang, Debasis Ganguly, Zaiqiao Meng, Craig Macdonald; U. Glasgow)
year: 2025
venue: IR-RAG Workshop @ SIGIR 2025 (arXiv 2507.10411)
code: none (built on PyTerrier / PyTerrier-RAG, github.com/terrierteam/pyterrier_rag)
---

# Tian et al. 2025 — Am I on the Right Track? What Can Predicted Query Performance Tell Us about the Search Behaviour of Agentic RAG

## TL;DR
Applies unsupervised post-retrieval QPP (NQC, Max(Score), Dense-QPP, A-Pair-Ratio) to the *LLM-generated intermediate queries* of two RL-trained agentic RAG models (Search-R1, R1-Searcher; both QWEN2.5-7B) on the 3,610-question NQ test set with three retrievers (BM25, BM25»MonoT5, E5). Findings: better retrievers shorten the reasoning loop and raise answer quality; per-query QPP of the *first* generated query correlates positively but weakly with final answer F1 (Spearman's ρ up to 0.25, far below the ρ>0.5 QPP achieves on standard ad-hoc retrieval); QPP estimates of generated queries decline over reasoning iterations. Adaptive QPP-gated retrieval is proposed only as future work — no routing intervention is run.

## Key claims / numbers
- **Setup**: Search-R1 and R1-Searcher (RL-trained QWEN2.5-7B agentic RAG), NQ test set (3,610 questions), 2018 Wikipedia corpus, top-3 documents fed per reasoning-retrieval iteration; retrievers BM25, BM25 top-20 » MonoT5 cross-encoder rerank, and E5 dense; all in PyTerrier-RAG. *our-relevance:* this is the near-twin experimental frame for our QPP-of-generated-queries line; unlike SCOPE it QPPs the generated query, and unlike our routing it never intervenes (C6-adjacent novelty check for the revised related-work section, C2/C8 literature coverage).
- **RQ1 — retriever effectiveness vs reasoning length**: Search-R1 EM/F1/Iter = 0.3391/0.4185/2.52 (BM25), 0.3873/0.4709/2.19 (MonoT5), 0.4838/0.5687/2.00 (E5); R1-Searcher EM = 0.2089/0.2307/0.3075. Spearman ρ(Iter, F1) is negative: −0.28 to −0.32 for Search-R1, −0.11 to −0.19 for R1-Searcher. *our-relevance:* independent evidence that retrieval quality shapes downstream behavior yet answer quality is only moderately coupled to it — consonant with our [[answer-conversion-gap]] and with C9's point that answers are substantially parametric.
- **RQ2 — QPP across iterations**: average QPP of generated queries decreases over iterations 1–5 for nearly all predictor/retriever combos (exception: NQC on MonoT5, attributed to NQC's score-distribution assumptions failing for neural rerankers). *our-relevance:* their appendix ties this decline to ambiguous questions producing *drifted sub-queries* — the same [[query-drift]] failure family we invoke for SCOPE's strong-query regressions (C10).
- **RQ3 — QPP-answer link (the headline for us)**: Spearman ρ between first-generated-query QPP and final answer F1 is *positive, statistically significant, but weak*: Search-R1 — NQC .1297 (BM25), .0524 (MonoT5), .2394 (E5); Max(Score) .2383 (MonoT5), .2369 (E5); Dense-QPP .1871 (E5); A-Pair-Ratio .2497 (E5, best overall). R1-Searcher is weaker still — NQC .1205/.0096/.0515; Max(Score) .1735/.0919; Dense-QPP .0834; A-Pair-Ratio .0297. Paper explicitly notes these are "less than what has been achievable for standard retrieval tasks (which can be as high as Spearman's ρ > 0.5)". *our-relevance:* their best no-gold QPP→answer correlation (ρ≈0.25) matches our finding that no-gold per-query QPP is weak (our best WIG-CE τ≈−0.11 for routing), while our *gold-anchored* CE-affinity mechanism reaches ρ≈0.44 — a legitimate contrast, but only because ours uses gold labels (diagnosis, not deployable routing). Also a template for fixing C7/C11: they significance-test every reported correlation.
- **Appendix case study**: ambiguous input queries cause topic drift in intermediate queries and wrong answers (Max(Score) 0.84–0.85 vs >0.9 for clean cases); 262/3,610 queries generated *identical repeated sub-queries* under E5 with a 3-iteration cutoff. *our-relevance:* generated-query degeneracy is a concrete failure mode of the [[generated-query-family]] that a revised SCOPE paper can cite when motivating why expansion must be gated (C4's fabricated-content worry has a retrieval-side analogue: drifted/repeated queries, not just fabricated content).
- **Future-work framing**: QPP could "adaptively decide whether the retrieved results are useful enough to show to the reasoning model" or serve as an RL reward signal — proposed, not implemented. *our-relevance:* our [[regime-routing]] (vanilla SCOPE on weak-query regimes, raw∪SCOPE pooling on strong) is an actual instantiation of exactly the adaptive step they leave open; this is the differentiation sentence for the revision.

## Bearing on the review
- **C2 / C8 (insufficient literature grounding)**: this paper must be cited in a revised related-work section as the closest QPP-for-generated-queries prior. Our RELATED_WORK_GROUNDING already tags it as near-twin #2; the wiki page pins the exact numbers so the citation is substantive, not decorative.
- **C7 / C11 (no significance tests, rigor gaps)**: Tian et al. report a full predictor × retriever × model correlation matrix and state that all correlations are significance-tested. A revised SCOPE paper needs at minimum this standard for SCOPE-vs-HyDE Hit@5 deltas and for any QPP/mechanism correlation we report.
- **C9 (answers driven by parametric knowledge)**: their weak QPP→F1 link (ρ ≤ 0.25 even with a good retriever) is external corroboration that retrieval quality only weakly converts into answer quality in RAG — we can cite it when honestly framing the 8x-retrieval-lift / +0.6pp-answer result as an instance of a known retrieval-generation gap rather than a defect unique to SCOPE.
- **C10 (query drift on strong queries)**: their iteration-wise QPP decline plus the ambiguity-drift case study give an agentic-RAG analogue of query drift; useful support for the regime story that expansion/generation degrades already-adequate queries.

## Differentiation
Honest position: Tian et al. *pre-empt the general idea* that unsupervised QPP estimates of LLM-generated queries carry signal about final RAG answer quality — we cannot claim that observation as novel. Differences that survive:
1. **What gets QPP'd**: they score the *generated* query after retrieval; our routing question scores the *raw* query (pre-expansion) to decide *whether to generate at all*. Theirs is post-hoc monitoring of an agentic loop; ours is an expansion-gating decision for a fixed two-call method.
2. **Correlational vs interventional**: they explicitly stop at correlation and leave adaptive QPP-based retrieval as future work; our regime-routing result (avoid ~14% of Housing dilution while keeping BarExam wins) is an intervention, though ours is regime/dataset-level, not per-query — because we found the same weakness of per-query no-gold QPP that their ρ≤0.25 implies.
3. **Mechanism**: our gold-anchored CE-affinity movement (ρ≈0.44 pooled, ≥0.30 on 7 datasets × 3 retrievers) is a *diagnostic* mechanism claim, stronger than their no-gold correlations but not comparable as a deployable predictor; the revision must keep this distinction explicit to avoid overclaiming against them.
4. **Domain/regime**: they study NQ only (single dataset, self-acknowledged limitation) with agentic multi-turn models; we study weak-vs-strong query regimes across legal/medical/multi-hop corpora with a fixed-cost two-call method. Their setting has no analogue of the weak-query vocabulary-gap regime where SCOPE's retrieval lift lives.

Where we are pre-empted: "QPP of generated queries correlates with RAG answer quality" is theirs (July 2025). Our novelty claim must be narrowed to: raw-query QPP for expansion gating + regime routing as the intervention + the gold-affinity geometric mechanism and its factuality falsification.

## Links
- [[qpp]] — this paper is the canonical agentic-RAG QPP application; NQC/Max(Score)/Dense-QPP/A-Pair-Ratio inventory.
- [[query-drift]] — appendix drift case study; iteration-wise QPP decline.
- [[answer-conversion-gap]] — weak QPP→F1 correlations as external evidence of the retrieval-generation gap.
- [[regime-routing]] — we implement the adaptive step they propose as future work.
- [[generated-query-family]] — SCOPE/HyDE and agentic sub-queries are siblings; query repetition/degeneracy failure mode.
- [[scope]], [[hyde]], [[weak-vs-strong-query-regime]], [[geometry-vs-factuality]], [[icml-ai4law-2026-rejection]]
- Sibling sources: [[gure]] (trained legal query rewriter, C8), [[koblex-parser]] (uncited near-twin #1, C6), [[hyde]], [[query2doc]], [[scope-paper-2026]].

## Raw source
- `references/tian-right-track.pdf` (arXiv 2507.10411v1, 13 pp., read in full including appendix)
