---
title: Hypothetical Documents or Knowledge Leakage? (ACL Findings 2025)
type: source
tags: [query-expansion, hyde, knowledge-leakage, evaluation-validity, fact-verification]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2504.14175
local: references/yoon2025leakage.pdf
authors: Yoon et al.
year: 2025
venue: ACL 2025 Findings
code: none
---

# Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion

## TL;DR
Yoon, Jung, Yoon & Park (Soongsil / MAUM AI / Adobe) ask whether HyDE/Query2doc gains come from generating genuinely *hypothetical* documents or from the LLM **reproducing benchmark gold evidence it saw in pretraining** ("knowledge leakage"). Using fact verification (FEVER, SciFact, AVeriTeC) with 7 LLMs, they NLI-match generated documents against gold evidence and find that QE gains are concentrated almost entirely on claims whose generated documents contain sentences *entailed by gold evidence*; on unmatched claims, expanded queries usually do **worse than the no-expansion baseline**. This is the sharpest published attack on the evidential value of HyDE-family results on public benchmarks.

## Key claims / numbers
- Query2doc (BM25, n=5 query copies) and HyDE (Contriever, N=1 generated doc, k=5) both significantly beat their baselines on all three benchmarks across 7 LLMs (p<0.001); e.g., FEVER Recall@5 BM25 31 -> Query2doc ~35-38; Contriever 26.8 -> HyDE ~35-40. *Our-relevance:* confirms the generated-query family (SCOPE included) reliably lifts retrieval metrics — the effect SCOPE reports on BarExamQA is normal for the family (C3, C5).
- NLI-based matching finds 27.6%-83.5% of expanded queries contain sentences entailed by gold evidence (max: FEVER + HyDE + GPT-4o-mini at 83.5%; min: SciFact + HyDE + Gemini-1.5-flash at 27.6%). *Our-relevance:* our pseudo-documents very likely also entail gold passages on a large fraction of BarExamQA rows; we have never measured this (C7, C9).
- On *matched* claims, QE performance is significantly higher than on all claims (p<0.001); on *unmatched* claims it typically falls **below** BM25/Contriever without expansion (e.g., FEVER HyDE Recall@5: matched 40.0 vs unmatched 23.4, baseline 26.8). *Our-relevance:* this is exactly our query-drift / weak-vs-strong crossover, but attributed to leakage rather than geometry — a direct rival explanation to our CE-affinity mechanism (C7, C9).
- Conclusion: "performance improvements from query expansion were consistent only when LLM-generated documents contained sentences entailed by gold evidence," so leakage "may be present... potentially inflating the perceived performance of LLM-based query expansion." *Our-relevance:* a reviewer armed with this paper can claim SCOPE's 8x BarExamQA retrieval lift is a memorization artifact of Llama/Gemma having seen bar-exam materials (C9, C4).
- Limitations they concede: no causal link to specific training data; NLI matcher imperfect (manually spot-checked); scope limited to fact verification. *Our-relevance:* leaves room for us to run the *same* NLI audit on legal corpora where leakage is less plausible (post-1970s state statutes, HousingQA) and turn the critique into an asset.

## Bearing on the review
- **C9 (parametric knowledge drives answers)**: this paper is the retrieval-side twin of C9. A revision must add a leakage audit: NLI-match SCOPE pseudo-documents against gold passages, and report SCOPE's Hit@5 lift separately for matched vs unmatched rows. If the lift survives on unmatched rows, we have a stronger claim than the original paper.
- **C7 (snap-conditioning shows no benefit, no significance tests)**: their matched/unmatched split with significance testing is the template for the missing SCOPE-vs-HyDE statistics.
- **C4 (fabricated legal content risk)**: connects fabrication to *evaluation validity*, not just user harm — our geometry-vs-factuality result (AUC 0.79-0.94 vs 0.55-0.58) is a direct empirical answer and should be cited against this frame.

## Differentiation
We are partially pre-empted: they published (mid-2025, before our submission) the core skepticism that generated-document gains are confounded by what the LLM already knows, on general benchmarks. We were not scooped on mechanism: they stop at an entailment correlation and explicitly decline a causal or geometric account, whereas our CE-affinity-movement mechanism (Spearman ~0.44, 3 retrievers, 5 BEIR datasets) explains *when* expansion helps including on rows without verbatim leakage, and our factuality falsification shows geometry, not judged truthfulness, predicts failure. Honest statement: SCOPE's BarExamQA lift is untested against their leakage audit, and until we run it, their explanation and ours are observationally confounded on our legal benchmarks.

## Links
[[scope]], [[hyde]], [[generated-query-family]], [[query-drift]], [[geometry-vs-factuality]], [[weak-vs-strong-query-regime]], [[vocabulary-gap]], [[icml-ai4law-2026-rejection]]; siblings: [[li2026legalmalr]], [[afane2026laborbench]]

## Raw source
references/yoon2025leakage.pdf (arXiv:2504.14175v2, read pages 1-9 including all main tables)
