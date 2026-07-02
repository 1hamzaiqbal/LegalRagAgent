---
title: GuRE Generative Query Rewriter for Legal Passage Retrieval (NLLP 2025)
type: source
tags: [legal-ir, query-rewriting, vocabulary-gap, trained-rewriter, long-tail, retrieval]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://aclanthology.org/2025.nllp-1.31.pdf
local: references/gure.pdf
authors: Kim et al. (Daehui Kim, Deokhyung Kang, Jonghwi Kim, Sangwon Ryu, Gary Geunbae Lee; POSTECH/KT)
year: 2025
venue: NLLP Workshop 2025 (pp. 424-438)
code: https://github.com/daehuikim/GuRE
---

# GuRE: Generative Query REwriter for Legal Passage Retrieval

## TL;DR

GuRE fine-tunes an LLM (SaulLM-7B, LoRA r=64, plain cross-entropy SFT) to generate the cited legal passage from its preceding drafting context, then uses that generated passage as a *replacement query* for any retriever. On LePaRD (U.S. federal court precedent citations; 10K/20K/50K passage pools), GuRE is retriever-agnostic and massively beats zero-shot expansion (Query2Doc, Q2D-CoT) *and* direct retriever fine-tuning — e.g. BM25 nDCG@10 jumps 15.33 -> 47.69. The paper's analytical spine is a corpus-level long-tail citation-frequency analysis: retriever fine-tuning with in-batch negatives degrades on frequently-cited passages while GuRE improves on them.

## Key claims / numbers

- **Task and framing**: Legal Passage Retrieval (LPR) = retrieve the target passage p_q given ongoing drafting context q; the stated core obstacle is "significant vocabulary mismatch between the query and the target passage" (Fig. 1). *our-relevance:* this is verbatim our vocabulary-gap motivation ([[vocabulary-gap]]), published in the legal-NLP venue the reviewers say we ignored (C8, C2).
- **Method**: SFT the LLM on InstructionPrompt(q, p_q) pairs to auto-regressively generate the passage; at inference the generated passage *replaces* the query (rewriting, not Q2D-style concatenation). *our-relevance:* GuRE is the trained sibling of the [[generated-query-family]]; SCOPE/HyDE are its zero-shot cousins, and C3's "essentially HyDE" charge sits exactly on this axis.
- **Benchmark**: LePaRD (Mahari et al. 2024), 10K/20K/50K target-passage pools = 1.92M/2.48M/3.50M data points; 90% train, test = 3x10,000 sampled from remaining 10%, averaged. Authors call it "the only publicly available LPR dataset to our knowledge" (Limitations). *our-relevance:* a legal retrieval corpus we never engage with (C2, C8); note its query is *drafting context preceding a citation*, not a bar-exam question, so the regime differs from BarExamQA.
- **Headline numbers (10K pool, R@1 / R@10 / nDCG@10)**: BM25 9.91/28.19/15.33 -> BM25+GuRE 34.88/62.20/47.69; DPR 1.99/5.49/3.39 -> DPR+GuRE 32.07/49.74/40.68 (vs DPR-FT 14.09/50.97/30.31); ModernBERT 7.11/22.47/13.94 -> +GuRE 33.14/60.24/45.86 (vs ModernBERT-FT 14.12/51.34/30.50). Gains marked significant by paired t-test p<0.01. *our-relevance:* trained rewriting delivers 3x-30x lifts with significance tests — dwarfing our unsigned +0.5-1.2pp SCOPE-vs-HyDE Hit@5 deltas (C5, C7).
- **Zero-shot expansion underperforms**: BM25+Q2D 21.15 and BM25+Q2D-CoT 22.22 nDCG@10 vs GuRE 47.69 (10K); paper: "few-shot prompting strategy struggles to address the underlying challenges in tasks requiring domain-specific knowledge." *our-relevance:* direct evidence that the zero-shot HyDE-family tier SCOPE lives in is the *weak* tier on legal retrieval when supervision is available (C3, C8).
- **Mechanism = lexical/semantic affinity to gold**: generated-text vs target-passage overlap (10K test): raw query BLEU 5.75 / ROUGE-L 18.98 / BERTScore-F 75.61; Q2D 8.56/19.19/78.6; Q2D-CoT 11.86/27.28/80.1; **GuRE 59.43/67.62/90.92** at matched length (50.9 vs 50.2 words). *our-relevance:* independent support for our CE-affinity-movement mechanism ([[scope]] gain ~ movement toward gold, Spearman ~0.44) — the trained rewriter simply buys far more affinity movement than zero-shot generation ([[geometry-vs-factuality]]).
- **Corpus-level long-tail analysis (the C8 item)**: top 1% of passages account for 16.23-16.86% of citations depending on pool (Table 5; text cites "top 1% ... 18% of all citations, while 64% receive only one citation" from Mahari et al.); Figure 4 plots the log-scale long-tail. Section 5.3 sorts test candidates by training-set frequency and shows GuRE beats fine-tuned retrievers at *every* threshold, with opposite trends: GuRE improves on frequent passages, retriever-FT degrades on them, because MNRL in-batch negatives punish duplicated frequent positives (Appendix H; contrastive-loss variant collapses to R@1 0.1). *our-relevance:* this is precisely the "long-tail passage frequency" distributional analysis C8 says we lack; we have no analogous frequency- or jurisdiction-stratified breakdown of BarExamQA/HousingQA corpora.
- **Data efficiency**: GuRE trained on only 10K examples already beats retriever fine-tuning trained on millions (Table 4: ModernBERT 13.94 -> GuRE-10K 26.58 -> GuRE-100K 32.31 nDCG@10, 10K pool). *our-relevance:* weakens any "training data is prohibitive" defense of zero-shot-only baselines (C8).
- **Cost/availability**: LoRA SFT 60/100/130 GPU hours (10K/20K/50K pools) on RTX A6000-class; inference ~10-12 min per 10,000 rewrites on one RTX 3090 via vLLM (<0.1 s/query); code public. Backbone matters: SaulLM-7B (legal-pretrained) 45.86 nDCG@10 vs Qwen2.5-7B 38.08 vs Llama3.1-8B 34.47. *our-relevance:* running GuRE as a baseline is feasible, not hypothetical (C8, C11).
- **Hallucination/distractor note**: case studies mark Q2D/Q2D-CoT pseudo-passage content that is "mostly irrelevant" or distractor-laden ("defendant's intent in adopting its mark") and say expanded queries "may incur hallucination problems." *our-relevance:* their qualitative distractor framing parallels our [[query-drift]] finding, though they never quantify factuality — our geometry-vs-factuality falsification goes further here (C4).

## Bearing on the review

- **C8 (directly named miss)**: a revised paper must cite GuRE, position SCOPE explicitly against the trained-rewriter tier, and add a corpus-level distributional analysis of our own corpora (passage-frequency long-tail for the barexam/housing collections; jurisdiction breakdown for HousingQA) in the style of their Table 5 / Figure 4 / Section 5.3 frequency-threshold curves.
- **C2/C3**: GuRE plus [[koblex-parser]] define the legal-NLP prior-art frame reviewers expected. SCOPE must be introduced as a *zero-shot, training-free* member of the generated-query family with GuRE as the supervised reference point, not as a novel legal method.
- **C7/C5**: GuRE reports paired t-tests (p<0.01) on every headline comparison and its lifts are order-of-magnitude. Our unsigned +0.5-1.2pp SCOPE-vs-HyDE deltas look untenable next to this; any revision needs significance tests and a candid effect-size discussion.
- **C11**: their reporting discipline (3x10K test resamples averaged, 99% CIs in Figures 3/7-9, GPU-hour and API-cost accounting including the $52.83 Q2D bill) is the rigor bar the reviewers implicitly applied to us, including first-stage generation cost accounting.

## Differentiation

- **Trained vs zero-shot**: GuRE requires (context, cited-passage) supervision and ~60-130 GPU hours of SFT per corpus; SCOPE is training-free and uses the same off-the-shelf answerer model as rewriter. This is a real setting difference, but it cuts against us on raw performance: on their benchmark, zero-shot generation (Q2D, the closest analog to SCOPE/HyDE in their tables) is decisively worse than training. We are not pre-empted on the zero-shot regime itself, but "zero-shot" alone is a weak differentiator unless we show settings where training data is genuinely unavailable.
- **Query regime**: LePaRD queries are drafting context immediately preceding a citation — already legal prose, arguably a *strong-query* surface with residual vocabulary mismatch. BarExamQA's colloquial fact patterns are a different, weaker query regime ([[weak-vs-strong-query-regime]]). GuRE never tests question-style queries, and we never test citation-context queries; a fair head-to-head needs one shared corpus.
- **What we have that GuRE lacks**: a per-query *mechanism* (CE-affinity movement, replicated across 5 BEIR datasets and 3 retrievers), a falsification of the hallucination story, [[qpp]]-based [[regime-routing]], and downstream *answer* accuracy measurement (GuRE is retrieval-only and never touches the [[answer-conversion-gap]]).
- **Fair-comparison note**: comparing SCOPE against GuRE head-to-head is only fair if labeled as zero-shot-vs-supervised; the honest experiment is to run GuRE-style SFT on BarExamQA gold (question, gold-passage) pairs as a supervised skyline next to SCOPE/HyDE/raw, using their public repo (github.com/daehuikim/GuRE; SaulLM-7B or our existing backbones; tens of GPU hours). The open question is training-pair volume: LePaRD supplies millions of pairs, BarExamQA supplies far fewer, so the data-scarce result (10K pairs suffice) is what makes this feasible at all.

## Links

- [[scope]] / [[scope-paper-2026]] — our method; GuRE is C8's named prior art for its motivation.
- [[hyde]] — zero-shot ancestor; GuRE's Q2D baselines proxy this tier.
- [[koblex-parser]] — the other reviewer-named legal prior art (C6).
- [[vocabulary-gap]] — identical stated motivation, with supervision as the fix.
- [[generated-query-family]] — GuRE = trained member; SCOPE/HyDE/Q2D = zero-shot members.
- [[weak-vs-strong-query-regime]] — LePaRD citation-context queries vs BarExamQA fact patterns.
- [[query-drift]] — their distractor case studies are qualitative drift observations.
- [[geometry-vs-factuality]] — their BLEU/BERTScore-to-gold table is affinity-movement evidence.
- [[qpp]], [[regime-routing]], [[answer-conversion-gap]] — our contributions absent from GuRE.
- [[legal-rag-benchmarks]] — LePaRD should be added to the benchmark map.
- [[icml-ai4law-2026-rejection]] — grounds C8 (and touches C2, C3, C5, C7, C11).

## Raw source

- `references/gure.pdf` (ACL Anthology 2025.nllp-1.31, 13 pages incl. appendices; all pages read).
