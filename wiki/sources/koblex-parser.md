---
title: KoBLEX / ParSeR (EMNLP 2025)
type: source
tags: [legal-rag, generated-queries, multi-hop, korean-law, benchmarks, hyde-family]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://aclanthology.org/2025.emnlp-main.200.pdf
local: references/koblex-parser.pdf
authors: Lee et al. (Jihyung Lee, Daehui Kim, Seonjeong Hwang, Hyounghun Kim, Gary Geunbae Lee; POSTECH/KT)
year: 2025
venue: EMNLP 2025 (Main, pp. 4019-4053)
code: https://github.com/daehuikim/KoBLEX
---

# KoBLEX: Open Legal Question Answering with Multi-hop Reasoning (introduces ParSeR)

## TL;DR

Introduces KoBLEX, a 226-instance bilingual (Korean + English-translated) benchmark of provision-grounded, multi-hop (1-3 hop), open-ended legal QA over Korean statutes, built via GPT-4o generation plus multi-stage LLM and human-expert validation. Proposes ParSeR: the LLM generates a *list* of "parametric provisions" — statute-style clauses written from parametric knowledge, explicitly "as if part of the relevant law" — and each one is used as a query in a three-stage Retrieve (BM25, top-100) -> Rerank (Korean-finetuned BGE reranker, top-10) -> LLM Selection (pick exactly 1) pipeline; selected provisions then ground a final answer call. Also proposes LF-Eval, a G-Eval-style GPT-4o judge for legal fidelity (Pearson 84.90 with human ratings). This is the uncited near-twin reviewer oSUu raised as C6.

## Key claims / numbers

- ParSeR parametric provision generation prompt (Fig. 19): given {background}+{question}, "identify and return all relevant statutory provisions that support the answer... Your role is not to provide interpretations, summaries, or conclusions... If no directly applicable statutory provision exists, generate the most plausible clause in the same format, as if it were part of the relevant law." Output = a JSON-parsed list of clauses, one per reasoning component. *our-relevance:* this is the exact "pseudo-statute as retrieval query" move of SCOPE's pseudo-document p, confirming C6/C3; but there is **no explicit intermediate answer/draft step** — no analogue of SCOPE's snap answer a0 or its discard guardrail (C12).
- Three-stage retrieval: BM25 top-k=100 per parametric provision -> finetuned reranker (dragonkue/bge-reranker-v2-m3-ko) to top-l=10 -> LLM selects exactly one provision ID. One supporting provision per parametric provision. *our-relevance:* structurally richer than SCOPE's single embed+CE-rerank top-5; the Selection stage is the piece we lack.
- Main results (Table 2, GPT-4o): provision retrieval F-1 59.41 vs one-time retrieval (SP+OR) 21.50 (+37.91), EM 26.99 vs 7.08 (+19.91); generation Token F-1 46.14 vs 26.75 (+19.39), LF-Eval 67.26 vs 36.45 (+30.81). Beats strongest baseline ProbTree by +12.23 Token F-1 / +14.64 LF-Eval. *our-relevance:* their retrieval lift **converts into large answer-quality gains**, unlike SCOPE's 8x Hit@5 lift moving answers 72.3->72.9 — directly sharpens C5/C9 and our [[answer-conversion-gap]].
- Even vs no-retrieval SP (Token F-1 36.20, LF-Eval 55.00, GPT-4o), ParSeR gains ~+10 Token F-1 / +12 LF-Eval. *our-relevance:* on KoBLEX the correct comparator (no-retrieval) is still beaten decisively; our headline was framed against the weakest baseline (C9).
- Section 7: baselines with low provision F-1/EM score *below* the no-retrieval baseline on generation metrics — "retrieving irrelevant provisions negatively impacts multi-hop legal reasoning." *our-relevance:* published precedent for our HousingQA regression / [[query-drift]] dilution story (C10).
- Ablation (Table 3, EXAONE-3.5-32B): full 48.74 F-1 / 57.58 LF-Eval; w/o Selection 40.61/50.18; w/o Reranking 40.64/54.02; w/o both 27.56/45.97; w/o parametric provisions too (plain top-k on the raw question) 21.41/45.52. *our-relevance:* per-component ablation discipline the reviewers demanded of us (C7, C11, C12); note generated-query gain alone (27.56 vs 21.41) is smaller than the Selection+Rerank machinery on top.
- Retriever swap (Table 4): sparse 48.74 / dense 50.43 / hybrid 48.16 F-1 — method robust across retrievers; they adopt BM25. k=100, l=10 chosen via sweeps (Tables 5-6). Efficiency: highest LF-Eval at the fewest generated tokens (Fig. 7). *our-relevance:* mirrors our three-retriever generality result; their token-efficiency claim covers the full pipeline, unlike our criticized accounting (C11).
- Benchmark: 226 instances (55/125/46 at 1/2/3 hops) distilled from 3,035 GPT-4o drafts (7% survival) via Partial/Full LLM checks, revision by Korean law graduates, and 3-expert Likert scoring (>=96% pairwise agreement); corpus = 608 statutes / ~233,544 paragraph-level provisions cited in Korean court decisions 1998-2024; CC BY-NC 4.0. *our-relevance:* the expert-in-the-loop construction and LF-Eval human-correlation study are exactly the legal-NLP grounding reviewers said we lack (C1, C2, C8); relevant to [[expert-judgment-replication]] and [[legal-rag-benchmarks]].
- Limitations section: civil-law (statute-first) focus; authors flag possible non-transfer to common-law jurisdictions (US/UK) where precedent dominates. No significance tests or confidence intervals anywhere in their tables either. *our-relevance:* our US common-law benchmarks are a genuinely different regime, and their rigor bar on CIs is no higher than ours.

## Bearing on the review

- **C6 (uncited near-twin) — confirmed, must cite.** The reviewer's mapping is accurate at the core: parametric provisions are LLM-generated statute-style text used purely as intermediate retrieval queries, with a rerank pipeline, in legal QA, published at EMNLP 2025 main before our submission. A revision must cite KoBLEX/ParSeR as closest prior art and position SCOPE inside the [[generated-query-family]] (with [[hyde]]).
- **C3 (just HyDE-in-legal):** ParSeR shows the space already moved past plain HyDE-in-legal (multi-provision decomposition + selection). Claiming the base move as novel is untenable; novelty must shift to what ParSeR does not do (below).
- **C7/C12:** ParSeR's clean per-stage ablation is the template. A revision must ablate snap-answer conditioning vs a no-answer provision-list generation (ParSeR-style) and vs passing a0 to the answer call — or drop the guardrail claim.
- **C9/C5:** their gains hold against the no-retrieval comparator; ours do not. Any revision must foreground LLM-only as the primary baseline and explain why open-ended provision-grounded answering converts retrieval to answers while US MC/yes-no formats let parametric knowledge bypass retrieval.
- **C10:** their "irrelevant retrieval is worse than no retrieval" observation is citable precedent for framing HousingQA as retrieval-dilution rather than "parity."

## Differentiation

Honest adjudication: **we are pre-empted on the core mechanism** — generating hypothetical statutory text from parametric knowledge as a legal retrieval query with reranking. Real differences that survive: (1) SCOPE's explicit snap answer a0 conditioning the pseudo-document, plus discarding a0 at answer time — ParSeR has no externalized answer step and no guardrail; but per C7 we never showed this component helps, so it differentiates without (yet) adding value. (2) Single pseudo-document vs ParSeR's N-provision multi-hop decomposition with an LLM Selection stage — theirs is strictly more machinery and their ablation shows Selection matters most after provision generation. (3) Setting: Korean civil-law, 226 open-ended free-form questions with provision-set gold and LLM-judge scoring, vs our US common-law-adjacent MC/yes-no at N=1195/6853 with exact passage ids — their own Limitations concede possible non-transfer to common-law jurisdictions. (4) What ParSeR entirely lacks and where our post-rejection work lives: per-query mechanism ([[geometry-vs-factuality]], CE-affinity movement), [[qpp]]-based analysis, the [[weak-vs-strong-query-regime]] contrast and [[regime-routing]], failure-mode characterization on strong queries, and cross-domain replication on BEIR/MedQA/MuSiQue. ParSeR reports that its method works; it never asks when or why generated provisions help or hurt. That analysis layer — not the generation trick — is our defensible contribution ([[vocabulary-gap]], [[scope]]).

## Links

[[scope]], [[hyde]], [[generated-query-family]], [[vocabulary-gap]], [[weak-vs-strong-query-regime]], [[query-drift]], [[qpp]], [[answer-conversion-gap]], [[geometry-vs-factuality]], [[regime-routing]], [[legal-rag-benchmarks]], [[expert-judgment-replication]], [[icml-ai4law-2026-rejection]]

## Raw source

- references/koblex-parser.pdf (ACL Anthology 2025.emnlp-main.200, 35 pp., read in full: body pp. 1-10, appendices A-J incl. all prompt figures)
