---
title: LegalMALR Multi-Agent Statute Retrieval (arXiv 2026)
type: source
tags: [legal-retrieval, query-reformulation, multi-agent, statute-retrieval, grpo, reranking]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2601.17692
local: references/li2026legalmalr.pdf
authors: Yunhan Li et al.
year: 2026
venue: arXiv preprint (cs.IR, 25 Jan 2026)
code: none
---

# LegalMALR: Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval

## TL;DR
LegalMALR (City U. Macau / SIAT-CAS / SUSTech) targets exactly our weak-query regime: real legal queries are "implicit, multi-issue, and expressed in colloquial or underspecified forms," so single-shot dense retrieval on the raw query misses statutory elements. Their fix is a Multi-Agent query understanding System (six specialised Qwen3-4B rewrite agents + planner, iterating up to 4 rounds), stabilised with GRPO reinforcement learning, feeding a zero-shot Qwen-Max LLM reranker. On Chinese statute retrieval (STARD test, 309 queries) it reaches Recall@10 0.8195 / MRR@10 0.7367, +6.16/+6.31pp over a matched-budget Qwen3 RAG baseline, and generalises out-of-distribution to their new 118-query CSAID dataset (MRR@10 0.9161, Recall@10 0.6841).

## Key claims / numbers
- Problem framing: dense retrievers "focus primarily on the literal surface form of the query" while colloquial queries omit key legal elements — verbatim the vocabulary-mismatch / weak-query diagnosis. *Our-relevance:* independent 2026 confirmation of the regime SCOPE targets on BarExamQA; must be engaged or we look unaware of the closest contemporaneous work (C2, C8).
- STARD (55,348-article corpus): BM25 Recall@10 0.3943 vs RAG baselines ~0.72-0.77 vs LegalMALR 0.8195; also beats a domain-finetuned Qwen3-SFT pipeline (0.7690). *Our-relevance:* shows both that raw-query dense RAG is already strong on layperson statute queries and that *trained/RL* query understanding beats prompt-only rewriting — a stronger baseline family SCOPE was never compared against (C3, C6).
- CSAID (harder: avg 7.16 relevant statutes/query, 79,055 articles, built partly from documented retrieval failures): LegalMALR +4.41 MRR@10 / +8.09 Recall@10 over the RAG baseline, zero-shot cross-domain. *Our-relevance:* a public hard weak-query legal set we could adopt to answer "legal-only, English-only" scope criticisms (C2, C8).
- Instability motivation: across 8 MAS rollouts per query, average max recall 0.8725 vs mean 0.8098 vs min 0.7511; 186+78 of 1,234 queries are "occasionally/never fully correct but unstable." GRPO (LoRA on frozen Qwen3-4B, ~16h on 8x RTX4090, step penalty -0.05, terminal reward = MAS recall) is introduced specifically to tame stochastic LLM rewrites. *Our-relevance:* they quantify the variance of LLM-generated reformulations that our per-query QPP/geometry analysis also observes; their solution is RL, ours is routing (C7, [[regime-routing]]).
- Budget fairness: MAS uses on average 60.3 embedding calls vs the fixed 60-call baseline (avg 2.01 retrieval rounds, 13.58-statute merged pool), so gains are not from extra retrieval budget. *Our-relevance:* a model for the token/cost accounting reviewers said we omitted (C11).
- Reranker note: explicit CoT rationales in the LLM reranker gave "modest improvements" but were dropped for cost/format-stability; the no-CoT config is used throughout. *Our-relevance:* mirrors our answer-stage findings that extra generated reasoning does not reliably convert retrieval gains (answer-conversion gap).

## Bearing on the review
- **C2/C6 (missing closest prior art)**: LegalMALR is now among the nearest neighbours of SCOPE in legal query generation (with KoBLEX/ParSeR and GuRE). A revision must cite it and position SCOPE as the *cheap, training-free, single-call* point in a spectrum that now includes trained rewriters (GuRE), RL-stabilised multi-agent reformulation (LegalMALR), and provision-generation scaffolds (ParSeR).
- **C3 (no legal expertise incorporated)**: their agents encode explicit legal-interpretation roles (element decomposition, supportive-law, jargon repair). We should either show SCOPE matches this without the machinery, or absorb the criticism honestly.
- **C5/C11 (marginal gains, no CIs / cost accounting)**: their matched-budget comparison and rollout-variance tables are the rigor template.

## Differentiation
They optimise *retrieval only* (Recall/MRR/nDCG on gold statute IDs) and never measure downstream answer accuracy — so the answer-conversion gap, SCOPE's central honest negative result, is untouched. They are Chinese-civil-law only, require GRPO training (16 GPU-hours) plus a commercial reranker, and do not analyse *why* reformulation helps (no mechanism, no QPP, no per-query geometry) or *when it hurts* (no strong-query regression analysis; no equivalent of our HousingQA finding). We are pre-empted on: the weak-query diagnosis, iterative LLM reformulation for statutes, and demonstrating this beats raw-query RAG at matched budget. We are not pre-empted on: mechanism (CE-affinity movement), regime routing, leakage/factuality falsification, or the retrieval-to-answer disconnect.

## Links
[[scope]], [[hyde]], [[vocabulary-gap]], [[weak-vs-strong-query-regime]], [[generated-query-family]], [[regime-routing]], [[qpp]], [[answer-conversion-gap]], [[legal-rag-benchmarks]], [[icml-ai4law-2026-rejection]]; siblings: [[yoon2025leakage]], [[afane2026laborbench]]

## Raw source
references/li2026legalmalr.pdf (arXiv:2601.17692v1, read pages 1-17 including method, Tables 1-6, GRPO section)
