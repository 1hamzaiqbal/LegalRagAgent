---
title: SCOPE — When Generated Legal Queries Help Legal RAG (our AI4Law 2026 submission)
type: source
tags: [scope, legal-rag, our-paper, rejected]
created: 2026-07-02
updated: 2026-07-02
status: maintained
url: (OpenReview AI4Law submission 97)
local: official_paper_and_review_icml_ai_4_law/97_SCOPE_When_Generated_Legal_.pdf
authors: Iqbal, Li, Li, Huang, Huang
year: 2026
venue: ICML AI4Law workshop (rejected)
code: eval/eval_harness.py
---

# Our submitted paper — what it actually claimed and showed

**TL;DR**: SCOPE = two-call generated-query legal RAG ([[scope]]). Evaluated at
the two extremes of the question–corpus vocabulary gap: BarExamQA (weak-query)
and HousingQA (strong-query), 3 model sizes, same model as generator+answerer.
Headline: BarExamQA Hit@5 1.4% → 9.5/12.1/11.0% (8B/26B/70B) with answer gains
framed vs raw-question RAG (+2.4/+4.0/+5.1pp). Rejected with two strong rejects
— inventory in [[icml-ai4law-2026-rejection]].

## The numbers that matter (for honest reuse)

**Table 1 (answers, %):**
| Method | BarExam 8B | 26B | 70B | avg | Housing 8B | 70B | avg |
|---|---|---|---|---|---|---|---|
| LLM-only | **57.3** | 80.8 | 78.7 | 72.3 | 55.4 | 44.8 | 50.2 |
| Raw-question RAG | 54.5 | 78.0 | 74.6 | 69.0 | **62.3** | 62.1 | **62.2** |
| HyDE | 56.1 | 80.3 | **80.2** | 72.2 | 59.1 | **62.2** | 60.7 |
| SCOPE | 56.9 | **82.0** | 79.7 | **72.9** | 59.0 | 59.6 | 59.3 |
| *Gold evidence* | *60.0* | *78.6* | *79.2* | *72.6* | *64.3* | *67.3* | *65.8* |

Read them cold: on BarExamQA vs **LLM-only** SCOPE is −0.4/+1.2/+1.0pp (C9);
on HousingQA SCOPE is the *worst* non-LLM method (C10). Gold evidence itself
only reaches 72.6 avg on BarExamQA — even perfect retrieval barely beats
parametric answering there ([[answer-conversion-gap]]); note the striking 26B
cell where SCOPE 82.0 > gold 78.6 (retrieved context beats the single labeled
gold — neighbor/context effects are real).

**Retrieval (Gemma 26B, k=5)**: raw 1.4/0.7 (Hit@5/MRR@5) → HyDE 11.4/5.4 →
SCOPE 12.1/6.0 on BarExamQA; Housing (state-filtered): raw 36.9/23.3 → HyDE
30.6/19.6 → SCOPE 38.1/24.5. Jurisdiction filter, not query method, drives
Housing retrieval (2.8 → 36.9 Hit@5).

**Other pieces**: one-shot corpus exemplar in the generation prompt lifts
retrieval on every benchmark (+0.6 to +7.6 Hit@5, Table 5) at flat answer
accuracy; pooled slice deltas (Table 6): snap-vs-raw +1.1, HyDE-vs-snap +0.1,
gold-vs-raw only +3.6; token-efficiency table excluded call-1 tokens (C11).

## What holds up vs what was framing
- Holds up: the weak-query retrieval lift (large, replicated, model-robust);
  the two-regime setup itself ([[weak-vs-strong-query-regime]]); the honest
  appendix (inclusion policy, 31/42 cells disclosed).
- Framing that failed review: gains quoted vs weakest baseline; Housing
  "parity"; snap-vs-HyDE presented as a win without a test; "leading
  token-efficiency" with stage-1 excluded; guardrail asserted (C12).
- Untold story the data already contained: retrieval-answer decoupling — the
  paper *observed* it (line 308: gold appears in ~1/9 top-5 lists; gold-only
  ceiling low) but didn't make it the object of study. Post-submission work
  does ([[geometry-vs-factuality]], [[regime-routing]]).

## Links
[[scope]] · [[icml-ai4law-2026-rejection]] · [[zheng-cslaw]] (benchmark
source) · [[hyde]] · [[query2doc]] · [[answer-conversion-gap]] ·
[[weak-vs-strong-query-regime]]

Raw source: `official_paper_and_review_icml_ai_4_law/97_SCOPE_When_Generated_Legal_.pdf`
