---
title: Legal RAG Benchmarks (landscape)
type: concept
tags: [benchmarks, legal-rag, evaluation]
created: 2026-07-02
updated: 2026-07-02
status: stub
---

# Legal RAG benchmark landscape

Hub for what exists, what we use, and what the legal-NLP community expects
(criticism C2). To be fleshed out from [[zheng-cslaw]],
[[legal-rag-benchmarks-src]] (LegalBench-RAG + Legal RAG Bench pages), and
discovery sweep A.

**We currently use** (comprehensive matrix, [docs/signoff_log.md](../../docs/signoff_log.md)):
BarExamQA (1195 MC, gold passages), HousingQA (6853 y/n, statutes +
jurisdiction filter), Legal-Link-EU (1127, EUR-Lex anchored), MASLegalBench
(303, no gold ids); historical: CaseHOLD, LegalBench-SCALR; widened:
MedQA-USMLE, MuSiQue, BEIR-5 (non-legal controls).

**Community benchmarks we cited but didn't use**: LegalBench, LexGLUE,
Pile of Law, LegalBench-RAG, Legal RAG Bench; reviewer-implied gaps: KoBLEX
(Korean statutory multi-hop), GuRE's setting (Korean precedent retrieval?
verify), plus whatever sweep A surfaces (L-MARS/LegalSearchQA, LRAGE, 2026
entrants).

**Known sharp edges** (keep honest):
- BarExamQA gold labels are single-passage while multiple passages may
  support an answer → Hit@k pessimistic; run
  `scripts/audit_retrieval_id_alignment.py` before exposure claims.
- MASLegalBench has no per-question gold evidence ids (proxy metrics only).
- HousingQA answer distribution issues (answer-bound; de-emphasized
  2026-05-25 memory note).

## Links
[[zheng-cslaw]] · [[icml-ai4law-2026-rejection]] · [[scope-paper-2026]] ·
[[weak-vs-strong-query-regime]]
