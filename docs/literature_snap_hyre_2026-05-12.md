# Literature Notes For Snap-HyRE Pivot - 2026-05-12

This historical note originally used the now-archived local
`literature/papers/` folder. The PDFs live durably in the EIT vault at
`/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/`; the removed local
PDF/text copies are also in the verified pre-pivot archive named in
`docs/archive_manifest_2026-07-17.md`.

## L-MARS / LegalSearchQA

Source: EIT `papers/arxiv_2509.00761v3.pdf`; current notes: `wiki/sources/l-mars.md`

Relevant claim: L-MARS reports a large retrieval gain on current-law
LegalSearchQA, while Bar Exam QA stays nearly flat with web retrieval. This is
useful motivation for the claim that legal RAG value depends on the task's
retrieval dependence.

Use in our paper:

- Cite as related work on current-law legal QA and agentic legal search.
- Use LegalSearchQA as motivation that stale parametric knowledge is a distinct
  regime.
- Do not treat LegalSearchQA as a main Snap-HyRE benchmark unless we snapshot
  URLs/pages into a frozen corpus and log qrels.

Why not mainline now:

- 50 questions.
- Source URLs are live web pages, not a stable local corpus.
- Answers are time-sensitive and were verified by the authors at a specific
  time.
- Search engine rankings and snippets can drift.

## A Reasoning-Focused Legal Retrieval Benchmark

Source: EIT `papers/arxiv_2505.03970.pdf`; current notes: `wiki/sources/zheng-cslaw.md`

Relevant claim: BarExamQA and HousingQA are designed so legal retrieval itself
requires reasoning. The paper separates retrieval metrics from downstream QA
and shows retrieval gains do not always convert into answer gains.

Design points to mirror:

- Report retrieval metrics directly: Recall/Hit@1, @5, @10 and MRR@10.
- Keep downstream answer accuracy separate from retrieval exposure.
- For HousingQA, distinguish any-gold retrieved from all-gold retrieved.
- Include no-passage, retrieved-passage, reasoning-pseudo-passage, and
  gold-passage style controls when budget allows.
- Treat BarExamQA and HousingQA differently in analysis: BarExam is heavy on
  issue/rule reasoning and answer-option anchoring; HousingQA is heavy on
  jurisdiction-scoped statutory retrieval and yes/no entailment.

## LRAGE

Source: EIT `papers/arxiv_2504.01840v1.pdf`; current notes: `wiki/sources/lrage.md`

Relevant claim: legal RAG performance is a component interaction among corpus,
retriever, reranker, LLM backbone, top-k, and metric/rubric.

Design points to mirror:

- Component ladder tables with absolute score and delta versus baseline.
- Explicit model axis; do not claim a method transfers unless it transfers by
  model.
- Top-k sensitivity figures.
- Negative-result discipline: RAG, rerankers, and generated queries can hurt.

## Consequence For This Branch

The comprehensive Snap-HyRE story should not be "retrieval always improves
legal QA." A stronger, simpler claim is:

> Snap-HyRE improves legal evidence exposure more reliably than it improves
> downstream answers. When downstream accuracy does not move, the row is still
> informative because it separates retrieval quality from context utilization.

This is compatible with a fixed method table as long as every row reports both
retrieval metrics and answer accuracy under the same top-k setting.
