---
title: L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search
type: source
tags: [legal-rag, agentic-search, effort-control, current-law, sufficiency]
created: 2026-07-17
updated: 2026-07-17
status: triaged
url: https://arxiv.org/abs/2509.00761
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2509.00761v3.pdf
authors: Wang and Yuan
year: 2026
venue: arXiv
code: https://github.com/boqiny/L-MARS
---

# L-MARS

## Why it matters

L-MARS combines query decomposition, online/local/case-law search, a judge for
evidence sufficiency and contradiction checks, and cited synthesis. Its
LegalSearchQA result is deliberately retrieval-dependent and time-sensitive,
while its BarExamQA result is nearly flat. That benchmark reversal is close to
our central observation that evidence utility depends on the task and reader.

The paper also describes an iterative mode that searches again when the judge
flags missing authority, jurisdiction, temporal specificity, or contradiction.
This is direct neighboring work for search-effort control and legal conflict
handling. A generic planner/judge/retrieve loop is therefore not a novelty
claim for us.

## Design consequences

- Use frozen corpora or archived pages for scientific evaluation; live search
  results drift and should not be mixed with fixed-corpus claims.
- Compare learned stopping against L-MARS-style sufficiency-triggered search,
  not only fixed `k` or fixed call budgets.
- Encode jurisdiction, date, authority, and contradiction state if pursuing a
  legal conflict subtrack.
- Include at least one truly retrieval-dependent benchmark; BarExam alone can
  make useful retrieval look inert for a strong reader.

## Reading state

The abstract, system/mode description, benchmark contract, and agentic-search
mechanism were reviewed during archive migration. Full results and released
code still require a dedicated reproduction pass before citation-level claims
are finalized.

## Links

[[three-dial]] · [[budget-constrained-agentic-search]] · [[sure-rag]] ·
[[conflictrag]] · [[arbgraph]] · [[zheng-cslaw]]

## Raw source

- EIT PDF: `papers/arxiv_2509.00761v3.pdf`; pinned code: `repos/L-MARS`
- SHA-256: `705a8222fe8192755926215c08b6a414c0e666b125023325c40e65da7c753b3a`
