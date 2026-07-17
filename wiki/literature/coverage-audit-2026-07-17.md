---
title: Literature Coverage Audit — 2026-07-17
type: review
tags: [literature, coverage, novelty]
created: 2026-07-17
updated: 2026-07-17
status: maintained
---

# Literature coverage audit — 2026-07-17

## Coverage obtained

| Cluster | Primary sources read in this pass | What changed |
|---|---|---|
| Reader-conditioned utility | Predicting Retrieval Utility, CUE-R, Beyond Relevance | “Retrieval helpfulness” alone is not a novelty claim. The remaining wedge is paired causal, reader-conditioned **set** utility with harm/break-even and cost. |
| Effort control | BCAS | Generic search-depth/cost ablation is occupied. The experiment must predict marginal value and stop, conditioned on reader/evidence state. |
| Conflict/sufficiency | ConflictRAG, SURE-RAG, ArbGraph | Generic conflict detection/arbitration is occupied. A legal contribution needs authority, jurisdiction, date, and precedent structure, plus abstention/sufficiency. |
| Skill internalization | SKILL0, SDAR, Skill1 | Broad skill internalization and same-policy co-evolution are occupied. Bare OPD is unsafe as the primary method; task RL and gap-gated distillation are the evidence-backed next design. |
| Reviewer-named legal work | GuRE, KoBLEX and the existing legal-RAG pages | The old SCOPE method cannot be revived without directly engaging trained legal rewriting and provision-generation near-twins. |

## Strongest defensible research opening

The project’s best under-owned object is not “utility,” “adaptive search,” or
“conflict resolution” in isolation. It is the **reader-conditioned marginal
utility of an evidence set**:

> Given a reader, question, current evidence set, and remaining budget, estimate
> whether another retrieval action will improve task success, leave it
> unchanged, or cause harm—and abstain/arbitrate when the set is insufficient
> or conflicting.

That object unifies the three dials already measured locally:

1. evidence exposure/quality;
2. reader ability to convert evidence into a correct answer;
3. cost and marginal search effort.

It also creates a precise relationship to the crowded areas: utility
prediction supplies prediction baselines, CUE-R supplies interventions,
SURE-RAG supplies set-level sufficiency, and budgeted search supplies fixed
policies. Our work must beat or explain those baselines, not rename them.

## Remaining high-priority gaps

1. Read and archive the primary papers/repos for RAGAS/ARES/RAGChecker and
   recent selective-RAG/abstention work; the present vault has partial coverage
   through citations and older source pages, but not a complete primary-source
   bundle.
2. Search legal authority-aware retrieval/conflict work specifically (court
   hierarchy, jurisdiction, recency, negative treatment). Generic RAG conflict
   papers are not enough to establish the legal wedge.
3. Add at least one non-legal reader×evidence replication before claiming the
   three-dial relation generalizes.
4. Keep a venue-date cutoff and rerun the literature search before submission;
   several decisive neighbors here appeared in April–May 2026.

## Reproducibility

PDF hashes and repository commits are tracked in [[literature/index]]. The full
primary sources live in the persistent EIT vault, while these Markdown pages
remain local, searchable, and Obsidian-linked.
