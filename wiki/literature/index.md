---
title: Literature Vault Index
type: hub
tags: [literature, primary-sources, reproducibility]
created: 2026-07-17
updated: 2026-07-20
status: maintained
---

# Literature vault

This page is the navigation layer for the project’s primary-source archive.
The tracked wiki stores synthesis; immutable PDFs and repository checkouts live
in the persistent EIT vault:

`/engrfs/project/jacobsn/hiqbal/literature/legalrag/`

The shared vault currently contains **110 checksummed PDFs**, **4 checksummed
primary web/code snapshots**, and **63 repository checkouts pinned to exact
commits**. See
[`eit_papers.sha256`](manifests/eit_papers.sha256) and
[`eit_web.sha256`](manifests/eit_web.sha256), plus
[`eit_repos.tsv`](manifests/eit_repos.tsv). This replaces the old assumption
that `/engrfs/tmp/.../references/` was the archival copy; scratch may still
contain working duplicates, but it is not the source of truth.

## Read first for the live directions

### Three-dial / reader-conditioned evidence utility

- [[predicting-retrieval-utility]] — direct prior art for utility prediction;
  makes a generic “helpfulness rather than relevance” claim non-novel.
- [[cue-r]] — intervention-based per-item evidence utility and non-additivity.
- [[beyond-relevance-utility]] — tutorial-level map of the utility-centric IR
  area; confirms this is a field, not an empty niche.
- [[budget-constrained-agentic-search]] — controlled search/cost ablations;
  motivates learned marginal-utility stopping rather than another budget grid.
- [[sure-rag]] — set-level sufficiency and abstention.
- [[conflictrag]] and [[arbgraph]] — conflict detection/arbitration; generic
  conflict resolution is crowded.

### OPD / skill distillation

- [[self-distillation-cluster-update-2026-07-17]] — integrated OPSD/SDFT/SDPO
  decision: context-to-weights self-distillation is occupied and capability-
  gated, while causal action value remains a separate object.
- [[opsd-self-distilled-reasoner]] — verified-solution-conditioned same-model
  on-policy distillation.
- [[sdft-continual-learning]] — demonstration-conditioned self-distillation
  with a critical paper/code reverse-KL versus forward-KL discrepancy.
- [[sdpo-rich-feedback]] — rich-feedback self-teaching whose token/logit
  pseudo-advantage is not causal action value.
- [[skill0]] — skills as transient scaffolding with helpfulness-driven
  withdrawal.
- [[sdar]] — decisive safety/method correction: standalone or naively mixed
  on-policy self-distillation can collapse; task RL plus gap gating matters.
- [[ema-policy-gradient]] — separates K1 values from K4-equivalent gradients
  and supplies the exact-head plus sampled-tail Top-k KL comparison.
- [[verl-opd-trainer]] — pinned upstream OPD implementation and its much
  heavier actor/teacher resource contract; reference implementation, not our
  task-reward gate.
- [[mopd-multi-teacher]] — multi-teacher routed OPD for integrating separately
  trained domain capabilities; one teacher per sample, not teacher averaging
  or conflict arbitration.
- [[mopd-multi-rollout]] — a distinct peer-conditioned method that also uses
  the acronym MOPD; archived separately to prevent a literature/naming mix-up.
- [[skill1]] — unified selection, use, and distillation of skills; broad
  “skill internalization” novelty is already occupied.

### Prior reviewer misses and legal retrieval family

- [[gure]], [[koblex-parser]], [[legal-rag-benchmarks-src]],
  [[icml-ai4law-2026-rejection]].
- [[zheng-cslaw]] — the native BarExamQA/HousingQA retrieval benchmark and its
  retrieval-to-answer conversion limit.
- [[lrage]] — legal RAG component-ablation/evaluation framework.
- [[l-mars]] — agentic legal search, sufficiency checks, and the contrast
  between time-sensitive LegalSearchQA and nearly-flat BarExam retrieval.

## Operating rule

Every new paper that materially changes a claim must get a source page, links
to the affected concept/track pages, an index entry, and a log entry. A paper
being downloaded is not the same as it being read. The coverage audit at
[[coverage-audit-2026-07-17]] distinguishes archived, read, and synthesized.
