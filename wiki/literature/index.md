---
title: Literature Vault Index
type: hub
tags: [literature, primary-sources, reproducibility]
created: 2026-07-17
updated: 2026-07-17
status: maintained
---

# Literature vault

This page is the navigation layer for the project’s primary-source archive.
The tracked wiki stores synthesis; immutable PDFs and repository checkouts live
in the persistent EIT vault:

`/engrfs/project/jacobsn/hiqbal/literature/legalrag/`

The vault currently contains **76 checksummed PDFs**, **4 checksummed primary
web/code snapshots**, **39 Git repository checkouts pinned to exact commits**,
and one extracted non-Git anonymous code snapshot, plus a separate checksummed
benchmark asset manifest. See
[`eit_papers.sha256`](manifests/eit_papers.sha256) and
[`eit_web.sha256`](manifests/eit_web.sha256), plus
[`eit_repos.tsv`](manifests/eit_repos.tsv) and
[`eit_benchmarks.sha256`](manifests/eit_benchmarks.sha256). This replaces the old assumption
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

- [[skill0]] — skills as transient scaffolding with helpfulness-driven
  withdrawal.
- [[sdar]] — decisive safety/method correction: standalone or naively mixed
  on-policy self-distillation can collapse; task RL plus gap gating matters.
- [[skill1]] — unified selection, use, and distillation of skills; broad
  “skill internalization” novelty is already occupied.

### Compute elasticity / specialist distillation

- [[compute_elasticity_handoff_2026-07-17/README]] — self-contained research,
  benchmark, harness, experiment, red-team, and implementation handoff for the
  main thread.
- [[compute-elasticity-distillation]] — corrected synthesis and minimal
  experiment. Broad token-frontier distillation is closed; the candidate is
  cost-conditioned internal-versus-tool choice with train-time-only skills.
- [[elastic-language-models]] and
  [[bard-budget-aware-reasoning-distillation]] — decisive novelty corrections:
  compute-elastic model distillation and budget-aware reasoning distillation
  already exist.
- [[rational-metareasoning]] and [[crisp]] — value-of-computation training and
  self-distilled concise reasoning.
- [[agent-distillation-tools]] and
  [[strategy-guided-policy-optimization]] — full tool-behavior and reusable
  strategy distillation; neither is price-conditioned at deployment.
- [[cost-aware-skill-rewriting]] — cost-aware external skill optimization;
  parameter internalization remains a separate question.
- [[adaptive-compute-allocation]] — external Lagrangian sample-budget
  allocation; closes generic difficulty-aware allocation novelty.
- [[privileged-information-distillation]] — PI-conditioned teacher to
  unconditioned student, including action-only frontier-agent transfer.
- [[rethinking-opd]], [[reward-gated-opd]], and
  [[rethinking-privileged-opd]] — OPD compatibility, verifier gating, and the
  critical long-budget behavior-collapse result.
- [[turnopd]] — turn-aware rollout and loss budgeting for long-horizon agents.
- [[distilling-step-by-step]], [[deepseek-r1-distillation]], and
  [[thinking-machines-expert-judgment]] — evidence that narrow specialists can
  exceed much larger or frontier generalists on selected tasks.
- [[intent-budget-constrained-agents]], [[bavt]],
  [[moc-one-model-for-all]], and [[corl-budget-controller]] — dynamic tool
  prices, budget-aware search, preference-conditioned policies, and
  budget-conditioned model routing; generic controllability is occupied.
- [[clawtrace-costcraft]], [[opid]], [[skillmoo]], and [[skillopt]] — the close
  2026 cost-aware skill tracing, on-policy skill distillation, and external
  skill-optimization cluster.
- [[reasoning-gym]] and [[agent-lightning]] — the recommended procedural task
  substrate and a reusable trace/training bridge.

### Student-specific action value / policy transport

- [[action-value-transport-reading-packet-2026-07-17]] — full seven-paper
  synthesis, novelty boundary, causal estimands, code custody, and the
  architecture-free rank-versus-threshold hypothesis.
- [[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] —
  the proposed forced-action measurement program, metrics, go/no-go gates, and
  clean connection to the three dials.
- [[student-teacher-deviations]] — owns the exact “does it pay to disobey?”
  title and broad beneficial-deviation concept; not an agent-action paper.
- [[model-adaptive-tool-necessity]] — model-specific no-tool reliability and
  knowing/doing probes; necessity is not causal tool benefit.
- [[tool-call-boundary-drift]] — OPD-induced tool-boundary movement and
  calibration; uses fixed should-call labels rather than target utility.
- [[reward-gated-opd]] and [[craft-counterfactual-credit]] — outcome-gated
  teacher trust and signed sibling-rollout token credit; neither estimates
  cross-scale forced-action transport regret.
- [[token-teachability]] — distributional absorbability is distinct from task
  utility; supplies the utility×teachability risk analysis.
- [[llm-specific-utility]] — closest reader-specific evidence-value precursor;
  deterministic binary labels motivate repeated, signed, cross-fitted effects.

### Latent reasoning and white-box intervention

- [[jacobian-global-workspace]] — J-lens/J-space readout, steering, and
  counterfactual-reflection training; requires activations and gradients.
- [[implicit-cot-distillation]], [[coconut]], and [[lori]] — hidden-state,
  continuous-latent, and low-rank trajectory transfer. Generic latent
  distillation is crowded.

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
