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

The vault currently contains **107 checksummed PDFs**, **4 checksummed primary
web/code snapshots**, **61 Git repository checkouts pinned to exact commits**,
and one extracted non-Git anonymous code snapshot, plus a separate checksummed
benchmark asset manifest. See
[`eit_papers.sha256`](manifests/eit_papers.sha256) and
[`eit_web.sha256`](manifests/eit_web.sha256), plus
[`eit_repos.tsv`](manifests/eit_repos.tsv) and
[`eit_benchmarks.sha256`](manifests/eit_benchmarks.sha256). This replaces the old assumption
that `/engrfs/tmp/.../references/` was the archival copy; scratch may still
contain working duplicates, but it is not the source of truth.

## Read first for the live directions

- [[research-question-recommendation-2026-07-17]] — current decision snapshot:
  student-specific forced-action value is the primary measurement question;
  placement-conditioned skill utility is the narrower secondary candidate.

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

- [[self-distillation-cluster-update-2026-07-17]] — integrated reading of
  OPSD, SDFT, and SDPO: generic privileged-context-to-weights self-distillation
  is occupied, capability-gated, and distinct from causal action value.
- [[opsd-self-distilled-reasoner]] — verified-solution-conditioned same-model
  on-policy soft distillation; mandatory unconditional privileged-context
  baseline, not a forced-action-value estimator.
- [[sdft-continual-learning]] — demonstration-conditioned same-model
  self-distillation for skill/fact acquisition with reduced forgetting; the
  source page records a critical paper/code reverse-KL versus forward-KL
  mismatch.
- [[sdpo-rich-feedback]] — feedback-conditioned hindsight pseudo-advantages,
  scale-dependent self-teaching, and test-time context compression; its
  log-ratio is not causal action value.
- [[skill0]] — skills as transient scaffolding with helpfulness-driven
  withdrawal.
- [[sdar]] — decisive safety/method correction: standalone or naively mixed
  on-policy self-distillation can collapse; task RL plus gap gating matters.
- [[skill1]] — unified selection, use, and distillation of skills; broad
  “skill internalization” novelty is already occupied.

### Skill lifecycle — context, weights, and teaching material

- [[skill-lifecycle-research-snapshot-2026-07-17]] — question-first synthesis,
  exact novelty boundary, crossed source/target/teaching-utility measurements,
  shortcut controls, and durability tests. No experiment has been launched.
- [[skillopt]] — held-out-gated optimization of an external textual skill;
  the paper itself names later weight internalization as future work.
- [[skillgen-verified]] — the strongest direct contextual collision: six fixed
  source-conditioned skills crossed with six readers on common held-out
  instances; context rank/reversal is a control, not the contribution.
- [[masa]] and [[skilllens]] — controlled three-artifact × seven-reader and
  two-artifact × six-reader matrices, respectively.
- [[skillrevise]] — fixed source-selected artifact transfer versus
  target-conditioned revision across four target readers.
- [[skillmaster]] — counterfactual skill-bank edit utility and aggregate
  retrieval-withdrawal persistence after skill-guided training; no
  fixed-artifact causal attribution.
- [[skilladaptor]] — training-free step-level skill repair; useful operational
  neighbor but not a collision with matched post-withdrawal acquisition.
- [[smartad]] — target-student-NLL selection of successful agent trajectories
  plus action/final-span-weighted SFT; capacity-aligned agent distillation is
  occupied.
- [[informative-alignment-rsr]] — 11-teacher × five-student post-training panel
  and Rank–Surprisal Ratio; generic student-specific teaching-utility metrics
  are occupied.
- [[promptkd]] — direct conceptual correction: teacher-side soft context is
  already optimized with student guidance for generative distillation;
  “student-friendly teaching context” is not a new general claim.
- [[lgtm-student-level-kd]], [[personalized-teacher-selection]], and
  [[distillation-traps-guards]] — teacher validation influence, per-student
  teacher routing, and direct control of LLM distillability. Together they
  close broad “best executor is not best teacher” and generic teachability
  claims; the surviving object is fixed procedural-artifact utility transport.
- [[opcd]] and [[skill-sd]] — direct context-conditioned OPD and dynamic
  teacher-only trajectory skills; they close the generic skills-plus-OPD
  method claim.
- [[seed-self-evolving-opd]] — self-evolving hindsight skills, task RL, and
  gap-gated OPD with no skill at deployment; the closest corrected-E3
  collision.
- [[skillc]] and [[skill-zero-five]] — paired skill/no-skill internalization
  credit and deliberate splitting of general in-weight versus task-specific
  external skills.
- [[latent-skill]] — textual skills compiled into selectable LoRA adapters;
  permanent weight internalization is not the only alternative to context.
- [[structured-prompt-optimization-extc]] — the literal structured-text
  optimization → large-to-small trace distillation → RL pipeline, including a
  legal classification benchmark.
- [[continual-facts-in-weights]] — context-versus-weight creation,
  composition, retention, and rescue for invented facts; procedural skills
  remain explicitly untested.
- [[sdft-continual-learning]], [[opsd-self-distilled-reasoner]], and
  [[sdpo-rich-feedback]] close generic demonstration/solution/feedback-to-
  weights claims. None ranks several fixed skill artifacts in one target's
  context and after matched withdrawal.

The pinned repositories are code-custody anchors, not complete reproductions.
In particular, SkillGen, SkillLens, SkillRevise, MASA, and SkillMaster do not
release every exact final/intermediate artifact, raw crossed matrix, run log,
or checkpoint needed to reconstruct all paper claims. Their dedicated pages
record the gaps. The SDFT checkout omits its Medical/Wikipedia and sequential
experiment surfaces and disagrees with the paper about the trained KL
direction. The Skill-Usage, Ctx2Skill, and RSR dataset links/commit
identities are recorded, but their full Hugging Face payloads are not mirrored
in this vault.

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
  synthesis, novelty boundary, causal estimands, code custody, and a
  question-first discovery plan with several candidate empirical patterns.
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
- [[smartad]] and [[informative-alignment-rsr]] — student-specific agent-trace
  compatibility and post-training teaching value; neither estimates immediate
  forced external-action value or target regret from a teacher boundary.
- [[sdpo-rich-feedback]] — assigns feedback-conditioned token/logit
  pseudo-advantages and exposes a model-scale reliability boundary, but never
  compares forced task outcomes or transfers a teacher action policy across
  readers.
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
