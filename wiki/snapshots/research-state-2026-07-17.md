---
title: Research State Snapshot — 2026-07-17
type: hub
tags: [snapshot, three-dial, opd, distillation, cleanup]
created: 2026-07-17
updated: 2026-07-17
status: maintained
---

# Research state snapshot — 2026-07-17

## Executive conclusion

The SCOPE/Snap-HyRE method story is closed as the project’s primary direction:
its method delta was small, adjacent legal work was missed, and generalizing it
across benchmarks did not produce a clean standalone contribution. The most
valuable surviving asset is the **three-dial empirical program**—how retrieval
quality, reader capability, and evidence/search cost jointly determine whether
retrieval helps or harms task success.

The best next scientific object is more precise than “adaptive RAG” or
“retrieval helpfulness”:

> **Reader-conditioned marginal evidence-set utility:** given a question,
> reader, current evidence set, and remaining budget, predict whether another
> retrieval action will improve task success, do nothing, or cause harm; stop,
> abstain, or arbitrate when the set is sufficient or conflicting.

OPD/skill distillation remains a promising implementation route for teaching a
small policy this behavior, but it is not yet a result. EIT jobs 93802 and the
July 17 follow-up 106078 prove the bare and negative-gap-gated plumbing paths,
respectively. The SDAR read changes the next step: ungated OPD is a baseline
with known collapse risk, so a real experiment needs a skill-gap gate, task
reward, and gap-gated dense supervision.

## What was actually done

### Local research program

1. SCOPE/HyRE was evaluated across legal and general benchmarks, with a
   geometric affinity-margin mechanism and multiple retrievers/generators.
2. A trained 9B legal evidence judge substantially improved top-5 gold exposure
   over the fixed CE on held-out pools.
3. Paired answer experiments measured whether that exposure helped readers.
4. An offline per-question bandit and a learned 9B allocation policy tested
   whether retrieval effort could be controlled cheaply.
5. An OPD stack was implemented: vLLM teacher logprobs, reverse-KL-style
   on-policy loss, LoRA student training, KD fallback, and a skill file.

The reconciled machine ledger now contains **671 valid JSON rows**; the 20
July-only rows were merged into the historical 651-row ledger without dropping
the prior record. Seventy-one larger July logs have a SHA/size/path manifest.

### EIT program

Ten relevant Slurm jobs were adjudicated from their stdout rather than scheduler
status. The durable summary is
[`docs/july_2026_completion_audit_2026-07-17.md`](../../docs/july_2026_completion_audit_2026-07-17.md).
The most important outcomes are:

- job 93632: successful specialist BarExam judge, 82/399;
- job 93660: successful mixed legal judge, 88/399 BarExam and 277/500 Housing;
- job 93770: allocation model training completed, but its policy evaluation was
  performed later in the local analysis, not by the EIT stdout alone;
- job 93802: Qwen3-8B teacher → Qwen3-1.7B student, three finite OPD steps,
  checkpoint created, full smoke PASS;
- follow-up job 106078: the clean OPD branch ran three finite `opd_gated`
  steps, logged gate means, wrote step/final checkpoints, and completed 0:0;
  this validates the safeguarded dense-objective implementation, not task RL;
- jobs 93598/93606/93629/93656/93658/93773 are partial, cancelled, or failed
  and are preserved as failure provenance rather than silently counted.

No relevant EIT job was still running at the audit point, and no new legal-RAG
experiment activity after July 2 was found.

## The three dials, strongly stated

### Dial 1 — evidence exposure and set quality

The trained judge lifted BarExam top-5 exposure from CE 3.8% to 20.6% and
Housing exposure from 38.2% to 55.0% on the paired July subsets. This proves a
selector bottleneck exists, but it does not establish end-task utility. Pool
recall, evidence diversity, redundancy, contradiction, and set sufficiency all
sit inside this dial.

### Dial 2 — reader–task ability to convert evidence

The same evidence behaves differently for different readers/tasks:

| Reader/task | Best evidence vs no evidence | Interpretation |
|---|---:|---|
| BarExam / 70B | −2.5pp, not significant | Strong parametric reader; distractor cost dominates at low exposure |
| Housing / 70B | +11.4pp, p≈5.5e-08 | Evidence-valuable statutory task; selector gains convert |
| BarExam / 8B | +11.8pp, p≈5.6e-05 | Weaker reader benefits even from imperfect topical evidence |
| Housing / 8B | −2.8pp, not significant | Evidence integration can fail even where the larger reader benefits |

On BarExam/70B, judge evidence was +2.4pp when gold was present but −3.8pp
when absent, implying a roughly 61% exposure break-even—well above the tested
pool ceiling. On Housing/70B, even gold-absent context was +12.0pp. This is the
core evidence that relevance and helpfulness are reader-conditioned and can
reverse sign.

These are paired subset findings, not universal thresholds. The Housing subset
runs hotter than its full-N baseline, and the 60% reader-accuracy crossover is
a hypothesis requiring replication, not a law of nature.

### Dial 3 — search effort and marginal cost

The offline bandit failed to beat the best fixed arm in all five tested cells,
despite a noise-inflated per-question oracle 8–24pp above fixed policies. The
9B allocation model learned coarse regime preferences and beat its zero-shot
version in important cells, but still did not beat the best fixed arm at zero
cost penalty and selected `llm_only` only 5/200 times for BarExam/70B—the exact
case where “do not retrieve” was the right rule.

Therefore the next effort-control paper cannot be another fixed-k sweep or a
cheap QPP gate. It needs repeated outcomes, reader/evidence-state features, and
a learned marginal-utility stopping decision evaluated as an accuracy–cost
frontier.

## Literature boundary after the new primary-source pass

The persistent EIT literature vault now contains 27 checksummed PDFs, one
checksummed primary web snapshot, and 11 repository checkouts pinned to commits;
navigation begins at
[[literature/index]]. The pass materially narrowed the novelty claim:

- Predicting Retrieval Utility and the Beyond Relevance tutorial occupy generic
  retrieval-utility framing.
- CUE-R occupies per-evidence causal interventions and demonstrates
  non-additivity.
- BCAS occupies controlled search-budget/cost ablations.
- SURE-RAG occupies set-level sufficiency/abstention; ConflictRAG and ArbGraph
  occupy generic conflict resolution.
- SKILL0, Skill1, and SDAR occupy skill internalization, unified skill
  evolution, and self-distilled agentic RL.

The defensible wedge is the joint object: **paired causal, reader-conditioned,
set-level evidence utility and harm under cost**, with legal authority
semantics where conflict is studied. This is narrower, but considerably more
real than the earlier generic claims.

## Track decisions

### Active: [[three-dial]]

Primary research track. Formalize the outcome/estimand, reconstruct a clean
paired dataset, quantify uncertainty with repeated generations, reproduce one
non-legal cell, and compare learned stopping against fixed-budget, utility-
prediction, intervention, and sufficiency baselines.

### Parallel, gated: [[opd-distillation]]

Engineering track for internalizing the three-dial decision into a small
policy. Bare and gap-gated software smokes are green. E2 must first establish that the
allocation skill makes a teacher measurably better. If it does, E3 compares
task-RL alone, bare OPD, gap-gated OPD+task-RL, skill-context inference, and
trace KD. If it does not, move to a task with genuine skill headroom rather
than forcing distillation.

### Historical: [[scope-old]]

The submitted method, reviews, generated-query experiments, and old paper
artifacts remain available for provenance and reusable controls. They are not
the active framing.

## Cleanup and recovery state

- `codex/scope_old` preserves the old submission/review package in Git.
- Tested ZIPs and per-file manifests exist at
  `/Users/hamzaiqbal/grad/LegalRagAgent_archive/`; they cover the historical
  SCOPE worktree, early agentic course project, class report, old paper tree,
  and pre-pivot root surface.
- A full pre-cleanup recovery bundle and worktree backup exists at
  `/Users/hamzaiqbal/grad/LegalRagAgent_recovery_20260717`.
- The active clean-development branches are `codex/three_dial` and
  `codex/opd_distillation`.
- The old dirty EIT worktrees are historical evidence until their diffs and
  untracked files are bundled. They must not be treated as synchronized active
  worktrees.

## Next sequence and decision gates

1. **Three-dial data contract:** define the unit as a paired
   `(question, reader, evidence set/action)` record with correctness, exposure,
   evidence role/conflict, tokens/calls/latency, and repeated-sample variance.
2. **Recompute the master table:** verify every July paired effect directly
   from row logs; add bootstrap/McNemar uncertainty and distinguish
   gold-present, gold-absent, sufficient, conflicting, and redundant sets.
3. **General-domain replication:** one dataset, two reader sizes, no broad
   benchmark sweep. Test the sign-reversal claim rather than SCOPE again.
4. **E2 skill-gap A/B:** teacher with versus without the allocation skill on a
   held-out, pre-registered task. Required to justify any distillation run.
5. **Safe E3:** if E2 passes, run task reward plus gap-gated OPD with a bare-OPD
   diagnostic arm. Do not interpret loss finiteness as task learning.
6. **Legal conflict subtrack:** only after a primary-source search for
   authority-aware legal RAG; encode jurisdiction/court/date/precedent rather
   than generic credibility.

## Questions this snapshot leaves open

1. Does the reader-conditioned sign reversal survive repeated generation and
   a non-legal dataset?
2. What observable state predicts marginal evidence utility better than
   question text and judge-score summaries?
3. Is set sufficiency a more stable target than exact gold-passage exposure?
4. Can an evidence-aware policy learn to abstain or stop without sacrificing
   the fixed-arm accuracy frontier?
5. Does skill context actually improve the teacher on the BarExam/70B
   `do-not-retrieve` case, or is the task itself too noisy for distillation?
6. Which legal authority signals are available in current corpora without a
   new annotation project?

## Evidence map

- Cite gate: [`docs/signoff_log.md`](../../docs/signoff_log.md)
- July audit: [`docs/july_2026_completion_audit_2026-07-17.md`](../../docs/july_2026_completion_audit_2026-07-17.md)
- Compact evidence: [`evidence/july_2026/`](../../evidence/july_2026/)
- Core result pages: [[judge-answer-conversion]], [[offline-bandit-v0]],
  [[alloc-internalization-rung2]]
- Literature audit: [[coverage-audit-2026-07-17]]

This snapshot records the 2026-07-17 state. Future work should append or link a
new dated snapshot rather than silently rewriting this one.
