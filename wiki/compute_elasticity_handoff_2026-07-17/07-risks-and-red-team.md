---
title: Risks and Red-Team Checklist
type: review
tags: [risks, red-team, novelty, reward-hacking]
created: 2026-07-17
updated: 2026-07-17
status: maintained
---

# Risks and red-team checklist

## Novelty risk

The exact intersection may already exist under “preference-conditioned agent
distillation,” “budget-conditioned tool learning,” or “multi-objective policy
distillation.” Mitigation: rerun a focused venue-date search before code freeze
and submission. Treat MOC, INTENT, CoRL, BARD, OPID, and ClawTrace as mandatory
baselines/citations, not adjacent work.

## No phenomenon to distill

A teacher may ignore price, always use Python, or obey price globally without
task sensitivity. That would invalidate the central object. The Phase 0 gate
comes before any training investment.

## Confounding skill with scale

A skilled teacher may improve because it receives more instructions, not
because those instructions encode reusable decision knowledge. Compare
unskilled/skilled teacher behavior at identical prices and use length-matched
control text.

## Price is an arbitrary scalar

Normalized prices can produce a curve that disappears under another scale.
Report raw success/calls and rescore across a broad lambda grid. Center final
prices on observed break-even regions. Add a real-cost external benchmark only
after the synthetic causal result.

## Prompt compliance masquerades as metareasoning

A model can map “expensive” to “do not call” without estimating value. The
same price must induce different decisions on easy versus hard instances.
Compare against global threshold and difficulty-only heuristics.

## Reward hacking

Reasoning Gym native scorers have heterogeneous shaping and partial credit.
Store native reward and predeclared exact success separately. Audit malformed,
non-prime factorization, valid-but-long path, and invalid-expression cases.

## Tool leakage and unfair access

Python may expose libraries or filesystem/network state unavailable to other
arms. Freeze imports, sandbox image, timeouts, and I/O. Teacher, student, and
frontier comparisons receive identical tools.

## Distillation collapse

Mixed-price SFT can average incompatible trajectories. OPD can erase
backtracking/verification or overfit teacher mistakes. Measure switch
preservation, action diversity, and verification rate, and gate teacher loss by
task reward/gap.

## Teacher is not an oracle

Teacher choices are not automatically optimal. Report regret to both teacher
and empirical per-item best action. A student may rationally disagree with a
teacher whose internal/tool capabilities differ.

## Identifiability across scale

The teacher and student have different internal compute/value functions, so
copying the teacher's action can be suboptimal for the student. Distinguish:

- **behavioral fidelity:** student matches teacher switches;
- **student utility:** student maximizes its own frontier;
- **knowledge transfer:** skilled teacher supervision improves student utility.

These are separate claims.

## Procedural benchmark artifacts

A student may learn generator templates or tool shortcuts. Use held-out seed
ranges, multiple families, prompt variants, and one external benchmark. Keep
the pilot seeds out of reported test data.

## Specialist-over-frontier overclaim

Beating a frontier API on procedural tasks can reflect specialization, tool
format mismatch, or cap differences. Match protocols and describe the result
as a specialized frontier under a fixed distribution, not general superiority.

## Harness drift

Verifiers, NeMo Gym, and VERL are changing quickly. Own the trace schema and
task manifests. Pin exact repository commits. Golden traces must remain valid
after any dependency update.

## Cluster/reproducibility risk

EIT home quota is constrained and a Python install already failed when uv used
the home default. Keep `UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` under the
persistent project vault or allocated scratch; record that path in sbatch
scripts. Never rely on an interactive login process for final experiments.
