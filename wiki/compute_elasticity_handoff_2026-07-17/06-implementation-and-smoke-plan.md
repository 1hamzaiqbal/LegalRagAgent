---
title: Implementation and Smoke Plan
type: plan
tags: [implementation, smoke-test, eit, verifiers]
created: 2026-07-17
updated: 2026-07-17
status: proposed
---

# Implementation and smoke plan

## Minimal project surface

Create a new implementation subtree only after the main thread accepts this
specification:

```text
experiments/elasticity/
  configs/
  data/
  env/
    task_source.py
    priced_python_env.py
    verifier_adapter.py
  adapters/
    verifiers_adapter.py
    inspect_adapter.py
  rollouts/
    trace_schema.py
    writer.py
  train/
    conditional_sft.py
    direct_rl.py
    gated_opd.py
  analysis/
    validate_traces.py
    elasticity_metrics.py
  tests/
```

Do not mix this code into the historical Snap-HyRE scripts or paper tables.

## Interfaces

`TaskSource.sample(split, seed) -> Task`

- returns stable ID, prompt, oracle metadata, family, difficulty, and generator
  version;
- has no model or price logic.

`PricedToolEnv.reset(task, price, limits) -> Observation`

- exposes task and price;
- executes one fixed Python tool;
- records calls and outputs verbatim;
- does not scalarize reward.

`Verifier.score(task, final_answer) -> RewardComponents`

- returns native score, exact success, and diagnostics;
- never reads price.

`TraceWriter.append(episode) -> None`

- validates against the canonical schema;
- writes append-only JSONL atomically;
- includes model/environment/config hashes.

## Smoke ladder

### S0 — data determinism

- Run the pilot generator twice in separate output directories.
- Compare SHA-256 hashes.
- Verify 300 unique task IDs and 12 family/tier cells.
- Run oracle answers through native scorers.

### S1 — synthetic golden trajectories

Hand-construct four traces:

1. correct internal answer, no tool;
2. correct tool-assisted answer;
3. wrong answer after tool;
4. timeout/truncation.

Validate schema, component accounting, and post-hoc utility at every lambda.
The checked-in `scripts/validate_elasticity_trace.py --self-test` implements
this first contract smoke and currently passes on four golden trajectories.

### S2 — deterministic scripted policies

Run always-tool, never-tool, and a declared difficulty heuristic. Confirm that
all three see identical tasks and produce expected call curves.

### S3 — one model, tiny paired-price eval

- 2 families × 2 tiers × 8 tasks × 3 prices;
- temperature 0 for the first contract smoke;
- no training;
- inspect every trajectory manually.

### S4 — stochastic teacher gate

- three rollouts per task/price;
- skill/no-skill factor;
- compute switching prevalence, monotonicity, utility frontier, and failures.

### S5 — 32-example conditional SFT overfit

The student should fit price tokens and reproduce obvious action switches. This
tests formatting, masking, tool-action serialization, and checkpoint loading.

### S6 — EIT training smoke

- one GPU, three optimizer steps;
- offline model cache settings;
- resume and artifact-write test;
- record a row in the experiment ledger with config and source log.

Only then launch the Phase 1 matrix.

## Harness smoke decision

Instantiate Reasoning Gym through Verifiers and confirm:

- a frozen task can be scored without a model;
- a Python action can be intercepted and counted;
- multiple reward components survive serialization;
- the same raw episode can be rendered through Inspect AI without changing
  scoring.

If Verifiers composition forces hosted sandboxes or obscures raw accounting,
use NeMo Gym's Reasoning Gym resource server or a thin local adapter for Phase
0. Preserve the trace schema either way.

## OPD implementation constraints

The existing `scripts/opd/` scaffold is valuable plumbing: it already contains
a vLLM teacher client, local judge, KD-forward loss helpers, and a three-step
Qwen smoke history. It does not yet provide a full multi-turn priced
environment or canonical trajectory logging.

For dense OPD:

- use teacher and student with compatible tokenization;
- score student-visited prefixes, not teacher-only prefixes;
- combine task reward with gap/reward-gated teacher supervision;
- preserve per-turn normalization for deeper tool episodes;
- monitor verification/backtracking behavior, not only reward;
- keep bare OPD as a diagnostic arm.

## Configuration discipline

Every run config must freeze:

- task manifest hash and seed ranges;
- model revisions and tokenizer revision;
- prompt/skill hash;
- price train/test sets;
- token and call limits;
- tool container/image hash;
- sampling parameters;
- verifier version;
- code commit.

No paper-facing number is citable without a source JSONL row and completion
audit, following the repository's existing results-lane contract.
