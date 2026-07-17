---
title: Benchmark Audit and EDA
type: review
tags: [benchmarks, eda, reasoning-gym, agent-distillation]
created: 2026-07-17
updated: 2026-07-17
status: setup in progress
---

# Benchmark audit and EDA

## Core benchmark recommendation: Reasoning Gym

Reasoning Gym is a better first substrate than legal QA, MedQA, or a fixed
agent benchmark because it provides:

- procedural, seed-disjoint task generation rather than a small static test;
- adjustable difficulty;
- algorithmic verification;
- more than 100 task generators across several domains;
- an existing Verifiers integration;
- Apache-2.0 code custody.

This addresses the prior legal benchmark problem: we do not need to infer a
new behavior from an underspecified golden passage or a weak retrieval corpus.

## Pilot task families

| Family | Why Python has value | Difficulty controls | Verifier caveat |
|---|---|---|---|
| `prime_factorization` | Easy cases can be internal; hard composites reward computation | value ranges up to 10,000 in built-in curriculum | Native scorer gives 0.5 for correct product with non-prime factors; log exact success separately |
| `bitwise_arithmetic` | Nested signed bitwise expressions are directly executable | built-in difficulty 1–10 | Native verification is binary; prompt typo is cosmetic |
| `countdown` | Search over expressions becomes valuable as number count grows | number count, target range, source value range | Native scorer has 0.01/0.05 shaping; threshold exact success at 1.0 |
| `shortest_path` | BFS is reliable; small grids remain human-solvable | rows and columns | Native scorer gives 0.5 for valid non-shortest paths; exact success means score 1.0 |

The deterministic generator in `scripts/generate_reasoning_gym_pilot.py`
creates 25 examples per family × tier cell (300 total) by default. It is EDA,
not a final train/test split.

The first EIT smoke exposed a benchmark pathology: `countdown` can raise
`ValueError` after exhausting its internal 100-attempt expression search for
some deterministic hard configurations. The generator now scans a fixed
candidate-index budget, skips those failures deterministically, records them
per cell, and fails closed if it cannot collect the requested sample count.
This failure rate must be reported before retaining `countdown`.

The corrected smoke completed 300 unique rows. Easy factorization skipped
three duplicate questions; hard `countdown` required 26 indices because one
generation failed (3.8% of examined candidates). Median question length ranged
from 117 characters for easy factorization to 1,668 for hard shortest path;
hard bitwise arithmetic had a 958-character median and 1,538-character maximum.
These long cells need token-budget and prompt-length checks before training.
The pilot was regenerated independently with byte-identical JSONL and summary
hashes. The frozen JSONL SHA-256 is
`1cfe214d4fd23deae219d34b161b4f1811f3910b53fd4ccd71880313de07e474`.

The native-scorer audit found 300/300 oracle answers score `1.0`, with 300
unique IDs and normalized questions. It also confirmed why exact success must
be separate: empty answers receive `0.01` on all 75 factorization and all 75
countdown tasks, and 63 composite factorization cases award `0.5` when the
model returns the unsplit composite number. The audit therefore enforces
`task_success_exact = int(native_score == 1.0)` while retaining native reward
as a diagnostic.

A one-example-per-cell manual review found coherent questions and answers, but
also two design warnings: hard bitwise expressions can exceed 1,000 characters
and may make Python uniformly dominant, while hard shortest-path prompts are
long mainly because they inline a 20–25 row grid. The teacher-switching gate
should compare bitwise difficulty 6 versus 8 and should drop any tier on which
always-tool dominates. The bitwise prompt's inherited “hexidecimal” typo is
cosmetic but should be fixed only in a versioned prompt adapter, not by silently
changing frozen pilot text.

## Required pre-model EDA

For every family/tier:

1. check duplicate task IDs and duplicate normalized questions;
2. verify determinism by regenerating from the same seed and hashing output;
3. run each oracle answer through the native scorer;
4. run empty, malformed, and known reward-hacking answers through the scorer;
5. inspect prompt/answer length distributions and generation failure rate;
6. confirm that the Python tool can solve a random sample under the same
   sandbox restrictions the model will receive;
7. inspect 25 examples manually across all cells;
8. freeze train/validation/test seed ranges before teacher rollout generation.

## Agent Distillation trajectory EDA

Two public teacher-trajectory files were downloaded to EIT. Observed directly
from their parquet files:

| File | Rows | Subsets | Median log characters | Important issue |
|---|---:|---|---:|---|
| baseline | 1,928 | 597 HotpotQA; 547 math-hard; 784 math-medium | 22,330 | scores encoded as `1`, blank, or `True`; math cost accounting differs by subset |
| first-thought prefix | 2,113 | 608 HotpotQA; 671 math-hard; 834 math-medium | 23,020 | same schema inconsistency; many zero cost fields |

No fields are missing, but the traces appear curated as successful training
trajectories: all Hotpot scores are `1`, math-medium/hard use `True` or blank,
and cost conventions differ. They are useful to reproduce Agent Distillation
formatting and SFT baselines. They are **not** a clean held-out benchmark, a
negative-trajectory set, or evidence of a price-response policy.

## External validation benchmarks

- **StableToolBench cost augmentation:** use only after the one-tool result.
  It directly tests changing tool prices but adds multiple APIs, stochastic
  tool behavior, and judging complexity.
- **SkillsBench:** useful for testing whether a learned metapolicy survives
  real skill/harness heterogeneity. It is too broad for causal Phase 0.
- **Agent Distillation code/retrieval tasks:** useful for transfer comparability,
  but regenerate paired-price teacher trajectories rather than treating the
  public successful traces as evaluation data.
- **Legal retrieval:** a later structured-cost stress test if we can construct
  a sound corpus and authority-aware verifier. It is not required for the core
  contribution.

## Split proposal

- Training: generator seeds `100000–109999`.
- Validation: `200000–201999`.
- Test: `300000–303999`.
- Development smoke: the dated `2026071700` pilot only; never report it as test.
- Use unseen item seeds at every price; paired-price evaluation repeats the same
  test task IDs, not the same stochastic rollout.

Before freezing these ranges, confirm that each generator uses `seed + index`
and that its configuration does not introduce a hidden nondeterministic source.

## Benchmark selection gate

Retain a family only if the teacher exhibits both internal and tool-assisted
success on different instances and a nontrivial price-dependent switch. Drop
families where Python always dominates, never helps, or the verifier can be
gamed. The paper needs two or three clean families, not a large benchmark zoo.
