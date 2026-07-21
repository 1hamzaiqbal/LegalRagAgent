---
title: DeepMath Global Inventory Attempt 108510
type: snapshot
tags: [opd, deepmath, data-quality, provenance, failure]
created: 2026-07-20
updated: 2026-07-20
status: sealed failed data-materialization attempt; no training authorization
---

# DeepMath inventory attempt 108510

Slurm job `108510` failed before it could write a global inventory manifest.
It successfully materialized and sealed all `103,022` DeepMath candidate rows
and all `225,129` OpenR1 rows, then stopped while normalizing the pinned
`AI-MO/NuminaMath-1.5` lineage source. At least one upstream lineage row has an
empty `problem` value. The original materializer rejected it with
`ValueError: required source value is empty`; it did not delete the row,
invent a prompt, or continue with a smaller source count.

This is an outcome-blind source-data contract failure, not a DeepMath
qualification outcome and not a training result. No optimizer step, teacher
training, OPD training, evaluation, or scientific authorization occurred.
The failed root and its `64,142,612`-byte partial lineage Parquet remain on
persistent EIT storage. Exact hashes and Slurm custody are recorded in
`evidence/july_2026/deepmath_inventory_failure_108510.json`.

The successor contract retains every upstream row and adds an explicit
`problem_missing` boolean. Empty prompts are never repaired at ingestion.
Candidate-C missing prompts have a preregistered maximum of zero and are
quarantined by the data audit. Missing prompts in external collision-reference
sources remain visible in per-source quality counts. Because the schema,
inventory-plan bytes, and Git commit changed, the successor must use a new EIT
root; the sealed C/O files from job `108510` cannot be silently reused.

## Immutable paths

- failed inventory root:
  `/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/deepmath_inventory_v1`
- original stdout:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_C_inventory_108510.out`
- persistent failure receipt namespace:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/campaigns/deepmath_inventory_failure_7cb4ed7_108510`

## Boundary

This data-only retry does not reopen the historical M teacher. M remains a
student-rollout/evaluation distribution in the O-teacher objective-family
campaign and an external transfer target in a later O/C study only.

## Links

[[opd-program-goal-2026-07-20]] - [[deepmath-103k]] -
[[opd-math-source-transfer]]
