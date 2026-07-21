---
title: DeepMath-103K
type: source
tags: [opd, math, dataset, rlvr, difficulty, decontamination]
created: 2026-07-20
updated: 2026-07-20
status: pinned raw bytes/schema verified on EIT; collision and feasibility gates pending; teacher untrained
url: https://arxiv.org/abs/2504.11456
dataset: https://huggingface.co/datasets/zwhe99/DeepMath-103K
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2504.11456.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/DeepMath
authors: He et al.
year: 2025
---

# DeepMath-103K

## TL;DR

DeepMath-103K is the strongest immediately accessible candidate for a future
second teacher source. The pinned Hugging Face revision exposes 103,022 rows
with question, final answer, difficulty, topic, and three DeepSeek-R1 solution
fields. We would use only the problem and final answer; importing the R1 traces
would confound source-RL with solution distillation.

This is a qualification candidate, not a qualified teacher. A deterministic
500-row screen found nonempty questions/answers and 500/500 boxed answers
self-verifying under our current strict Math-Verify path. That only checks
static schema/verifier compatibility. It does not establish full-corpus label
quality, Qwen reward support, a teacher skill gap, or student improvement.

## Raw intake checkpoint - job 108481

The outcome-blind plan at
`configs/opd_math/deepmath_qualification_plan.json` was committed before raw
intake. EIT Slurm job `108481` downloaded the ten source Parquet shards at the
pinned dataset revision and completed with exit `0:0` in 66 seconds. The
wrapper and an independent reopen both verified:

| Field | Verified value |
|---|---:|
| Source bytes | `2,136,106,260` |
| Rows | `103,022` |
| Shards | `10` |
| Columns | exact seven-field source schema |
| Raw identity manifest SHA-256 | `d015623e4e86073de4ba76a517746ac4561e1a384035e99ffb9b466d1b88f0da` |
| Download custody SHA-256 | `ce0599487f09d1b6c17a2a86319afaeb48e5a6492eb1e89da9cbce17ce7d8f02` |

Large files remain only on EIT under
`/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/deepmath_C/5cf055d1fe3d7a2eb19719ac020211469736ae44`.
Both receipts explicitly set `teacher_training_authorized=false` and
`scientific_use_allowed=false`. This establishes raw-byte and schema custody
only. It does not satisfy the global collision graph, label/parseability,
prompt-bound, raw-model feasibility, role-freeze, or teacher-gap gates.

## Construction and caveats

The paper reports roughly 95K difficulty-level 5--9 questions plus 8K easier
questions. Primary source pools are the Math StackExchange portions of MMIQC
and WebInstructSub plus NuminaMath-CoT. Questions are decontaminated against
named benchmarks using embedding retrieval followed by a Llama-3.3-70B
paraphrase judgement. Answers are retained when three DeepSeek-R1 solutions
and the original answer, when available, agree under the rule verifier.

This is stronger curation than a raw scrape, but not complete independence for
our experiment. Row-level original-source lineage is absent, part of the data
descends from NuminaMath, and OpenR1 also descends from NuminaMath-1.5. The
paper reports 82.81K problems unique relative to the RL corpora it compares,
but our exact O/M/evaluation inventory was not one of its registered study
objects. We therefore need our own global exact and semantic collision ledger.

The repository also records that 48 answer-revealing hints were corrected in
May 2025. That is a reason to pin the current dataset revision and hash
canonical problem text rather than relying on the paper-era bytes.

Anchors read: source analysis and curation pipeline on PDF pp. 3-5; difficulty,
decontamination, and answer-consistency construction on pp. 6-9; limitations
and examples in the appendices.

## Outcome-blind qualification before teacher training

1. Pin dataset revision
   `5cf055d1fe3d7a2eb19719ac020211469736ae44` and record row hashes.
2. Exclude collisions with O/Numina lineage, M train/test, MATH-500,
   AIME/AMC, MATH-Beyond, and every frozen held-out set.
3. Require at least 5,000 eligible unique clusters, at least 99% gold
   parseability, zero unresolved label conflicts, and zero prompt truncation.
4. On 256--512 disjoint records, require non-floor/non-ceiling raw Qwen3-8B
   reward, Qwen3-1.7B pass@4 at least `0.05`, mixed-group fraction at least
   `0.05`, and verifier-error fraction at most `0.001`.
5. Freeze train, teacher-confirmation, student, and source-holdout roles before
   teacher training. Once training starts, a failed teacher gap is terminal.

## Version and custody

- Paper PDF SHA-256:
  `9aa8b416b125bb8ab1c16160c1da9cd01f803c0ee16ecabeb809a4fd9bacf171`.
- Official repository: https://github.com/zwhe99/DeepMath.
- EIT repository checkout pinned at
  `0d97187a17b0e1b54ecef109c43a904b50c99506`.
- Dataset revision screened:
  `5cf055d1fe3d7a2eb19719ac020211469736ae44`.

## Links

[[big-math]] - [[opd-m-teacher-clarification-and-source-options-2026-07-20]] -
[[opd-objective-family-expansion-2026-07-20]] - [[opd-math-source-transfer]]
