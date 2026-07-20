---
title: M Teacher Failure, Usable MATH Roles, and Second-Source Options
type: hub
tags: [opd, math, teacher-gap, datasets, source-transfer, preregistration]
created: 2026-07-20
updated: 2026-07-20
status: decision recorded; no new training launched
---

# M teacher failure and second-source options - 2026-07-20

## Bottom line

`M` is overloaded in the old arm names. The failure applies to the
**MATH-trained Qwen3-8B teacher**, not to the MATH dataset as a whole.

- The M-trained adapter cannot be merged or supervise `M_M` or `M_O`.
- MATH remains valid for Qwen3-1.7B task-RL, O-teacher-to-MATH OPD (`O_M`),
  M source-heldout evaluation, and frozen MATH-test transfer evaluation.
- The objective-family campaign in
  [[opd-objective-family-expansion-2026-07-20]] consequently already has two
  usable student/evaluation distributions: MATH (`M`) and OpenR1 (`O`). It
  needs one qualified O teacher, not two qualified teachers.
- A replacement teacher source is needed only for a later symmetric
  teacher-source x student-source study or MOPD extension. That is a separate
  campaign, not a rescue of M.

## What actually failed

The M teacher did not crash and did not get worse. Its 100-step training run
completed with informative reward, stable adapter bytes, and clean custody. It
then failed the preregistered *qualification* test on 353 disjoint M records,
with four paired completions per record:

| Quantity | Result |
|---|---:|
| Base Qwen3-8B accuracy | `0.753541` |
| M-trained Qwen3-8B accuracy | `0.764164` |
| Paired difference | `+0.010623` |
| Paired-bootstrap 95% CI | `[-0.002125, +0.024079]` |

The point estimate is a gain of about 1.06 percentage points, but the lower
confidence limit is negative. The registered rule required a strictly positive
point difference **and** a confidence lower bound above zero; ties and
inconclusive intervals fail. The correct claim is therefore:

> The M run did not establish additional teacher skill. It is inconclusive,
> not evidence that M training harmed the teacher.

A direct paired count found only 15 net additional correct completions among
1,412 paired base/trained samples: 1,027 both correct, 296 both wrong, 52
base-wrong/trained-correct, and 37 base-correct/trained-wrong. At the record
level, 26 improved, 18 worsened, and 309 tied.

The later strict audit rescored all 2,824 base and trained completions and found
zero verifier errors. It exactly reproduced the point estimate and interval.
This is what “failed under teacher-favorable sensitivity” means here: there
were no uncertain verifier cases to assign in the teacher's favor, so the most
favorable admissible reassignment is numerically identical and still fails.
The durable artifact account is in
[[opd-math-verifier-recovery-2026-07-20]].

## Why the gate matters

The scientific question is whether a teacher transfers skill acquired from a
named source. Without a demonstrated base-to-trained teacher gap, downstream
student improvement cannot be attributed to that acquired source skill; it may
come from the base teacher, ordinary task RL, sampling noise, or the auxiliary
objective itself. The gate prevents a clean training run or a positive point
estimate from being promoted into that stronger causal story.

M plausibly had little detectable headroom: the base was already at 75.4%, its
training reward averaged about 0.80, only 10 of 100 training prompt groups had
mixed reward, and its gate had 353 records versus 4,585 for O. Those facts help
explain the wide interval, but they do not alter the registered verdict.

“Permanent” is a provenance rule for this campaign: no extra M steps, seed
shopping, threshold changes, alternate prefix, retraining, merge, `M_M`, or
`M_O` can retroactively rescue the sealed result. It is not a universal claim
that no MATH-trained teacher could ever improve under a different study.

## The two experiments must stay separate

### A. Objective-family campaign - ready to retain M and O

Once the successor's fresh strict O teacher passes, use it on both student
sources:

| Student distribution | No-teacher baseline | O-teacher objectives |
|---|---|---|
| MATH (`M`) | task RL on M | `O_M` variants |
| OpenR1 (`O`) | task RL on O | `O_O` variants |

This is the right first campaign for ordinary K1 OPD versus task RL, clipping,
positive-gap gating, bare OPD, and the pinned veRL reference. Its primary
question is objective behavior and transfer across student distributions. It
does **not** claim a symmetric same-source advantage, because there is only one
qualified teacher.

The raw Qwen3-1.7B predecessor support evidence is already encouraging: M
pass@4 `0.620083` with mixed-group fraction `0.198056`; O pass@4 `0.1772` with
mixed-group fraction `0.1273`. Exact-environment support must still be
regenerated on the final campaign commit.

### B. Later source-transfer or MOPD campaign - requires a new teacher source

For a clean symmetric successor, replace the M teacher role with a new source
`C` and use teacher and student sources in `{O,C}`:

| | Student C | Student O |
|---|---|---|
| C teacher | `C_C` | `C_O` |
| O teacher | `O_C` | `O_O` |

MATH can remain a preregistered external transfer target. This restores the
same-source/cross-source estimand without pretending the failed M adapter is a
qualified teacher. A multi-teacher study can then route between two
independently passing teachers and compare against the best single teacher.

## Replacement-source audit

This audit used official dataset cards, papers, and live Hugging Face Dataset
Viewer/API metadata. It did not train a candidate teacher or inspect a new
held-out outcome.

| Candidate | What recommends it | Blocking issue / role |
|---|---|---|
| [[big-math]] | 251,122 RL-ready problems; source/domain fields and a 64-draw Llama-3.1-8B solve rate support outcome-blind difficulty stratification | Best scientific candidate, but EIT's authenticated dry run returned gated-access `403`; unavailable until access is granted before preregistration. It also needs MATH-source exclusion and global O/Numina/eval decontamination. |
| [[deepmath-103k]] | 103,022 accessible MIT-licensed rows with question, final answer, topic, and difficulty; paper reports 95K level-5--9 plus 8K easier problems, rule-verifiable answers, benchmark decontamination, and 82.81K problems unique relative to the compared RL corpora | Strongest immediately accessible qualification candidate. Row-level original-source lineage is not retained, it partially derives from NuminaMath, has no native test split, and must pass our own O/eval collision and label audits. Ignore all R1 traces. |
| [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | Apache-2.0, integer answers, native veRL-like schema | Served corpus is 1,791,700 rows: a live audit found a repeated 17,917-row cycle 100 times. Deduplicate by UUID first; source/difficulty metadata is absent. Operational fallback, not first scientific choice. |
| [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) / [Nemotron RL derivative](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning) | 306K unique parent problems and demonstrated 1.5B/7B training value; compact derivative has 112,867 train rows | AoPS/competition overlap is substantial; compact derivative drops source/difficulty fields; answers include functions, intervals, sets, prose, and approximations. Requires a unique-problem lineage join and extensive verifier audit. |
| [DeepScaleR Preview](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 40,315 rows and direct 1.5B RL precedent | 81.7% of rows have empty solutions, and a live preview exposed at least one answer/solution conflict. Not eligible without an independent label audit. |
| [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) | Large, licensed, metadata-rich | Not independent: OpenR1-Math-220k was generated from NuminaMath-1.5 problems. Use only as a deliberate same-family control. |
| [DeepMind Mathematics](https://github.com/google-deepmind/mathematics_dataset) | Fully procedural, Apache-2.0, exact answers, clean interpolation/extrapolation control | Excellent contamination/implementation control, but short school-level operations likely saturate modern Qwen models and create a large style confound. |
| [MATH-Beyond](https://arxiv.org/abs/2510.11653) | Reserved hard transfer stress test | Only 181 test problems, selected from DAPO/DeepScaleR specifically because <=8B models fail even with large sampling. Do not use for teacher training; quarantine it from any parent-source training pool. |

The [[deepmath-103k|DeepMath paper]] was read beyond its
abstract. Its source pool is primarily Math StackExchange subsets of MMIQC and
WebInstructSub plus NuminaMath-CoT. It decontaminates against named math/STEM
benchmarks using embedding retrieval plus a Llama-3.3-70B paraphrase judge,
then keeps only answers consistent across three DeepSeek-R1 solutions and the
original answer when available. Its own limitations note residual judgement
and multiple-choice questions. In a deterministic 500-row local API screen
(five spaced blocks), all questions/answers were nonempty and 500/500 boxed
answers self-verified under our strict Math-Verify path. This is encouraging
static compatibility, not a full-corpus label audit or a teacher result.

## Outcome-blind qualification rule for C

Do not train candidates until one “works” and then report that winner. The
candidate choice and fallback boundary must be fixed before any trained-teacher
gap is observed.

For an immediately accessible campaign, the provisional order is:

1. DeepMath-103K at Hub commit
   `5cf055d1fe3d7a2eb19719ac020211469736ae44`;
2. no automatic fallback after a DeepMath teacher outcome is seen.

Big-Math may replace DeepMath in the order only if gated access is obtained and
the choice is frozen **before** candidate teacher training. It cannot become a
post-failure rescue.

DeepMath must first pass data-only and raw-model feasibility:

1. pin revision, license, schema, counts, checksums, and a canonical problem
   hash;
2. build one global collision graph across C, O (including upstream Numina
   lineage), M train/test, MATH-500, AIME/AMC, MATH-Beyond, and other frozen
   evaluation sets; quarantine cross-source/evaluation clusters and retain an
   auditable semantic-candidate ledger;
3. require at least 5,000 unique eligible clusters, no skipped collision
   buckets, no unresolved label conflicts, at least 99% gold parseability, and
   zero prompt truncation under one source-independent bound;
4. run a disjoint 256--512-record feasibility surface for raw Qwen3-8B and
   Qwen3-1.7B; require non-floor/non-ceiling 8B reward, student pass@4 at least
   `0.05`, mixed-group fraction at least `0.05`, and verifier-error fraction at
   most `0.001`;
5. after feasibility, freeze deterministic train, teacher-confirmation,
   student, and source-holdout roles before teacher training;
6. once C teacher training begins, a failed gap is terminal for that campaign.

For a later confirmatory O/C 2x2, retrain fresh O and C teachers under the same
commit, data budget, recipe, and predeclared seeds. Use equal-sized teacher-gap
sets rather than repeating the old 353-versus-4,585 power asymmetry. Both
teachers must pass the same independently recomputed, multiplicity-aware gate
before any source-interaction matrix launches.

## Decision

Proceed with the O-teacher objective-family campaign on M and O after its code,
fidelity ladder, fresh O gate, and exact support gates are complete. Do not wait
for a second teacher source: MATH already supplies the second benchmark.

In parallel, qualify DeepMath only as infrastructure and raw-model feasibility
for a later source-transfer/MOPD successor. No candidate teacher training and no
new OPD arm was launched during this audit.

## Links

[[opd-objective-family-expansion-2026-07-20]] -
[[opd-math-verifier-recovery-2026-07-20]] - [[opd-math-source-transfer]] -
[[verl-opd-trainer]] - [[mopd-multi-teacher]]
