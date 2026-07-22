---
title: OPD verifier score-ledger boundary
date: 2026-07-22
status: applied; independently reconstructed; merge rearm eligible
tags: [opd, math, verifier, evaluation, custody, score-ledger]
---

# OPD verifier score-ledger boundary - 2026-07-22

## Decision in one sentence

Replace promotion-time symbolic rescoring with a **score-once, attest-many
ledger**, restrict the symbolic teacher gate to a gold-only
symbolic-eligible population, retain scorer failures as bounded unknowns, and
allow corrections only through immutable sample-hash-bound adjudications.

This is a post-hoc recovery boundary. It does not rewrite the failed canonical
merge, its source evaluations, or either predecessor gate.

## Problem

The gate at commit `d89ba3d7be728d9ee3197f37d8a8836a4a9640c5`
replayed Math-Verify over every completion whenever the gate was consumed.
That is not a deterministic custody check:

1. Math-Verify 0.9.0 bounds parsing and comparison with `signal.alarm()`.
2. A verdict can therefore change from determinate to timeout under a different
   machine load even when the completion, gold, code, and environment hashes are
   unchanged.
3. Gate jobs `123419` and `123420` finished with identical passing bytes, but a
   later merge replay stopped on base row 3910. Manual algebra showed the stored
   `incorrect` label was a false negative and the later replay was inconclusive.
4. After a single disclosed adjudication, canonical merge job `123666` stopped
   on base row 17337. Its answer, “essential singularity at z=1,” is also
   mathematically correct. The registered gold is the concatenated prose string
   `z_{0}=1isanessentialsingularityofthefunctionf(z)`, which Math-Verify parses
   as a product of single-letter symbols rather than a semantic label.

The two failures expose different defects:

- **measurement instability:** a timeout-bounded scorer cannot also serve as a
  byte-reproducible promotion attestation; and
- **construct mismatch:** syntactic parser success does not establish that a
  heterogeneous prose/categorical answer is suitable for symbolic equivalence.

Increasing timeouts or replaying until one verdict passes would hide both
problems and make promotion depend on retries. Manually whitelisting every
failure would be outcome-adaptive and unscalable.

## External grounding

- [Math-Verify](https://github.com/huggingface/Math-Verify/tree/ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b)
  explicitly uses different extraction configurations for mathematical and
  string answers, states that the gold representation guides asymmetric
  comparison, and implements timeout protection with Unix signals. Its own
  README warns that answer configuration must follow the dataset's gold type.
- [Math-Verify issue 44](https://github.com/huggingface/Math-Verify/issues/44)
  records concrete cases where answer extraction produced an incorrect parsed
  result. Parser success must therefore not be treated as proof of semantic
  scoreability.
- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai/tree/86ca6dfc8701ec684e96bb1ee5992a45ea88edcd)
  separates sample errors/retries from aggregate scoring and supports unscored
  samples rather than forcing every scorer failure into an ordinary wrong
  answer.
- [Rethinking Math Reasoning Evaluation](https://arxiv.org/abs/2604.22597)
  directly studies symbolic-evaluator rigidity. Its proposed semantic path uses
  independent question answering, ground-truth validation, repeated judgments,
  and majority voting. That is a reasonable future path for excluded semantic
  answers, but it is not silently substituted into this symbolic gate.

Pinned source copies live on EIT at:

- `/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/Math-Verify`
- `/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/lighteval`
- `/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/inspect_ai`
- `/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/lm-evaluation-harness`
- `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/2604.22597-rethinking-math-reasoning-evaluation.pdf`

## New measurement contract

### 1. Define the symbolic population without looking at model outputs

Eligibility uses only the registered gold `answer` and `solution`. A record is
excluded from this symbolic metric when its answer contains textual LaTeX
without a registered string ontology, a run of at least five uncommanded ASCII
letters, or at least eight alphabetic characters with more letters than digits.

The rule never sees arm identity, completion text, stored reward, or the
base-versus-trained difference. Exclusion changes the estimand: the result is a
**symbolic-eligible verifier-aligned reward gap**, not accuracy on all O tasks.

The minimum eligible coverage is fixed at `0.75`. The existing scientific
minimum of 200 paired records remains in force.

### 2. Score once, attest many

Evaluation-time sample rewards are measurement observations. The ledger binds
each observation to record ID, sample index, completion SHA-256, source sample
SHA-256, task SHA-256, evaluator artifacts, and the predecessor gate. Promotion
reconstructs these rows and statistics from sealed bytes but does **not** invoke
Math-Verify.

A later rescore is an audit measurement. It may discover disagreement, but it
does not overwrite the original observation or become a retry-until-pass path.

### 3. Keep inconclusive scores unknown

`verifier_error_zeroed` remains zero only for the descriptive point estimate.
Scientific authorization additionally assigns every base-side unknown to 1 and
every trained-side unknown to 0, then requires both the strict delta and the
bootstrap lower bound to remain positive. The inverse assignment is retained as
the best-case envelope.

### 4. Adjudications are append-only evidence, not row edits

An adjudication must bind the exact task, base and trained sample files, arm,
record ID, sample index, completion hash, stored score, stored status, verdict,
reasoning hash, and post-hoc disclosure. It changes the effective ledger score
without changing source evaluation bytes. Duplicate or partially bound
adjudications fail closed.

The already sealed row-3910 adjudication is the only adjudication admitted in
this recovery boundary. Row 17337 is not separately whitelisted; its entire
record is excluded by the gold-only population rule.

### 5. Separate future semantic evaluation

Excluded answers require a registered semantic scorer. A future scorer should
be calibrated on a blinded human-labeled set and follow the independent-solve,
ground-truth-validation, repeated-vote structure from arXiv:2604.22597. Until
that exists, excluded records contribute coverage statistics but not a symbolic
accuracy claim.

## Frozen implementation

- Branch: `codex/opd_verifier_ledger_v2`
- Implementation commit: `990e4b829cd4cc13bc9d21b6914113034b83786c`
- Builder/validator: `scripts/opd_math/score_ledger.py`
- Merge dispatch: `scripts/opd_math/merge_adapter.py`
- Focused tests: `tests/test_opd_score_ledger.py`

The legacy `teacher_gap_v1` path remains available for provenance and is not
rewritten. Only a new `teacher_gap_score_ledger_v1` gate uses the new contract.

## Authorization boundary

This decision authorizes exactly these next actions:

1. Build one immutable score-ledger bundle from the two sealed O evaluation
   surfaces, the old passing O gate, and the row-3910 adjudication.
2. Independently reconstruct the bundle without invoking Math-Verify.
3. Record coverage, exclusions, point estimate, bootstrap interval, worst-case
   uncertainty envelope, and pass/fail result.

It does not yet authorize a model merge or OPD training. A merge may be rearmed
only if the new gate passes, byte-identical independent reconstruction passes,
the implementation checkout is clean at the frozen commit, and a successor
record explicitly authorizes one new merge attempt under the ledger gate.

## Result

The frozen policy was applied once on EIT as CPU Slurm job `124779`. The job
completed `0:0` and published a read-only bundle. A separate process in job
`124781` reconstructed every ledger row and statistic from the sealed source
bytes without invoking Math-Verify. It completed `0:0`; its reconstructed gate
SHA-256 exactly matched the published gate.

| Quantity | Result |
|---|---:|
| Registered paired records | 4,585 |
| Symbolic-eligible paired records | 4,434 |
| Excluded records | 151 |
| Eligible coverage | 96.7067% |
| Base mean reward | 27.1876% |
| Trained O-teacher mean reward | 27.8360% |
| Paired difference | +0.6484 pp |
| Paired 95% bootstrap CI | [+0.1804, +1.1164] pp |
| Eligible base / trained verifier errors | 2 / 4 |
| Worst-case paired difference | +0.6371 pp |
| Worst-case 95% bootstrap CI | [+0.1691, +1.1107] pp |

All six registered requirements passed, including minimum coverage and the
worst-case positive bootstrap lower bound. The new gate therefore authorizes a
single scientific O-teacher merge **after** a separate successor rearm record;
it does not authorize student training by itself.

The result changes the estimand and must be reported as a post-hoc
**symbolic-eligible verifier-aligned reward gap**. It is not accuracy over all O
tasks, and it is not evidence that OPD improves the student. The 151 exclusions
were determined solely from registered gold answers. One previously sealed,
hash-bound adjudication affects one base sample; no source evaluation byte was
edited.

### Immutable evidence

- Intent:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/conductor/methodology-intent-score-ledger-20260722T200824Z.json`
  (`716df9e2...65e50`)
- Bundle:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/score_ledgers/O_gap_ac65062_v1`
- Gate SHA-256: `18e5faf7c391219dd61b9670e5226d23009d42d2d5084bac93c267994316b5fd`
- Ledger SHA-256: `2b0ddecb9c19ce20ac10c03461a3e672137502ccb7553e4435d37b535651034f`
- Build / independent reconstruction jobs: `124779` / `124781`
- Compact tracked receipt:
  `evidence/july_2026/opd_verifier_score_ledger_O_gap_ac65062_v1.json`

## Links

[[opd-math-verifier-recovery-2026-07-20]] ·
[[opd-objective-family-expansion-2026-07-20]] ·
[[opd-distillation]]
