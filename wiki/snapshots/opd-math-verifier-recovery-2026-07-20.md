---
title: OPD Math Verifier Recovery Boundary
type: snapshot
tags: [opd, math, eit, verifier, recovery, custody]
created: 2026-07-20
updated: 2026-07-20
status: recovery substrate validated locally; original four-arm successor unlaunched and superseded at design stage
supersedes: opd-math-scientific-cutover-2026-07-18
---

# OPD math verifier recovery - 2026-07-20

> [!NOTE]
> This snapshot preserves the exact recovery and four-arm plan as it stood.
> Before that successor was sealed or launched, the comparison expanded to
> [[opd-objective-family-expansion-2026-07-20]]. Reuse the strict verifier,
> M-negative audit, support, O-teacher, and custody substrate; do not execute the
> four-arm preregistration as the active scientific campaign.

## Bottom line

The OPD execution path works, but the old `ae90bc7` evaluation authorization
artifacts used a silently lossy Math-Verify policy. The O teacher's training
weights are clean and its positive skill-gap conclusion survives an adversarial
reanalysis, yet the old O support/gap/held-out gates are not reused as formal
authorization. A successor commit repairs the verifier contract and will run a
fresh, single-commit O-only campaign: two task-RL baselines plus `O_M` and
`O_O`. M remains a genuine teacher-gap failure and is not rescued.

This snapshot is the durable handoff. It distinguishes historical numerical
evidence, valid plumbing evidence, and results that still must be regenerated.

## What is known from the sealed predecessor

### Teacher gates

The 100-step M teacher failed its preregistered gate on 353 records:

- base accuracy `0.753541`;
- trained accuracy `0.764164`;
- paired delta `+0.010623`;
- paired-bootstrap 95% CI `[-0.002125, +0.024079]`.

The point estimate was positive, but the lower bound was not. M cannot be
merged and neither `M_M` nor `M_O` may be launched in this campaign.

The old O gate reported, on 4,585 records and four samples per record:

- base accuracy `0.264667`;
- trained accuracy `0.275900`;
- paired delta `+0.011232`;
- paired-bootstrap 95% CI `[+0.006543, +0.016085]`.

That conclusion is numerically robust but the old gate is formally superseded
because its scorer silently converted verifier failures into ordinary wrong
answers.

### Exact strict reward audit

The root defect was `math_verify.verify(..., raise_on_error=False)`, both in the
student helper and TRL's teacher reward. `ValueError`, `KeyError`, and
`TimeoutException` were therefore reduced to `False`; notably,
`TimeoutException` derives from `BaseException` rather than `Exception`.

Strict replay established:

| Surface | Strict result |
|---|---|
| O teacher training `107419` | 400/400 samples replayed; 119 correct, 281 incorrect, 19 mixed groups; zero verifier errors and zero reward mismatches |
| O gap base | 7/18,340 initially uncertain, all stored as ordinary zero/incorrect |
| O gap trained | 7/18,340 initially uncertain, including one deterministic SymPy `Unequality` `KeyError` |
| Student M support | 4 hidden failures among 8,644 samples |
| Student O support | 5 hidden failures among 8,644 samples |
| M/O baseline training traces | zero hidden failures in either 400-sample trace |
| M baseline held-out | zero hidden failures among 1,480 samples |
| O baseline held-out | one hidden `Unequality` failure among 1,480 samples |

Three fixed 5-second attempts reduced the O gap uncertainty to five base and
five trained samples (`0.0273%` per arm). Four load-sensitive cases resolved
consistently to wrong answers; nine comparisons remained timeouts and one
remained a deterministic `KeyError`. Longer externally bounded checks resolved
five more timeouts as wrong after 22--42 seconds, while four still exceeded 60
seconds. No uncertain completion ever resolved correct.

For the formal sensitivity, however, every uncertain base completion was set
to correct and every uncertain trained completion to wrong. Even that monotone
worst case retained an O delta of `+0.010851` with 95% CI
`[+0.006052, +0.015703]`. This supports the scientific premise that O learned a
skill gap; it does not rehabilitate the old gate bytes.

Audit anchors on EIT:

- scratch audit root:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/audits/strict_reward_ae90bc7_20260720T174500Z`;
- strict audit script SHA-256:
  `f6f64c10482040e852b9743400eee3310b8e1bb5ce82742bfc815e4299516f9b`;
- strict result SHA-256:
  `496a2f2830c57cfd3b7d51ed235b2a66377123f6733f397fa60e31b9aa4e8414`;
- targeted result SHA-256:
  `0a2465f9f96fb964e54dffb39b6753307045c8661e978c5dd917871bbb0825be`;
- allocated jobs `108275`, `108276`, and `108277` all completed `0:0`.

### One-step OPD diagnostic

The first corrected-HOME attempt `107961` failed before training because the
launch used `/home/hiqbal` rather than `/home/compute/hiqbal`. No optimizer
step or OPD result existed, and the failure remains sealed at
`campaigns/O_M_one_step_diagnostic_terminal_failure_ae90bc7_107961`.

Retry `108244` is a valid full-custody plumbing result:

- one O-teacher-to-M-student step, four student samples and 2,048 scored tokens;
- all four task rewards zero, so no task-quality evidence;
- total loss `0.00037958`, task loss zero, score-function surrogate
  `0.0379580`, sampled K1 value `0.1050842`;
- positive-gap fraction `0.151855`, pre-clip gradient norm `0.0034653`;
- a real LoRA update, with all 65,536 inspected LoRA-B entries nonzero;
- stable final adapter tree
  `818b2edcfc3d327a5059bbb1a60425c0251fc17cad8629324e1e86afd2b0854e`.

This proves exact-token teacher scoring, the OPD loss path, gradient flow,
parameter movement, and artifact promotion. It does not show improvement. Its
terminal audit must be copied into an immutable campaign seal before the new
launch.

## Successor verifier and analysis contract

The recovery patch makes six linked changes:

1. **Training is zero-tolerance.** Gold parsing, candidate parsing, or symbolic
   verification errors abort before a reward vector and optimizer step. Both
   parser and verifier request `raise_on_error=True`, and
   `TimeoutException` is caught explicitly. TRL reward verification refuses an
   unbounded non-main-thread execution path.
2. **Evaluation retains bounded uncertainty.** Candidate-side failures receive
   three fixed strict attempts. Exhausted errors are visibly stored at reward
   zero with type, stage, message, and the complete retry history. Gold-side
   failures still abort. The full merged error fraction must not exceed
   `0.001`.
3. **Authorization is worst-case robust.** Teacher-gap authorization assigns
   every uncertain base reward to one and every uncertain trained reward to
   zero. Student support removes error-induced mixedness. Held-out contrasts
   classify effects using a binary worst-case paired-bootstrap envelope, not
   the zero-mapped point interval.
4. **Timeout replay is honest.** A later merge/gate replay may resolve a stored
   timeout without invalidating the bounded-unknown row. A determinate stored
   reward must still independently replay; three unresolved attempts make that
   audit fail rather than silently changing its label.
5. **Student trajectory custody is analyzable.** Schema-v2 training traces
   store every student and teacher token log-probability on the sampled student
   trajectory. NLL, mean/absolute gap, min/max gap, and positive-gap fraction
   are reconstructed from the serialized token values.
6. **Intermediate loss claims are independently replayable.** The task loss,
   score-function surrogate, sampled K1 value, gap mean, reward/mixedness
   summaries, token count, and total loss are reconstructed in pure Python from
   the exact sample arrays. Only parameter-space gradient norms and provenance
   of the model-produced log-probabilities remain outside arithmetic replay.

The final local repository suite passes 329 tests. It includes real
regressions for the observed `nan/zoo` `ValueError` and `Unequality` `KeyError`,
parser/verifier strictness, the `BaseException` timeout path, retry exhaustion,
global error-cap enforcement, exact trajectory arrays, coherent teacher-array
tampering with stale step metrics, the four-arm O-teacher readout, and
gate/result custody.

The legacy M selection context now uses a dedicated negative-only compatibility
audit rather than passing the old schema through the new teacher-gate parser.
An independent read-only exercise rebound the complete historical artifact
graph and rescored all 2,824 base/trained completions with the current strict
evaluator. It found zero verifier errors and exactly reproduced the negative
result: delta `+0.010623`, 95% CI `[-0.002125, +0.024079]`, gate failed. The
audit hardcodes no merge and no `M_M`/`M_O` authorization and reruns the live
replay whenever it is consumed.

## New campaign boundary

The successor campaign will retain the stronger same-commit invariant rather
than introduce a special legacy-teacher exception. It will therefore retrain
only the O teacher under the strict scorer. The old O weights remain useful as
a reproducibility target; M is not retrained.

Ordered execution:

1. commit and push the recovery code and this snapshot;
2. fast-forward the clean EIT clone once and create read-only train/serve
   freezes for the exact commit;
3. seal the old `108244` plumbing audit and a successor operational launch
   ledger; the final four-arm preregistration cannot be sealed until its O
   teacher and M/O support identities exist;
4. regenerate full raw-student support for M and O under the repaired exact-v2
   evaluation contract;
5. run one strict 100-step O teacher and require informative training reward;
6. run paired 32-record base/trained timing prefixes, build the dual-timing
   successor plan from the slower `ElapsedRaw`, retain its registered
   five-shard minimum, then evaluate raw and trained O on the complete
   4,585-record O gap surface and require the worst-case teacher-gap gate;
7. merge only the passing O teacher and run a new one-step schema-v2 custody
   diagnostic;
8. only after the diagnostic is clean, seal the student-outcome-blind four-arm
   preregistration with the selected O teacher's complete stable identity, both
   support identities, exact stable run/adapter/gate/output paths, and then
   launch matched 100-step `baseline_M`, `O_M`, `baseline_O`, and `O_O` runs;
9. evaluate every arm on its exact 370-record source holdout and publish an
   outcome-blind O-only paired readout with 10,000 bootstrap draws, seed zero,
   lexicographic record-ID order within each source, M-then-O draws from one
   RNG stream, Bonferroni intervals for the two co-primary contrasts, and
   verifier-error worst-case envelopes.

Every primary student arm uses a predeclared filesystem-safe run ID rather than
a Slurm-derived path. The readout publishes read-only JSON, Markdown, and a
checksum bundle manifest or cleans up the partial bundle on failure. The O
selection condition is known when the four-arm preregistration is sealed;
student-arm outcomes are not. This is recorded operator custody, not
cryptographic proof of chronology.

The tracked result builder now implements that exact four-arm readout directly;
it never requires the prohibited `M_M` or `M_O` artifacts. Each O arm
transitively recomputes the passing O teacher gate and its immutable merged
checkpoint before the conditional analysis can run. The read-only hashed
preregistration and launch ledger are recorded operator custody; they do not
cryptographically prove prelaunch chronology.

The resulting study is explicitly not the original six-arm factorial matrix.
It answers whether the selected, independently skill-improved O teacher helps
or harms a matched task-RL student on M and O. Any report must state that the O
teacher was selected because O passed while M did not.

## Stop and iteration rules

- Never launch `M_M` or `M_O` from this boundary.
- If fresh M or O raw-student support fails, do not launch that target's
  baseline or OPD arm.
- If strict O teacher training aborts or its worst-case gap gate fails, preserve
  the negative result and launch no OPD arm.
- A verifier-error fraction above `0.001` invalidates the whole merged
  evaluation; do not hide it through shard-level filtering.
- A one-step finite loss, gradient, update, or checkpoint remains plumbing.
  Improvement requires the paired held-out comparison and uncertainty.
- Operational failures may be corrected only through a recorded successor
  launch ledger that preserves failed attempts. Scientific gates and thresholds
  are not relaxed in response to outcomes.

## Links

[[opd-math-source-transfer]] · [[opd-math-scientific-cutover-2026-07-18]] ·
[[opd-math-eit-handoff-2026-07-18]] · [[opd-distillation]]
