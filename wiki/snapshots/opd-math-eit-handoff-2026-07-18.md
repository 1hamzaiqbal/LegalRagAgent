---
title: OPD Math EIT Plumbing Handoff
type: snapshot
tags: [opd, math, eit, source-transfer, plumbing, custody]
created: 2026-07-18
updated: 2026-07-18
status: bounded plumbing validated; science gated
---

# OPD math EIT plumbing handoff - 2026-07-18

## Bottom line

The bounded M/O OPD-math path works on one EIT A100-SXM4-80GB GPU:

1. exact pinned Qwen and dataset snapshots are visible offline;
2. the isolated train and serve environments pass GPU preflight;
3. a deliberately partial 64-row-per-split data audit produces disjoint role
   files and fails closed for scientific use;
4. one-step teacher GRPO smokes run for both M and O;
5. a live Qwen3-8B vLLM teacher passes the tokenizer and exact-token scoring
   contracts; and
6. Qwen3-1.7B completes one `task_rl_k1_gap` optimizer step on both the M and
   O student sources, observes a finite nonzero gradient and parameter update,
   and promotes a stable LoRA adapter.

The main plumbing ran at
`35b6c23ca066fa4a03241e763ca10d0537c43de3`. An independent artifact audit
then caught a manifest-label bug: ungated smokes correctly skipped local
process custody, but the completion manifest called the skipped requirement
validated. Commit `80836d4a6f8a7c508cd53761d53a079e5e7345ae` separates
`local_server_process_binding_required` from
`live_local_server_process_binding_validated`. Regression job `106687`
exercises the corrected semantics below.

This is **plumbing evidence only**. The audit data are non-scientific, the
student smokes used the raw teacher rather than a trained-and-gated teacher,
both two-sample task-reward groups were all wrong, and no held-out task
evaluation was run. No full teacher or student training job was launched.

## Reproducibility anchors

| Object | Exact anchor |
|---|---|
| Branch | `codex/opd_math_pipeline` |
| Main plumbing execution | `35b6c23ca066fa4a03241e763ca10d0537c43de3` |
| Custody-label correction | `80836d4a6f8a7c508cd53761d53a079e5e7345ae` |
| EIT checkout | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math` |
| Train environment | `/engrfs/project/jacobsn/hiqbal/envs/opd_math_train` |
| Serve environment | `/engrfs/project/jacobsn/hiqbal/envs/opd_math_serve` |
| Audit data | `/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/audit64_35b6c23` |
| Run artifacts | `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math` |
| Model cache | `/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache` |
| Train freeze SHA-256 | `69edfc8f75927379124c88391578097fed7fa66d6e779df593b297cbbf2ada06` |
| Serve freeze SHA-256 | `f36743fed4fe4c1b8a7fefaac39d9ce8a6f441897920c75ca11de8c487bf443d` |

The commit-specific freezes live under
`artifacts/legalrag/opd_math/environment_freezes/35b6c23ca066fa4a03241e763ca10d0537c43de3/`.
The EIT checkout was clean during every smoke. It is dissociated from the
reference checkout and has no Git object alternates.

Pinned inputs:

- M: `DigitalLearningGmbH/MATH-lighteval` at
  `0530c78699ea5e8eb5530600900e1f328b48acad`;
- O: `open-r1/OpenR1-Math-220k` at
  `e4e141ec9dea9f8326f4d347be56105859b2bd68`;
- teacher: `Qwen/Qwen3-8B` at
  `b968826d9c46dd6066d109eabc6255188de91218`; and
- student: `Qwen/Qwen3-1.7B` at
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

The train environment resolves Torch 2.11.0, Transformers 4.57.6, TRL 1.8.0,
Datasets 4.8.5, PEFT 0.19.1, Accelerate 1.14.0, and math-verify 0.9.0. The
serve environment resolves Torch 2.11.0, Transformers 5.12.1, and vLLM
0.24.0.

## EIT execution ledger

| Job | Status | Purpose and evidentiary boundary |
|---:|---|---|
| `106656` | completed, `0:0`, 3s | Exact model-cache check. Cache visibility only. |
| `106657` | completed, `0:0`, 2m08s | Initial train preflight; infrastructure evidence only and superseded by `106667`. |
| `106658` | cancelled intentionally, batch `0:15`, 10m23s | Initial serve preflight repeated the expensive vLLM import through a second CLI process. It never printed `PASS`; no training ran. This led to commit `35b6c23`. |
| `106667` | completed, `0:0`, 12s | Final train preflight; A100, CUDA, bf16, packages, and exact cached snapshots passed. |
| `106668` | completed, `0:0`, 21s | Final serve preflight; vLLM 0.24.0 imported in-process and passed. |
| `106669` | completed, `0:0`, 44s | Partial data audit. Its manifest explicitly sets `scientific_use_allowed=false`. |
| `106670` | completed, `0:0`, 1m15s | One-step M teacher GRPO smoke. Adapter creation only; no informative reward. |
| `106671` | completed, `0:0`, 1m05s | One-step O teacher GRPO smoke. Adapter creation only; no informative reward. |
| `106672` | completed, `0:0`, 7m25s | M->M student smoke. Exact-token scoring, finite gradient/update, stable adapter; no task signal. Its completion custody flag is superseded by the correction. |
| `106676` | completed, `0:0`, 4m15s | O->O student smoke. Same checks on the other physical student-source role file; no task signal. Its completion custody flag is superseded by the correction. |
| `106687` | completed, `0:0`, 4m15s | M->M regression on commit `80836d4`; real update preserved and skipped process custody reported honestly. |

There were no OPD-math jobs left queued or running at handoff.

## Data-audit result

The audit intentionally read at most 64 rows from each source split. Its only
scientific blocker is `partial source scan`; `complete_collision_scan=false`
and `scientific_use_allowed=false`. The token-5-shingle semantic scan itself
completed, required no human review, and skipped no oversized buckets.

| Source | Teacher train | Student OPD | Teacher gap | Source holdout |
|---|---:|---:|---:|---:|
| M | 38 | 19 | 3 | 4 |
| O | 40 | 18 | 2 | 4 |

The primary matched budgets recorded by the manifest are 38 teacher examples,
18 student prompts, two gap records, and four holdout records. Every emitted
gold in the 192-row bounded surface parsed under math-verify 0.9.0. These facts
make the audit useful for integration testing, not for inference.

## What the smokes actually showed

### Teacher smokes

- M: four of four sampled completions were correct, so reward standard
  deviation, loss, and gradient were all zero. Peak CUDA allocation was about
  17.32 GB.
- O: zero of four sampled completions were correct and all four reached the
  256-token cap, so reward standard deviation, loss, and gradient were again
  zero. Peak CUDA allocation was about 17.49 GB.

Both jobs completed the GRPO/LoRA save path, but neither provided an
informative update. The contrast is precisely why a one-group stochastic
smoke must not gate teacher teachability.

### Student smokes at `35b6c23`

| Quantity | M->M (`106672`) | O->O (`106676`) |
|---|---:|---:|
| Prompts / samples / scored tokens | 1 / 2 / 128 | 1 / 2 / 128 |
| Total loss | 0.0004157 | 0.0002744 |
| Task loss | 0 | 0 |
| Score-function surrogate | 0.04157 | 0.02744 |
| Sampled K1 value | 0.14866 | 0.22179 |
| Mean positive-gap gate | 0.43264 | 0.43951 |
| Positive-gap token fraction | 0.21094 | 0.25000 |
| Pre-clip gradient norm | 0.01628 | 0.00487 |
| Mean task reward / informative groups | 0 / 0 | 0 / 0 |
| Parameter update observed | yes | yes |

All four sampled completions reached the 64-token smoke cap without EOS and
were recorded as `prediction_parse_failed`.

Both runs used one task row rather than the matched budget of 18. Their
teacher-gap, teacher-provenance, student-support, and scientific-environment
gates were null by design. Both served raw `Qwen/Qwen3-8B`, not either
one-step teacher-smoke adapter. Their M->M/O->O names identify the selected
source role file; they are not completed source-transfer matrix arms.

The promoted adapter-tree hashes are
`24378189899556375a2d2d145596bf97714bc89030ef712fade46ebff750065f`
for M->M and
`da3231e11f147a3dc28d8f161579b498c21587f1b12787fd2a42df51f72ee0bb`
for O->O.

These observations establish that the auxiliary can move the student when
task reward is flat. They do **not** establish that the movement helps math
accuracy, that OPD beats task RL, or that same-source teaching is preferable.

### Custody-label regression at `80836d4`

Job `106687` repeated the M->M smoke from the clean correction commit. It
again used one prompt, two samples, and 128 scored tokens. Total loss was
0.00041582, task loss was zero, the score-function surrogate was 0.041582,
the sampled K1 value was 0.142623, and the pre-clip gradient norm was
0.016373. A parameter update and stable final artifact hash were observed;
the promoted adapter-tree hash is
`8c7eea1dccf9bd3c32ae658429c4f85d0f1052d34181f4f4c0d445c50bcc4311`.
Both completions again reached the 64-token cap without EOS and failed answer
parsing, so this regression adds no task-quality evidence.

Most importantly, all three custody surfaces now agree:

- scoring probe: `local_process_binding_validated=false`, binding `null`;
- run binding: `live_local_server_process_binding_validated=false`; and
- completion: `local_server_process_binding_required=false`,
  `live_local_server_process_binding_validated=false`, end binding `null`.

The completion claim boundary now says, “No local server process custody is
claimed for this run.” Scientific `task_rl_k1_gap` runs still require and
revalidate the binding; the corrected smoke does not pretend that a waived
requirement passed.

## Canonical-preparation continuation - prelaunch checkpoint

Job `106884` completed the first full source audit at
`/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1_semantic_audit_ab9b289_b256`.
Its manifest SHA-256 is
`2fdedc71e9426c66ec971d146c325bff41f248220441432b91a8fe44ff8b6698`.
The semantic scan was complete and skipped no bucket, but it surfaced 666
unresolved review pairs. The review packet contains 666 rows with SHA-256
`b44faf43cbf062397101b1185e8aacb06413500b2704e5bd0d9d3301c40842b4`.
Under the conservative leakage policy, all 666 were finalized as `duplicate`;
the decision file has SHA-256
`e135fd2994f5d9ff243ccd392116661ff2a7a69e35ce9544ad3736445122ee4c`.

This audit cannot be promoted. In addition to unresolved decisions at creation
time, the old partition path emitted only 4,995 of the 5,000 frozen MATH-test
questions: five legitimate related subquestions shared a stem or diagram and
were incorrectly removed. The repaired contract retains every frozen question
while quarantining any touching training records, and fails closed unless the
partitioned test count equals the parsed input count.

At this checkpoint, the finalized decisions had not yet been persisted to the
canonical EIT review path and the reviewed canonical preparation had not run.
The hardened branch also adds exact teacher/student trace geometry, independent
TRL-reward recomputation, a committed 100-step student plan, full held-out
student custody, and the deterministic six-arm matrix readout. All 143 local
tests passed before this prelaunch documentation pass. No scientific teacher or
student performance result was created.

## Remaining gates before a scientific run

1. Prepare the full M/O source surface in a fresh directory. Complete the
   collision scan, review every semantic candidate that requires review, and
   create a second fresh canonical directory from the resolved review file.
2. Freeze environments under the exact clean commit used for that canonical
   preparation.
3. Evaluate the raw Qwen3-1.7B student with repeated groups on both student
   sources. Require nonzero pass@k and the preregistered mixed-reward fraction;
   otherwise stop rather than interpreting a KL-only update as learning.
4. Run the committed 100-step, matched-budget teacher recipe separately on M
   and O. Informative reward is required for the scientific teacher run even
   though it was correctly optional for a one-step plumbing smoke. Require an
   informative trainer signal, an independently verified mixed-reward trace
   group, and the exact expected without-replacement record diversity.
5. Evaluate each trained teacher against its own raw checkpoint on the complete
   registered teacher-gap role with repeated samples. Require the paired
   teacher-quality gate; ties fail.
6. Merge and serve only a teacher whose gap gate passes. Preserve the merge
   provenance, model tree hash, tokenizer contract, process binding, and both
   environment freezes.
7. Run one `task_rl` baseline per student source, then the four M/M, M/O, O/M,
   and O/O `task_rl_k1_gap` arms.
8. Evaluate all six adapters on the exact matched 370-row prefix of each
   registered `source_holdout` file, create six held-out custody gates, and
   build the exact six-arm readout with 10,000
   paired record-bootstrap draws. Compare reward, learning curves, token/latency
   cost, mixed-group frequency, teacher NLL, and gap strata.

The stop rule remains simple: without a reproducible teacher skill gap and
raw-student task support, do not launch the scientific OPD main arm.

## Canonical reviewed data and raw-support continuation

The conservative 666-row review file is now durably stored under
`/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/reviews/v1`, together
with the initial decisions, 59 reconciliation overrides, duplicate spot audit,
and verified checksums. EIT job `106951` completed the repaired reviewed
canonical preparation at
`/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1_canonical_reviewed_19b24c2`.
Its manifest SHA-256 is
`dc4cf7dc36ae5b5178b782bb9c9841e096fbf42dabcbebaa74fb0ed6afcdf430`;
`scientific_use_allowed=true` and the blocker list is empty. An independent
streaming audit rehashed all 13 registered files and row counts. The frozen
MATH test contains all 5,000 questions. Matched role budgets are 4,322 teacher
training records, 2,161 student records, 353 teacher-gap records, and 370
source-holdout records per source. That 353 gap value is the primary matched
budget, not the scientific gate size: the full registered gap files contain 353
M rows and 4,585 O rows, and the predeclared scientific gate rejects a prefix.

Timing-only raw-student jobs `106955` (M) and `106956` (O) were launched on
64-record prefixes from clean commit `19b24c2`. Job `106955` completed 64 x 4
samples in 710 wall-clock seconds, including 601.904 generation seconds. Its
55% sample accuracy is a timing/support diagnostic only; the favorable prefix
cannot feed a scientific gate. Job `106956` completed in 876 wall-clock
seconds, including 770.424 generation seconds; its sample accuracy was 9.8%.
The O prefix likewise cannot feed a gate. Together they project roughly 12.9
GPU-hours of generation for the two complete 2,161-record support surfaces,
before model-load and filesystem overhead.

The evaluator now supports immutable contiguous Slurm arrays.
Every record receives a task-hash/index/ID-derived seed; GPU tasks write fresh
transactional schema-v2 shards; a CPU merger proves complete disjoint coverage,
recomputes rewards, and emits one canonical artifact; the quality gate
independently reconstructs it from the bound shards. The planned full support
run uses 34 balanced shards per source with four-way concurrency. This
continuation still creates no teacher, OPD, or held-out performance claim.

## Known nonblocking hardening work

- Scientific same-host process custody validates the live PID, start identity,
  command line, checkpoint, alias, port, and context length, but does not yet
  prove through `/proc/net` that the PID owns the listening socket.
- The candidate-to-final rehash/rejection path is statically covered and was
  exercised by successful promotions, but its rejection branch still lacks a
  direct end-to-end integration test.

## Operational cleanup performed

The first ordinary EIT clone and one scratch retry stalled. Only those two
incomplete checkouts created during this setup (approximately 37 KiB and
107 MiB) were removed. The canonical EIT clone was rebuilt with a local
reference, dissociated, connectivity-checked, and verified to have no object
alternates. No dataset, model, environment, experiment artifact, archive, or
literature-vault material was deleted.

An 82 MB local temporary dependency target used to supply the pinned math
verifier for the initial handoff suite was also removed after all 121 tests
then present passed.

## Links

[[opd-math-source-transfer]] · [[opd-distillation]] ·
[[self-distillation-cluster-update-2026-07-17]] · [[ema-policy-gradient]] ·
[[verl-opd-trainer]]
