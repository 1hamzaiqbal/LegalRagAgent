---
title: OPD Math EIT Plumbing Handoff
type: snapshot
tags: [opd, math, eit, source-transfer, plumbing, custody]
created: 2026-07-18
updated: 2026-07-18
status: canonical data and raw-student support passed; teacher science pending
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
evaluation was run. Later continuation below records full-run launch attempts;
neither initial teacher attempt entered training or created an artifact.

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
independently reconstructs it from the bound shards. At that checkpoint, the
full support run was predeclared as 34 balanced shards per source with four-way
concurrency. No teacher, OPD, or held-out performance claim had been created.

## Complete raw-student support gate

The planned support campaign subsequently completed from clean execution commit
`f283c9c22251bb5dc9693754dfd8282afb07b61e`. For each source, 34 balanced
GPU shards covered the exact registered 2,161-row `student_opd` surface once,
with four non-thinking samples per problem under temperature 1, top-p 1,
top-k 0, a 512-token completion cap, and seed 0. The array roots were jobs
`106974` (M) and `106977` (O); CPU merges were `106978` and `106979`; and
scientific gates were `106983` and `106984`. Every task, merge, and gate
completed with exit code `0:0`.

| Source | Records | Samples | Sample accuracy | pass@4 | Mixed-reward groups | Gate SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| M | 2,161 | 8,644 | 0.5243 | 0.6201 | 0.1981 | `352c7763545376d883d2c1758126977b4045cf2ce373446b72b7520e14342c44` |
| O | 2,161 | 8,644 | 0.1041 | 0.1772 | 0.1273 | `1b7628759f8d9e2e45fa595870455e9268f312d0fb772f4964a2efca26997eee` |

Both schema-v3 gates set `passed=true` and
`authorizes_scientific_training=true`. This establishes the preregistered
raw-student support needed to attempt task-reward training on both sources. It
does not show that training improves the student, that either teacher is
useful, or that OPD beats the matched task-RL baseline.

The M gate was independently reproduced byte for byte on the login lane and is
retained as diagnostic evidence. An initial O
recomputation on the login lane hit a transient `gold_parse_failed` on the
valid registered gold `\\boxed{x^2=80}`; that failed diagnostic is preserved
rather than erased. Two independent allocated-CPU jobs, `107137` and `107138`,
then reproduced the canonical O gate byte for byte. The immutable audit lives
at
`artifacts/legalrag/opd_math/audits/student_support_recompute_f283c9c_verified_v2/`;
its manifest SHA-256 is
`4c2488587bdfa0866fc181c7e992dc6d1ef037d21ff6778cd13b73ddbf54ea04`.
Its operational conclusion is that future scientific gate recomputations must
run on allocated Slurm CPU nodes rather than the login node.

The predecessor bundle ending in `_verified/` is also preserved. Its data
files and hashes are correct, but three manifest artifact fields retained the
transactional `.partial.1625416` path after promotion. The v2 manifest names
that predecessor and corrects only those paths plus the login-versus-allocated
lane wording; no gate file or numeric result changed.

This closes only the data and raw-student-support prerequisites. The next
scientific work remains the exact 100-step M and O teacher recipes, each
teacher's own-source held-out gap gate, and then the matched two-baseline plus
four-OPD student matrix.

## First frozen-commit preflight and recovered code defect

Commit `248f9e75893abcc8742099b0d0a9b85e08e6e5d0` was pushed, cleanly
fast-forwarded to EIT, and bound to fresh read-only train/serve freezes. Their
SHA-256 values remained
`69edfc8f75927379124c88391578097fed7fa66d6e779df593b297cbbf2ada06`
and
`f36743fed4fe4c1b8a7fefaac39d9ce8a6f441897920c75ca11de8c487bf443d`;
the new exact live-environment verifier passed both full installed-distribution
maps and the serve executable/shebang. Allocated CPU jobs `107169` and
`107170` then regenerated the M and O support gates byte for byte under the
new code before any GPU smoke.

Teacher smoke `107171` completed `0:0`. Its single M prompt group produced
four incorrect 256-token-capped samples, zero reward variance, zero gradient,
and a final adapter. Its run manifest is explicitly non-scientific; the four-row
teacher trace SHA-256 is
`53b9b44f2f52354be55c6078dc51c78cf03b1c11e3a878aeb038cfa22a96685a`,
and the independently rehashed final-adapter tree is
`17b0c8ee9db3f28e4081c272e54af6b81fd5a489f87474390048d452113c74bf`.
This validates the current callback, trace, and save path only; it is not the
required informative 100-step teacher run.

The first full-wrapper M task-RL smoke, job `107172`, failed `1:0` after one
optimizer step because a later local integer named `sample_trace_rows`
shadowed the same-named trace-builder function. The preserved step row shows
that this was not the predeclared homogeneous-reward fallback case: reward mean
was 0.5, informative-group fraction 1.0, task loss 0.0408608, and pre-clip
gradient norm 0.783315. However, the crash occurred before sample-trace,
completion-manifest, parameter-signature, or final-adapter promotion, so none
of those missing checks may be inferred from the step row. The incomplete run
remains at `students/baseline_M/task_rl/run_107172/`, and its stdout SHA-256 is
`31088242cd4e08f1399ff1cf630aefdf1fb6bbb6fa863cf2a9f1ea4b7aab4f34`.

The branch fix renames the local count and adds a regression proving that
`run` cannot bind the trace-builder name. The prelaunch decision record is the
read-only EIT artifact
`campaigns/scientific_preflight_248f9e7/smoke_fallback_policy.json`, SHA-256
`d8d96d3c0b42c3540af4690ad891e0d22faadc17f2c4fdae994d74ea48e92eb1`.
Its predeclared 20-step stochastic fallback was not activated: this was a code
failure, and the observed group was already informative. A new frozen commit
must repeat the same one-step contract as a code-fix regression before full
training proceeds.

## Recovered smoke, first full launch, and teacher prompt-bound correction

Commit `6be96e65bfde2563d526205e4e8b870925b39b50` contains the local-name
fix. Allocated-CPU jobs `107176` and `107177` reproduced the canonical M and O
raw-student support gates byte for byte under that commit. Teacher smoke
`107178` then completed `0:0`; it again had flat reward, but completed the
callback/save contract and produced final-adapter tree SHA-256
`9fdc79383d0752cffefe0bc897621b23b49f3b68e3191688491515c667590aa0`.
Task-RL regression smoke `107179` completed `0:0` with rewards `[0,1,1,0]`,
informative-group fraction 1.0, task loss 0.0408608, pre-clip gradient norm
0.783315, a parameter update, and final-adapter tree SHA-256
`a462b8c874ec113ded98ad3355977bf5de48dd13af65bf4e1806c7b25da7581b`.
These are plumbing regressions, not task-performance results.

The first exact 100-step teacher jobs, M `107180` and O `107181`, failed `1:0`
before trainer construction or optimization. The fail-closed whole-pool prompt
check found M selected row 3,301 at 1,546 rendered tokens and O selected row
2,305 at 1,731 tokens, both above the predeclared 1,536-token teacher bound.
Each run directory contains zero files: there is no trace, checkpoint, adapter,
or scientific result. Their stdout SHA-256 values are respectively
`face236201c5a5c4953dc6a5e04808f520c6e43cea3bf66fb521904b5fc05a19`
and
`91e3dd4b32f9eeb7af428635dff242b026940a52c7f33c5f4905a0f0ac8c2f6c`.

Allocated-CPU diagnostic `107185` subsequently rendered every registered
teacher prompt under the pinned tokenizer. M had 4,322 rows, mean 109.48,
p99 457, maximum 1,546, and exactly one row above 1,536. O had 4,322 rows,
mean 133.68, p99 379, maximum 2,076, exactly two rows above 1,536, and one
above 2,048. Its stdout SHA-256 is
`067b20ff6e95732c1590e19e3015130441aa7aad2ebcc43b27b94f2ed158c4a8`;
the diagnostic-script SHA-256 is
`08c705520a1846f549905e188355db555b106f9450ff5d8d4d0929ce82f12de7`.
Setup attempt `107184` is preserved separately: it failed immediately because
its node could not see a login-node `/tmp` script and contributes no prompt
evidence.

The immutable decision record is
`campaigns/teacher_prompt_contract_6be96e6/decision.json`, SHA-256
`b7f0db058729821432602e5841372860d4657376dd3cab86f9dcbd3cd99362eb`.
It binds the source logs, full distributions, zero-file run directories, old
and new bounds, unchanged student contract, and non-authorizing claim boundary.

Successor teacher plan `opd_math_teacher_primary_v2` raises only the shared
teacher prompt bound from 1,536 to 2,304 tokens. Together with the unchanged
1,024-token completion cap, this preserves every matched teacher row without
implicit truncation. No source is
filtered or granted a different bound, and the student's separate 1,536-token
prompt contract is unchanged. Full task-RL baselines `107182` and `107183`
were launched from `6be96e6`; at that earlier checkpoint no terminal result
was asserted. Regardless of their subsequent eligibility, the current
six-arm readout requires every student-training and
held-out-evaluation artifact to share one exact Git commit. They therefore
remain predecessor-commit diagnostics and must be repeated on the successor
commit for the final matrix. The teachers must likewise be relaunched from
that clean successor commit and still pass the unchanged informative-reward
and own-source held-out skill-gap gates before any OPD arm.

## Predecessor task-RL diagnostics and successor custody gate

Both predecessor task-RL jobs subsequently completed cleanly from
`6be96e65bfde2563d526205e4e8b870925b39b50`. They establish that both
registered student sources can produce task-reward learning signal under the
100-step recipe. They are not members of the final six-arm matrix, have not
been evaluated on `source_holdout`, and both completion manifests retain
`scientific_use_allowed=false`.

| Source/job | Informative groups | Reward outcomes | Max pre-clip gradient | Final adapter tree SHA-256 |
|---|---:|---|---:|---|
| M `107182` | 22/100 | 218 correct, 41 incorrect, 141 parse failures | 0.871966 | `d4567b1497b415752037072b2cc737c199b15b3c542ebfc89558c0055ed93bc4` |
| O `107183` | 12/100 | 35 correct, 60 incorrect, 305 parse failures | 1.148583 | `5ce3f38fcf36d97b7396a70c40a427b11d74aa92d8b505df0ecfe72a6c4dc51f` |

Each run has exactly 100 optimizer steps, 400 traced samples, 100 prompt
groups, 100 unique records and prompt hashes, four samples per group, finite
numerics, a nonzero-gradient step for every informative group, a measured
parameter update, clean start/save/end Git custody, an unchanged exact train
environment, and a stable final artifact hash. The M run/completion/step/sample
manifest hashes are respectively
`1c83f8d7571a3b7e780df3ff2b975f6072da0423b3fe4bebfdf5ab2581f0a389`,
`dd90aa48448a0b8e26f67b6579b0cbb92b388b5b969a5fac473a981722e0ea3c`,
`82b48499e44300b3df40c583174597ceb1bbc92170a41e32165ca5e5840b62b2`,
and `b66eee5cf88801502301789093647664f05baccdc371fc46b7666931911c8b43`;
its stdout hash is
`78cc9570c221399799cf80fe843f2b0c67ad72f62af745a0889148ea4230ac0f`.
The corresponding O hashes are
`807f36c9340910ce501df876c8a8e3dea3d430cb8db5cc41f6e483bf1594d3f7`,
`40f13504fdf1879ea039f78083f7611d543f9db21dfcdc7abf2440618f1feb51`,
`78596f9ba85cea531008f68b1a162f85e7b7ad4eabd51efbb2b3b5dcfb542be1`,
and `b4f721804f8d3bb88327627c1b20371a84f0c850cb38bdd4e76357e3a963dde4`;
its stdout hash is
`ba7f90266c2d9a4e01b045b1bb847e8d27c01d88c18237513bc110daf2e3195d`.
Allocated-CPU read-only audit job `107198` then reran the full student-run
contract validator over both exact traces, manifests, and adapters and
completed `0:0`.
The immutable terminal record is
`campaigns/predecessor_task_rl_6be96e6_terminal/terminal_audit.json`, SHA-256
`1fbe3b648eca9b6f63d584488c079de14275de20c8198387d44748de83ecbf52`.
The 305/400 O parse failures are a substantive diagnostic to retain in
held-out interpretation rather than hide behind pass/fail training status.

Commit `cef34f6b42b6db87f3d3703497d7a75425dc71af` recorded the shared
2,304-token teacher prompt plan, but a prelaunch audit stopped it before EIT
execution: scientific teacher training did not yet bind and reverify the exact
commit-specific full train environment. The successor patch now requires that
environment at trainer and wrapper entry, rechecks it before candidate save,
after candidate save, after final promotion, and at the teacher quality gate,
and carries the normalized identity through teacher-gap output, merged-teacher
provenance, OPD training, and held-out result validation. The merged-teacher
provenance schema is consequently `opd_math_merged_teacher_v3`; no v2 consumer
remains.

The independent audit also caught and fixed one subtle no-promotion defect
before launch: a disappearing installed distribution raised
`PackageNotFoundError` after candidate promotion instead of being converted to
a failed custody check and quarantine. The corrected path fails closed, and
the precommit suite passes 223 tests plus shell syntax, workspace, compile, and
diff checks. No successor teacher or student job has been launched from this
patch. At this checkpoint there are no active OPD-math jobs, and the EIT clone
remains clean at `6be96e6`; it must be fast-forwarded only after the successor
commit and fresh immutable freezes exist.

## Known nonblocking hardening work

- Scientific same-host process custody validates the live PID, start identity,
  command line, checkpoint, alias, port, and context length, but does not yet
  prove through `/proc/net` that the PID owns the listening socket.
- The candidate-to-final rehash/rejection path is statically covered and was
  exercised by successful promotions, but its rejection branch still lacks a
  direct end-to-end integration test.
- The campaign deliberately uses one successor commit for teacher training,
  merge, student training, and final evaluation. Teacher commit identity is
  preserved through every artifact, but merge code does not independently
  reject a different later clean commit; the immutable launch ledger must keep
  this operational constraint explicit.

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
