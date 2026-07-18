# OPD math source-transfer pipeline

This directory implements the first bounded experiment for the question:

> When does a teacher's training-data relationship make it a better
> on-policy distillation teacher for a smaller acting model?

The experiment is a 2 x 2 source matrix. `M` is pinned MATH-lighteval and `O`
is the recommended `default` config of OpenR1-Math-220k. A pair `(T, S)` means
the Qwen3 teacher is trained on source `T`, while the Qwen3-1.7B student
generates OPD rollouts on source `S`.

The current bounded EIT state, exact execution commits, artifact roots, job
ledger, custody correction, and remaining scientific gates are recorded in
[`opd-math-eit-handoff-2026-07-18.md`](../../wiki/snapshots/opd-math-eit-handoff-2026-07-18.md).
That snapshot validates plumbing only; it is not a task-performance result.

## Non-negotiable design choices

- Main same-source arms use disjoint problem clusters. Exact-row reuse is a
  separately named `same_items` memorization positive control.
- MATH test stays frozen external evaluation. All 5,000 benchmark questions are
  retained, including legitimate related subquestions that share a stem or
  diagram. Any training record in a cluster touching one of those questions is
  quarantined. Cross-M/O exact, formatting-only, or reviewed semantic clusters
  are quarantined from the primary matrix. Formatting normalization retains
  semantic whitespace, so `a b` is never silently equated with `ab`.
- Match teacher examples, OPD prompts, optimizer updates, and rollout counts;
  measure prompt and completion tokens rather than pretending source-dependent
  token counts are equal. OpenR1 is much larger, so unmatched full-corpus runs
  are secondary dose-response arms, not the primary comparison.
- The shared Qwen base may have unknown pretraining exposure to either source.
  Interpret the matrix as controlled **post-training-source** transfer, not as
  proof of a teacher's complete historical data exposure.
- `prepared_manifest.json` records the minimum role counts across M and O as
  the primary matched budgets. Role files are stable-rank sorted; use the
  recorded teacher limit and student `--task-limit` rather than silently giving
  OpenR1 more unique examples. Every pair also registers the exact row count and
  SHA-256 for each referenced role file.
- Role assignment is deterministic within each source x source-specific
  stratum. Collision edges, removed representatives, label conflicts, schema
  exclusions, and math-verifier exclusions remain in auditable JSONL ledgers.
- Every emitted role/evaluation gold parses with pinned `math-verify`. A
  semantic scan globally orders token-5 shingles and indexes at least
  `|S|-ceil(candidate_threshold*|S|)+1` shingles per record. This guarantees
  candidate recall for the declared Jaccard threshold unless an explicit
  bucket cap is hit. Skipped buckets or unresolved review-required candidates
  leave `scientific_use_allowed=false`. Review JSONL rows contain `pair_id` and
  `decision` (`duplicate` or `distinct`) and are supplied with
  `prepare_data.py --semantic-review-jsonl ...` on a fresh output directory.
  If a bounded block is skipped, increase the explicit
  `--semantic-max-bucket-size` on a sufficiently provisioned preparation job;
  the chosen bound and every skip remain recorded in the manifest.
- Teacher training uses verifiable task reward. The label is **GRPO with DAPO
  loss normalization**, not the complete DAPO algorithm.
- The student main arm is `task_rl_k1_gap`: grouped task reward plus a weighted,
  clipped, positive-gap-gated K1-value score-function reverse-KL surrogate.
  Only the ungated, unclipped, on-policy limit is K4/r-trick gradient-equivalent.
  `task_rl` is the primary baseline. `k1_bare` and `k1_gap_only`
  are diagnostics; their legacy names do not imply direct autodiff through K1.
- Teacher and student scoring uses exact generated token IDs. A pinned
  tokenizer fingerprint must pass; matching family names is not enough.
- Qwen3 thinking is disabled through the chat template. The teacher and
  student use the same rendered prompt contract.
- Finite loss, checkpoint creation, or a one-step smoke is plumbing evidence
  only. The main OPD arm also requires a positive held-out teacher gap and a
  student-support gate with nonzero/mixed-reward rollouts.
- The scientific exact-token probe binds the same-host Linux vLLM PID,
  `/proc` start time and command line to the merged checkpoint, its in-tree
  provenance manifest, served-model alias, port, and declared maximum context
  length. The student rechecks that live identity before and
  after training. This is local process custody, not cryptographic remote
  attestation. Student artifact eligibility also requires the same clean Git
  commit at training start and end.

## Pinned inputs

The machine-readable source of truth is
[`source_manifest.json`](../../configs/opd_math/source_manifest.json). One
important correction is recorded there: the user-provided
`One-Shot-RLVR-Qwen2.5-Math-1.5B-7.5k-MATH` URL is a model repository, not the
7,500-row MATH dataset. The empty third-party `Qwen3-8B-DAPO` repository is
also provenance-only. This lane trains its own teacher from pinned
`Qwen/Qwen3-8B`, with pinned Qwen3-4B only as a memory fallback.

The primary M/O teacher comparison is additionally bound to one committed
[`teacher_training_plan.json`](../../configs/opd_math/teacher_training_plan.json).
It fixes optimizer steps, generation/update batch geometry, prompt and
completion bounds, decoding, LoRA, and seed. A non-smoke run that differs in
any fixed field fails before loading a model; both source gates carry the same
plan and config hashes. The selected prompts are explicitly measured and any
rendered prompt over 1,536 tokens is rejected rather than silently truncated.

The two student baselines and four main arms are bound to the committed
[`student_training_plan.json`](../../configs/opd_math/student_training_plan.json):
100 optimizer steps, one prompt group per step, four rollouts per group, seed
zero, 1,536 prompt tokens, 512 completion tokens, and the exact task-reward and
K1-gap settings. This is a matched 100-prompt pilot, not full exposure to every
eligible role-file row or a training-seed study.

## Stage order

1. Create the isolated TRL environment with
   `scripts/hpc/setup_opd_math_env.sh`, reconstruct the vLLM environment with
   `setup_opd_math_serve_env.sh`, and populate the exact pinned model snapshots
   with `slurm_opd_math_cache_models.sh`. Validate both environments on a GPU
   with their preflight jobs. All serving launchers default to the reconstructed
   `/engrfs/project/jacobsn/hiqbal/envs/opd_math_serve` path.
2. Run `slurm_opd_math_prepare_data.sh` online. Start with a nonzero
   `OPD_MATH_AUDIT_LIMIT` in a disposable audit path; it is explicitly
   non-scientific. Then run the full corpus into a new semantic-audit path,
   review every `requires_review` row, and run a fresh canonical path with
   `OPD_MATH_SEMANTIC_REVIEW_JSONL` set. Do not reuse an output directory or
   treat the first full candidate surface as the canonical dataset.
3. Evaluate the raw student on the complete matched M and O `student_opd`
   surfaces and build both `student-support` manifests. If either source has
   nearly all-zero groups, stop. This repository does not implement an
   identity-bound warm-start path, and an ungated smoke is not a scientific
   substitute. Sampling must exactly match training (`temperature=1`,
   `top_p=1`, `top_k=0`, `max_new_tokens=512`, seed zero, group size four).
4. Run the one-step teacher smoke to validate the current callback and trace
   schema. Then train both M and O teachers with the matched 100-step recipe.
   Scientific eligibility requires an informative trainer-log signal, at least
   one mixed-reward trace group whose stored reward independently recomputes
   under TRL's exact accuracy contract, and exactly
   `min(matched_teacher_pool_rows, 100)` unique realized records.
5. Evaluate each base/trained teacher pair on identical frozen
   `teacher_skill_dev` records and separately report the `target_gap_dev`
   distribution. Use repeated non-thinking samples and build `teacher-gap`
   manifests with `slurm_opd_math_evaluate.sh` followed by
   `slurm_opd_math_quality_gate.sh`. A scientific teacher gate requires
   `OPD_MATH_EVAL_MAX_RECORDS=0`, meaning the complete registered
   `teacher_gap_dev` role file; a favorable prefix is not accepted. Before the
   full O evaluation, run a labeled timing-only prefix, choose enough immutable
   shards that every projected shard fits 24 hours with at least 25% headroom,
   and record the aggregate GPU-hour budget. Do not feed the prefix to a gate.
6. Merge only a teacher that passed its gate with
   `slurm_opd_math_merge_teacher.sh`, serve it with the separate vLLM
   environment, and run both the tokenizer and exact-token scoring probes.
7. Run one `task_rl` baseline per student source (`M`, `O`), then compare each
   to `task_rl_k1_gap` for M_M, M_O, O_M, and O_O. Do not manufacture four
   baseline labels when the teacher coordinate is unused.
8. Evaluate every promoted student adapter on the exact matched 370-row prefix
   of its registered `source_holdout` file with four samples and the exact
   student decoding contract. Build six independent held-out custody gates,
   then combine exactly
   `baseline_M`, `baseline_O`, `M_M`, `M_O`, `O_M`, and `O_O` into the matrix
   readout. Authorization is independent of whether an effect helps, harms, or
   is inconclusive.
9. Mine the over-collected traces for accuracy, mixed-group frequency,
   completion length, student NLL, and **teacher NLL on the student
   trajectory**. Low teacher NLL is not assumed to imply correctness.

Only after this `k=0` path and the task-reward baseline work should estimator
variance be added as a factor: compare `k=0` with Top-k reverse KL at `k=16`
and `k=32`, plus an explicitly biased truncated-head condition if resources
permit. EMA replacement is not folded into this matrix because it changes the
fixed trained teacher into a lagged student and answers a different question.

## Artifact layout on EIT

```text
/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1_canonical_reviewed_19b24c2/
  prepared_manifest.json
  roles/{M,O}/{teacher_train,student_opd,teacher_gap_dev,source_holdout}.jsonl
  eval/M_test.jsonl
  audit/quarantine.jsonl
  audit/ingestion_exclusions.jsonl
  audit/collision_edges.jsonl
  audit/semantic_candidates.jsonl

/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/
  smoke/
  teachers/
  evaluations/
  students/
```

Large data, checkpoints, and row traces stay on EIT. Only code, configs,
compact manifests, and genuinely signed-off evidence belong in Git.

## Executable EIT launch matrix

Bootstrap the fresh checkout from the pushed experiment branch, create the
Slurm log directory before the first `sbatch` (Slurm opens stdout before job
code can create it), and use one consistent set of persistent paths:

```bash
export OPD_MATH_REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math
export OPD_MATH_TRAIN_ENV=/engrfs/project/jacobsn/hiqbal/envs/opd_math_train
export OPD_MATH_SERVE_ENV=/engrfs/project/jacobsn/hiqbal/envs/opd_math_serve
export OPD_MATH_HF_HOME=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
export OPD_MATH_RUN_ROOT=/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math
export OPD_MATH_DATA_ROOT=/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1_canonical_reviewed_19b24c2
mkdir -p /engrfs/tmp/jacobsn/hiqbal_legalrag
test ! -e "$OPD_MATH_REPO"
git clone --branch codex/opd_math_pipeline --single-branch \
  https://github.com/1hamzaiqbal/LegalRagAgent.git "$OPD_MATH_REPO"
cd "$OPD_MATH_REPO"
test -z "$(git status --porcelain=v1)"
: "${OPD_MATH_EXPECTED_COMMIT:?Set this to the locally validated pushed commit}"
test "$(git rev-parse HEAD)" = "$OPD_MATH_EXPECTED_COMMIT"
```

The stages, launch surfaces, and required per-job settings are:

| Stage | Launcher | Required settings beyond the common paths | Promotion condition |
|---|---|---|---|
| Train environment | `bash scripts/hpc/setup_opd_math_env.sh` | none for a new path | `requirements.freeze.txt` exists |
| Serve environment | `bash scripts/hpc/setup_opd_math_serve_env.sh` | same `OPD_MATH_SERVE_ENV` used below | `requirements.freeze.txt` exists |
| Exact model cache | `sbatch scripts/hpc/slurm_opd_math_cache_models.sh` | network available; exact revisions come from `source_manifest.json` | log ends in `PASS` |
| GPU preflights | `slurm_opd_math_env_preflight.sh`, `slurm_opd_math_serve_preflight.sh` | cached teacher/student; reconstructed serve path | both logs end in `PASS` |
| Data preparation | `slurm_opd_math_prepare_data.sh` | unique `OPD_MATH_DATA_ROOT`; audit/review settings below | canonical manifest alone may be scientific |
| Teacher signal smoke | `slurm_opd_math_teacher_smoke.sh` | `OPD_MATH_TEACHER_SOURCE=M` or `O`; audit or canonical data root | plumbing only |
| Teacher training | `slurm_opd_math_teacher_train.sh` | source, exact teacher limit, explicit steps, `primary_matched` | eligible run manifest; quality still unknown |
| Repeated evaluation | `slurm_opd_math_evaluate.sh` | source, role, pinned model/revision, explicit record count and label; adapter only for trained model | artifact only |
| Evaluation merge | `slurm_opd_math_merge_evaluation.sh` | same source/role/label/run ID, exact shard count and task root; CPU dependency after every shard succeeds | one reconstruction-checked evaluation artifact; still no task claim |
| Teacher gate | `slurm_opd_math_quality_gate.sh` | base/trained summaries and samples, adapter, teacher run manifest, source | scientific gate passes |
| Student-support gate | same quality-gate launcher | raw-student summary/samples, pinned identity, source | scientific gate passes; otherwise stop |
| Teacher merge | `slurm_opd_math_merge_teacher.sh` | passing gate, exact adapter/base identity, fresh output | provenance and checkpoint hash exist |
| Student baseline/main | `slurm_opd_math_student_train.sh` | explicit mode, steps, task limit, budget and support gate; commit-specific train freeze; main also needs pair, teacher gate/checkpoint/provenance, and serve freeze | training artifact only; held-out evaluation remains required |
| Held-out student result / matrix | `slurm_opd_math_student_results.sh` | `heldout`: one eligible run, completion, exact adapter, and exact matched 370-row `source_holdout` prefix; `matrix`: exactly six passing held-out gates | deterministic custody readout; effect sign does not determine authorization |

Create the environments on a networked login node, fill the shared model cache
online, and only then run the offline GPU preflights:

```bash
bash scripts/hpc/setup_opd_math_env.sh
bash scripts/hpc/setup_opd_math_serve_env.sh
CACHE_JOB=$(sbatch --parsable scripts/hpc/slurm_opd_math_cache_models.sh)
echo "wait for model-cache job $CACHE_JOB before preflight"

# Run these only after the cache job reports PASS.
sbatch scripts/hpc/slurm_opd_math_env_preflight.sh
sbatch scripts/hpc/slurm_opd_math_serve_preflight.sh
```

Preserve the resolved dependency closures once, under a commit-specific fresh
directory. These freezes are campaign provenance; do not overwrite them:

```bash
COMMIT=$(git rev-parse HEAD)
FREEZE_ROOT="$OPD_MATH_RUN_ROOT/environment_freezes/$COMMIT"
test ! -e "$FREEZE_ROOT"
mkdir -p "$FREEZE_ROOT"
cp "$OPD_MATH_TRAIN_ENV/requirements.freeze.txt" "$FREEZE_ROOT/train.freeze.txt"
cp "$OPD_MATH_SERVE_ENV/requirements.freeze.txt" "$FREEZE_ROOT/serve.freeze.txt"
sha256sum "$FREEZE_ROOT/train.freeze.txt" "$FREEZE_ROOT/serve.freeze.txt" \
  > "$FREEZE_ROOT/SHA256SUMS"
```

Scientific student launchers pass these exact commit-specific copies into the
trainer. The trainer validates the pinned live train packages, binds both file
hashes for the main arm, and rehashes them before adapter promotion.

### Data audit and canonical preparation

Every target must be new. The full semantic audit and reviewed canonical run
must use the same fingerprint and bucket settings. If a skipped bucket forces a
larger value, update the exported value and create a new audit directory before
review; carry that same value into the canonical run.

Suggested preparation sequence (every target is a new directory):

```bash
export OPD_MATH_SEMANTIC_FINGERPRINT_SIZE=8
export OPD_MATH_SEMANTIC_MAX_BUCKET_SIZE=256

OPD_MATH_DATA_ROOT=/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/audit64 \
OPD_MATH_AUDIT_LIMIT=64 sbatch scripts/hpc/slurm_opd_math_prepare_data.sh

OPD_MATH_DATA_ROOT=/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1_semantic_audit \
sbatch scripts/hpc/slurm_opd_math_prepare_data.sh

OPD_MATH_DATA_ROOT=/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1_canonical_reviewed_19b24c2 \
OPD_MATH_SEMANTIC_REVIEW_JSONL=/absolute/path/to/reviewed-decisions.jsonl \
sbatch scripts/hpc/slurm_opd_math_prepare_data.sh
```

Freeze review decisions with the tracked finalizer before the canonical run.
It requires exactly one decision for every packet pair, rejects unknown or
duplicate pair IDs, applies only explicit override files, preserves packet
order, and refuses to overwrite an existing output:

```bash
REVIEW_ROOT=/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/reviews/v1
mkdir -p "$REVIEW_ROOT"
test ! -e "$REVIEW_ROOT/reviewed-decisions.jsonl"
"$OPD_MATH_TRAIN_ENV/bin/python" scripts/opd_math/finalize_semantic_reviews.py \
  --packet /absolute/path/to/full-audit-review-packet.jsonl \
  --review /absolute/path/to/complete-initial-review-decisions.jsonl \
  --override /absolute/path/to/explicit-reconciliation-overrides.jsonl \
  --output "$REVIEW_ROOT/reviewed-decisions.jsonl"
sha256sum "$REVIEW_ROOT/reviewed-decisions.jsonl"
```

If the full audit records skipped buckets, rerun another new audit path with a
larger positive `OPD_MATH_SEMANTIC_MAX_BUCKET_SIZE`, regenerate decisions from
that surface, and reuse the same setting for canonical preparation. Never
hand-edit the prepared manifest. Before any scientific job, require:

```bash
"$OPD_MATH_TRAIN_ENV/bin/python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["scientific_use_allowed"] is True, d["scientific_blockers"]; print(d["primary_matched_budgets"])' \
  "$OPD_MATH_DATA_ROOT/prepared_manifest.json"
```

### Teacher, evaluation, and gate settings

Read the numeric matched limits from the canonical manifest rather than typing
them from notes:

```bash
TEACHER_LIMIT=$("$OPD_MATH_TRAIN_ENV/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["primary_matched_budgets"]["teacher_train"])' \
  "$OPD_MATH_DATA_ROOT/prepared_manifest.json")
STUDENT_LIMIT=$("$OPD_MATH_TRAIN_ENV/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["primary_matched_budgets"]["student_opd"])' \
  "$OPD_MATH_DATA_ROOT/prepared_manifest.json")
export TEACHER_LIMIT STUDENT_LIMIT
```

For each teacher source, use the committed recipe. The tracked launcher defaults
to its 100-step primary recipe; environment overrides are accepted only when
they remain exactly plan-compliant. A completed teacher run is not a passed
teacher:

```bash
export OPD_MATH_TEACHER_LIMIT="$TEACHER_LIMIT"
export OPD_MATH_BUDGET_MODE=primary_matched
OPD_MATH_TEACHER_SOURCE=M sbatch scripts/hpc/slurm_opd_math_teacher_train.sh
OPD_MATH_TEACHER_SOURCE=O sbatch scripts/hpc/slurm_opd_math_teacher_train.sh
```

The evaluation launcher always needs these explicit variables:

```text
OPD_MATH_EVAL_SOURCE=M|O
OPD_MATH_EVAL_ROLE=teacher_skill_dev|target_gap_dev|student_support|source_holdout|external_M_test
OPD_MATH_EVAL_MODEL=<pinned base model>
OPD_MATH_EVAL_MODEL_REVISION=<40-hex revision>
OPD_MATH_EVAL_MAX_RECORDS=<explicit integer; 0 means the complete role>
OPD_MATH_EVAL_LABEL=<new filesystem-safe label>
OPD_MATH_EVAL_ADAPTER=<adapter directory; omit for a base/raw-model evaluation>
OPD_MATH_EVAL_SHARDS=<positive contiguous shard count; default 1>
OPD_MATH_EVAL_RUN_ID=<stable filesystem-safe ID; required for every evaluation>
```

Every evaluation writes an immutable schema-v2 shard, including a one-shard
run; anything intended for a gate must pass through the CPU merger. `MAX_RECORDS` is the
global selected prefix, not a per-shard budget. Each record's RNG seed binds
the base seed, complete task-file hash, global row index, and record ID, so a
retry or a different shard count cannot change that record's random stream.
Each array task writes transactionally to
`.../$RUN_ID/shards/shard_NNNNN`; a failed task leaves only a separately named
partial directory. Resume by submitting only the missing shard indices. Never
append to or replace a completed shard. If the original array failed, its
`afterok` merge dependency will not run; after the missing-index retry succeeds,
submit a fresh CPU merge job against the same stable run ID.

After every shard succeeds, merge on CPU with the matching source, role,
label, run ID, shard count, data root, and run root:

```bash
ARRAY_RAW=$(sbatch --parsable --array=0-33%4 \
  scripts/hpc/slurm_opd_math_evaluate.sh)
ARRAY_JOB=${ARRAY_RAW%%;*}
MERGE_JOB=$(sbatch --parsable --dependency="afterok:$ARRAY_JOB" \
  scripts/hpc/slurm_opd_math_merge_evaluation.sh)
printf 'array=%s merge=%s\n' "$ARRAY_JOB" "$MERGE_JOB"
```

The merger requires the exact numbered shard set, reopens the task and any
adapter, verifies clean start/end code custody, recomputes record seeds and
math rewards, proves gap-free coverage and canonical sample order, and emits a
fresh merged summary plus samples. A raw shard is never accepted by a quality
gate. The gate independently reopens every bound shard and reconstructs the
merged byte stream, so deleting or mutating a shard invalidates the result.

For the O teacher, first run a timing-only base-model prefix, for example 32
records. Use its `total_generation_latency_seconds` to choose a shard count
whose projected **per-shard** runtime is at most 18 hours, and separately record
the complete campaign's projected GPU-hours. The prefix is not gate evidence.
The full base and trained contracts still use `MAX_RECORDS=0`; resumability is
provided by immutable shards, not by truncating the registered O role.

A teacher gate additionally requires all of the following, with a fresh output:

```text
OPD_MATH_GATE_KIND=teacher_gap
OPD_MATH_GATE_SOURCE=M|O
OPD_MATH_GATE_BASE_{SUMMARY,SAMPLES}=<matching base artifacts>
OPD_MATH_GATE_TRAINED_{SUMMARY,SAMPLES}=<matching trained artifacts>
OPD_MATH_GATE_BASE_MODEL=<pinned teacher>
OPD_MATH_GATE_BASE_REVISION=<40-hex revision>
OPD_MATH_GATE_TRAINED_ADAPTER=<exact evaluated adapter>
OPD_MATH_GATE_TEACHER_RUN_MANIFEST=<that adapter's run_manifest.json>
OPD_MATH_GATE_OUTPUT=<new persistent JSON path>
```

`teacher_skill_dev` and `target_gap_dev` are evaluator aliases for the
source-specific physical `teacher_gap_dev` file. Only the teacher's own-source
`teacher_skill_dev` base/trained pair feeds its scientific teacher gate;
`target_gap_dev` is reported separately. Do not confuse the manifest's 353-row
primary matched budget with the scientific gate's complete-role requirement:
the registered M gap file has 353 rows and the registered O gap file has 4,585.
The scientific base and trained evaluations therefore use 353 M rows and 4,585
O rows respectively. Run a timing-only O prefix before choosing its shard count;
do not silently cap the O gate at 353.

### Student support, merge, and student runs

Evaluate the pinned raw student with role `student_support`, record count exactly
`$STUDENT_LIMIT`, and the future training contract: four samples, temperature
1, top-p 1, top-k 0, max-new-tokens 512, and the same seed. The reviewed
canonical budget is 2,161 records per source. The measured M timing prefix took
601.904 generation seconds for 64 records; the matching O prefix took 770.424
seconds. Use 34 balanced shards (63--64 records each) and `%4` concurrency for
both sources. Timing prefixes are not gate evidence. Feed only each CPU-merged
`summary.json` and `samples.jsonl` to the gate. Then set:

```text
OPD_MATH_GATE_KIND=student_support
OPD_MATH_GATE_SOURCE=M|O
OPD_MATH_GATE_STUDENT_{SUMMARY,SAMPLES}=<raw-student artifacts>
OPD_MATH_GATE_STUDENT_MODEL=Qwen/Qwen3-1.7B
OPD_MATH_GATE_STUDENT_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
OPD_MATH_GATE_OUTPUT=<new persistent JSON path>
```

If this gate fails, stop. There is currently no supported warm-start adapter
input and no scientific permission to substitute a smoke gate.

Merge a teacher only after its scientific gate passes, using a fresh output:

```text
OPD_MATH_MERGE_BASE_MODEL=Qwen/Qwen3-8B
OPD_MATH_MERGE_BASE_REVISION=b968826d9c46dd6066d109eabc6255188de91218
OPD_MATH_MERGE_ADAPTER=<exact evaluated adapter>
OPD_MATH_MERGE_GATE=<passing teacher-gap JSON>
OPD_MATH_MERGE_OUTPUT=<new merged-checkpoint directory>
```

Run one `task_rl` baseline for each student source before the four main pairs.
Every baseline/main comparison must reuse the same explicit steps, task limit,
group size, seed, rollout length, and enabled gradient checkpointing. The
checkpointing choice is registered in `student_training_plan.json`, passed by
the full launcher, and recorded in each run manifest; no ambient environment
toggle participates in the scientific contract. Baselines require
`OPD_MATH_STUDENT_SOURCE=M|O`; main runs require `OPD_MATH_PAIR=M_M|M_O|O_M|O_O`
plus the merged checkpoint, teacher gate, merge provenance, and teacher base
identity required by `slurm_opd_math_student_train.sh`. Its internal tokenizer
and exact-token probes must pass. Final adapters still require repeated held-out
evaluation; the training completion manifest is not a task-performance result.

For each promoted adapter, run the evaluation launcher with role
`source_holdout`, `OPD_MATH_EVAL_MAX_RECORDS=370`, four samples, temperature 1,
top-p 1, top-k 0, max-new-tokens 512, and seed zero. The physical O file is
larger; `MAX_RECORDS=0` would violate the matched held-out contract and fail the
result gate. Then create a fresh held-out gate with:

```text
OPD_MATH_RESULT_KIND=heldout
OPD_MATH_MATRIX_KEY=baseline_M|baseline_O|M_M|M_O|O_M|O_O
OPD_MATH_RESULT_SOURCE=M|O
OPD_MATH_STUDENT_RUN_MANIFEST=<eligible run_manifest.json>
OPD_MATH_STUDENT_COMPLETION_MANIFEST=<sibling completion_manifest.json>
OPD_MATH_STUDENT_EVAL_{SUMMARY,SAMPLES}=<exact merged 370-row holdout artifacts>
OPD_MATH_STUDENT_ADAPTER=<exact evaluated final adapter>
OPD_MATH_RESULT_OUTPUT=<new held-out gate JSON>
```

The gate recomputes held-out rewards and exact 100-step/400-sample training
geometry. The matrix requires the same realized record/prompt sequence within
each M or O student-source stratum. Once all six held-out gates pass, run
`OPD_MATH_RESULT_KIND=matrix` with
`OPD_MATH_RESULT_{BASELINE_M,BASELINE_O,M_M,M_O,O_M,O_O}` and fresh JSON and
Markdown outputs. The readout uses 10,000 paired record-bootstrap draws for the
four baseline deltas, same-versus-cross contrasts, and stratified interaction;
`helps`, `harms`, and `inconclusive` are all valid signed results.

## Current boundary

EIT job `106884` produced a complete semantic candidate surface, but its old
partition code retained only 4,995 of 5,000 MATH test questions and its 666
review decisions were not yet applied to a fresh canonical directory. It is an
audit artifact, not canonical data. The repaired code and CPU tests establish
the intended data, reward, trace, held-out, and matrix contracts. No scientific
teacher or student training result exists until the reviewed canonical rerun
and the ordered gates above complete.
