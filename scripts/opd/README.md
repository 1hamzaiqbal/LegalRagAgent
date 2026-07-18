# OPD scaffold for agentic retrieval skills

This directory contains a minimal on-policy distillation (OPD) scaffold for
internalizing agentic retrieval behavior from a larger teacher into a smaller
student. It follows the design in `wiki/concepts/skill-distillation-bridge.md`:
student completions are sampled on-policy, the teacher scores those exact
completion token IDs, and the student is updated with a score-function
reverse-KL surrogate. The detached multiplier is the sampled K1 log-ratio; in
the ungated, unclipped on-policy case, its expected gradient is equivalent to
the K4/r-trick estimator in [[ema-policy-gradient]]. Direct autodiff through K1
would instead average to zero. The branch implements an SDAR-inspired
positive-gap gate plus grouped math task reward. The gate/clipping deliberately
bias the estimator, and the K1 value estimate is not a full-vocabulary KL.

## Current evidence boundary

- EIT job 93802 validated bare OPD plumbing: Qwen3-8B teacher,
  Qwen3-1.7B student, three finite steps, checkpoints written.
- That smoke is not evidence of task learning.
- SDAR reports that standalone/naively mixed on-policy self-distillation can
  collapse. Use `opd_gated` for the next plumbing test, and do not launch a
  scientific E3 until the teacher skill-gap A/B is positive.
- The math child lane now CPU-tests `task_rl` and `task_rl_k1_gap`. No EIT math
  task result exists until the new Slurm smokes and quality gates complete.

## Tokenizer constraint

Teacher and student MUST pass the exact tokenizer contract. OPD aligns teacher
and student token IDs, so matching family names alone is insufficient. The
contract fingerprints the vocabulary mapping, special IDs, chat template,
rendered probes, and live vLLM tokenization.

Use pairs such as:

- Qwen3 family teacher plus Qwen3 family student.
- Llama-3.x family teacher plus Llama-3.x family student.

Do not mix Qwen and Llama, or any pair where `tokenizer.encode(prompt)` and the
completion tokenization are not aligned across teacher and student.

## Three-process layout

Process 1: teacher server, OpenAI-compatible vLLM endpoint.

```bash
vllm serve /path/to/merged-teacher --served-model-name opd-math-teacher \
  --port 8000 --max-model-len 4096
```

For two-GPU teachers, add tensor parallelism:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000 \
  --tensor-parallel-size 2 --max-model-len 4096
```

Process 2: student trainer on its own GPU. vLLM is not imported by the trainer;
it is only contacted as an HTTP server.

```bash
python scripts/opd/opd_train.py \
  --mode task_rl_k1_gap \
  --pair-id M_M \
  --task-file "$DATA_ROOT/roles/M/student_opd.jsonl" \
  --task-limit "$MATCHED_STUDENT_LIMIT" \
  --budget-mode primary_matched \
  --prepared-manifest "$DATA_ROOT/prepared_manifest.json" \
  --student Qwen/Qwen3-1.7B \
  --student-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --teacher-url http://127.0.0.1:8000 \
  --teacher-model opd-math-teacher \
  --teacher-checkpoint /path/to/merged-teacher \
  --teacher-server-max-model-len 4096 \
  --teacher-base-model Qwen/Qwen3-8B \
  --teacher-base-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --teacher-gap-manifest teacher_gap.json \
  --teacher-provenance-manifest /path/to/merged-teacher/merge_provenance.json \
  --student-support-manifest student_support.json \
  --tokenizer-contract tokenizer_contract.json \
  --server-scoring-contract server_scoring_contract.json \
  --out-dir runs/opd-qwen17b \
  --steps 100 \
  --group-size 4 \
  --max-new-tokens 128
```

`task_rl_k1_gap` combines grouped verifiable task reward with a weighted,
clipped, positive-gap-gated K1-value score-function auxiliary. Its ungated,
unclipped, on-policy limit has the K4/r-trick-equivalent gradient; the executed
main objective does not. Metrics separately report the sampled K1 value and
the score-function surrogate. The auxiliary multiplies each token update by
`sigmoid(beta * (teacher_logp - student_logp))`; `--gap-gate-beta` defaults to
`5.0`. `task_rl` is the matched primary baseline. `k1_bare` and `k1_gap_only`
are diagnostic arms; historical aliases `opd` and `opd_gated` remain accepted.

Process 3: task data as JSONL. Math reward rows contain a conversational prompt
and parseable gold solution.

```jsonl
{"record_id":"M:...","source":"M","prompt":[{"role":"user","content":"Solve ..."}],"solution":"\\boxed{2}"}
```

Closed-teacher or SFT fallback mode samples teacher completions and trains with
plain supervised next-token loss:

```bash
python scripts/opd/opd_train.py \
  --mode kd \
  --task-file tasks.jsonl \
  --student Qwen/Qwen3-1.7B \
  --teacher-url http://127.0.0.1:8000 \
  --teacher-model Qwen/Qwen3-8B \
  --out-dir runs/kd-qwen17b \
  --steps 100
```

If a JSONL row already has `completion_text`, KD mode uses that text instead of
sampling it from the teacher.

The sampled reverse-KL modes require untruncated sampling (`temperature=1`,
`top_p=1`, `top_k=0`). Prompt and completion token IDs are preserved through
teacher scoring; decoded text is used only for task verification and hashes.

Top-k KL is a later estimator ablation, not part of this minimal server
protocol. It requires the student's top-k token IDs plus teacher scores on
those arbitrary IDs and a sampled-tail correction; one teacher log-probability
per generated token is insufficient. The pinned veRL implementation and
EMA-PG code are recorded in [[verl-opd-trainer]] and [[ema-policy-gradient]].

## Batching and checkpoints

- `--micro-prompts` explicitly controls prompts per optimizer step. It is
  recorded in the run manifest and must match across baseline/main arms.
- `--group-size` controls completions sampled per prompt. Default: `4`.
- `--save-every` writes `step_XXXXXX/` LoRA checkpoints.
- The final checkpoint is written to `final/`.
- The completion manifest records a SHA-256 tree identity for `final/` and can
  mark it eligible for held-out evaluation. It never labels a training artifact
  itself a scientific result.

## A100 smoke test

The historical smoke test requires the judge-lane venv plus `pip install vllm`. It starts a
local Qwen3-8B teacher, generates eight tiny prompts, runs three OPD steps with a
Qwen3-1.7B student, checks that the logged loss is finite, and confirms that a
checkpoint exists.

```bash
bash scripts/opd/smoke_test.sh
```

The default is now the gated plumbing mode. Set `OPD_SMOKE_MODE=opd` only to
reproduce the historical bare-OPD smoke.

The math pipeline has separate environment, data, teacher, and student Slurm
smokes under `scripts/hpc/slurm_opd_math_*.sh`; see
[`../opd_math/README.md`](../opd_math/README.md).

On EIT, submit the tracked clean-worktree launcher:

```bash
sbatch scripts/hpc/slurm_opd_gated_smoke.sh
```

It runs offline against the shared model cache and writes the exact Slurm log
to `/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_gated_smoke_<job>.out`.

The script writes only under `scripts/opd/_smoke_tmp/` and kills the vLLM server
on exit.

## H100 scaling

| Teacher | Server layout | Example serve command | Student side |
|---|---:|---|---|
| Qwen3-32B bf16 | 1 x H100 | `vllm serve Qwen/Qwen3-32B --dtype bfloat16 --port 8000` | Qwen3 1.7B-9B LoRA, typically under 20GB |
| Llama-3.3-70B | 2 x H100, TP=2 | `vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max-model-len 4096` | Llama-3.x 1.7B-9B LoRA, exact tokenizer contract required |
| Qwen3-235B-A22B-FP8 | 2 x H100, TP=2 | `vllm serve Qwen/Qwen3-235B-A22B-FP8 --tensor-parallel-size 2` | Qwen3 1.7B-9B, exact tokenizer contract required |

Keep the teacher and student as separate processes. Scaling up should only
require changing model names and the vLLM tensor-parallel flag.
