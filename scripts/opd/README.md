# OPD scaffold for agentic retrieval skills

This directory contains a minimal on-policy distillation (OPD) scaffold for
internalizing agentic retrieval behavior from a larger teacher into a smaller
student. It follows the design in `wiki/concepts/skill-distillation-bridge.md`:
student completions are sampled on-policy, the teacher scores those exact
completion tokens, and the student is updated with a policy-gradient reverse-KL
surrogate. The branch also implements SDAR-style negative-gap gating as a
stability safeguard. This is not a complete agentic-RL method: task reward and
the retrieval environment still need to be combined with the dense objective.

## Current evidence boundary

- EIT job 93802 validated bare OPD plumbing: Qwen3-8B teacher,
  Qwen3-1.7B student, three finite steps, checkpoints written.
- That smoke is not evidence of task learning.
- SDAR reports that standalone/naively mixed on-policy self-distillation can
  collapse. Use `opd_gated` for the next plumbing test, and do not launch a
  scientific E3 until the teacher skill-gap A/B is positive and task reward is
  integrated.

## Tokenizer constraint

Teacher and student MUST share a tokenizer family. OPD aligns teacher and
student per-token logprobs, so mismatched tokenizers make the objective invalid.

Use pairs such as:

- Qwen3 family teacher plus Qwen3 family student.
- Llama-3.x family teacher plus Llama-3.x family student.

Do not mix Qwen and Llama, or any pair where `tokenizer.encode(prompt)` and the
completion tokenization are not aligned across teacher and student.

## Three-process layout

Process 1: teacher server, OpenAI-compatible vLLM endpoint.

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 2048
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
  --mode opd_gated \
  --task-file tasks.jsonl \
  --student Qwen/Qwen3-1.7B \
  --teacher-url http://127.0.0.1:8000 \
  --teacher-model Qwen/Qwen3-8B \
  --out-dir runs/opd-qwen17b \
  --steps 100 \
  --group-size 4 \
  --max-new-tokens 128
```

`opd_gated` multiplies each token update by
`sigmoid(beta * (teacher_logp - student_logp))`; `--gap-gate-beta` defaults to
`5.0`. `--mode opd` retains the historical bare objective as a diagnostic arm.

Process 3: task data as JSONL. Each row must contain `prompt_text`.

```jsonl
{"prompt_text": "Question and retrieval-state prompt goes here."}
{"prompt_text": "Another prompt."}
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

## Batching and checkpoints

- `JUDGE_MICRO` controls prompts per optimizer step, matching the compact judge
  lane convention. Default: `1`.
- `--group-size` controls completions sampled per prompt. Default: `4`.
- `--save-every` writes `step_XXXXXX/` LoRA checkpoints.
- The final checkpoint is written to `final/`.

## A100 smoke test

The smoke test requires the judge-lane venv plus `pip install vllm`. It starts a
local Qwen3-8B teacher, generates eight tiny prompts, runs three OPD steps with a
Qwen3-1.7B student, checks that the logged loss is finite, and confirms that a
checkpoint exists.

```bash
bash scripts/opd/smoke_test.sh
```

The default is now the gated plumbing mode. Set `OPD_SMOKE_MODE=opd` only to
reproduce the historical bare-OPD smoke.

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
| Llama-3.3-70B | 2 x H100, TP=2 | `vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --port 8000` | Llama-3.x 1.7B-9B LoRA, same tokenizer family, typically under 20GB |
| Qwen3-235B-A22B-FP8 | 2 x H100, TP=2 | `vllm serve Qwen/Qwen3-235B-A22B-FP8 --tensor-parallel-size 2 --port 8000` | Qwen3 1.7B-9B LoRA, same tokenizer family, typically under 20GB |

Keep the teacher and student as separate processes. Scaling up should only
require changing model names and the vLLM tensor-parallel flag.
