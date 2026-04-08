# HPC Throughput & Experiment Tracking

Reference for estimating wall-clock time and planning cluster experiments.

## Throughput Profile: Qwen3-8B on A40 (fp16, vLLM 0.19.0)

Measured from job 40248/40249 on `a40-2206` (NVIDIA A40 48GB), 2026-04-07.

| Metric | llm_only | golden_passage |
|---|---|---|
| LLM calls/question | 1 | 1 |
| Input tokens (avg) | ~230 | ~420 |
| Output tokens (avg) | ~3,000 | ~2,800 |
| Output chars (avg) | 12,000 | 11,200 |
| Time/question (avg) | 78.6s | 71.8s |
| Time/question (range) | 14–242s | 11–240s |
| Generation throughput | ~40 tok/s | ~40 tok/s |
| vLLM startup overhead | ~15 min | ~15 min |

### Timing Formula (A40, single GPU, fp16)

```
wall_time ≈ (N_questions × avg_output_tokens × calls_per_mode) / 40 tok/s + 15min startup
```

### Quick Reference

| N questions | 1-call mode | 2-call mode (e.g. arb) | Est. with startup |
|---|---|---|---|
| 10 | ~13 min | ~26 min | ~28 / 41 min |
| 50 | ~1.1h | ~2.1h | ~1.3h / 2.4h |
| 200 | ~4.2h | ~8.3h | ~4.5h / 8.5h |
| 1195 (full) | ~24.9h | ~49.7h | ~25h / 50h |

### Scaling to Other Models (rough estimates)

Output throughput is the bottleneck. Approximate tok/s by model size on single GPU:

| Model Size | GPU | Est. tok/s (fp16) | Est. time/q (3K output) |
|---|---|---|---|
| 8B | A40 (48GB) | ~40 | ~75s |
| 8B | A100 (80GB) | ~50-60 | ~55s |
| 14B | A40 (48GB) | ~25-30 | ~110s |
| 27-32B | A100 (80GB) | ~10-15 | ~220s |
| 27-32B | H100 (80GB) | ~20-25 | ~130s |

These are rough — actual throughput depends on model architecture, quantization, and KV cache pressure.

## Active HPC Runs

| Job ID | Mode | Model | GPU/Node | Questions | Status | Submitted |
|---|---|---|---|---|---|---|
| 40248 | llm_only | Qwen3-8B | A40 / a40-2206 | full (1195) | RUNNING | 2026-04-07 |
| 40249 | golden_passage | Qwen3-8B | A40 / a40-2206 | full (1195) | RUNNING | 2026-04-07 |

SLURM logs: `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/{jobid}.out`

## Completed Full-Set Results (N=1195, barexam)

From `logs/experiments.jsonl` across all branches/machines:

| Model | Size | llm_only | golden_passage | rag_simple |
|---|---|---|---|---|
| deepseek-chat | — | 82.1% (N=28) | 85.7% (N=28) | — |
| groq-llama70b | 70B | 72.5% (N=200) | 81.0% (N=100) | 73.0% (N=100) |
| or-gemma27b | 27B | 58.0% | 65.5% | 54.6% |
| or-qwen3-32b | 32B | 61.4% | 66.7% | 63.1% |
| or-qwen3-14b | 14B | 57.7% | — | — |
| or-qwen3-8b | 8B | 54.2% | — | — |
| groq-qwen3-32b | 32B | 59.3% | — | — |
| groq-llama-8b | 8B | 53.0% | — | — |
| **cluster-vllm qwen3-8b** | **8B** | **~55% (in progress)** | **~60% (in progress)** | — |

Notes:
- DeepSeek and groq-llama70b results are from API runs on other branches (N=28-200, not full set)
- The cluster-vllm results are from local vLLM on HPC, not API — validates against or-qwen3-8b (54.2%)
- Golden passage typically adds 5-10% over llm_only for the same model

## Candidate Models for Next HPC Runs

| Model | HF ID | Size (total/effective) | GPU Needed | Est. time/q | Notes |
|---|---|---|---|---|---|
| Qwen3-8B | Qwen/Qwen3-8B | 8B | A40+ | ~75s | Current run |
| Gemma 4 E4B | google/gemma-4-E4B-it | 8B/4.5B eff | A40+ (~16GB fp16) | ~50-75s | 128K ctx, multimodal, vLLM support TBD |
| Gemma 4 E2B | google/gemma-4-E2B-it | ~4B/2B eff | Any GPU | ~30-40s | Smallest Gemma 4 |
| Qwen3-14B | Qwen/Qwen3-14B | 14B | A40+ | ~110s | Next size up |
| Gemma-3-27B | google/gemma-3-27b-it | 27B | A100+ (~54GB fp16) | ~220s | Need A100/H100 |
| Qwen3-32B | Qwen/Qwen3-32B | 32B | A100+ (~64GB fp16) | ~220s | Tight on A100 |

## Monitoring Commands

```bash
# Job status
ssh wustl "squeue -u hiqbal"

# Progress + accuracy
ssh wustl "grep -c 'PASS\|FAIL' /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/{JOBID}.out"
ssh wustl "grep -c PASS /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/{JOBID}.out"

# Timing stats from a log
ssh wustl "grep -oP '\((\d+\.\d+)s' /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/{JOBID}.out | sed 's/(//;s/s//' | awk '{sum+=\$1; count++} END{print \"avg=\"sum/count\"s N=\"count}'"

# vLLM throughput
ssh wustl "tail -5 /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/vllm_qwen3_8b_{JOBID}.log"
```
