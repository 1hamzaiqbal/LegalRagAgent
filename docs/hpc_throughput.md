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

## Throughput Profile: Gemma 4 E4B on A6000 (bf16, vLLM 0.19.1rc1)

Measured from job 40707 on `a60-2208` (NVIDIA A6000 48GB), 2026-04-08.

| Metric | llm_only |
|---|---|
| LLM calls/question | 1 |
| Output tokens (avg) | ~700 |
| Output chars (avg) | ~3,000–5,000 |
| Time/question (avg) | ~10s |
| Time/question (range) | 5–27s |
| Generation throughput | ~61 tok/s |
| vLLM startup overhead | ~5 min |
| GPU memory used | 15.16 GiB |

Much shorter answers than Qwen3-8B (~3K chars vs ~12K chars) → dramatically faster per question.

### Timing Estimates (Gemma 4 E4B, A6000, bf16)

| N questions | 1-call mode | 3-call mode (snap_hyde) | Est. with startup |
|---|---|---|---|
| 50 | ~8 min | ~25 min | ~13 / 30 min |
| 200 | ~33 min | ~1.7h | ~38 min / 1.8h |
| 1195 (full) | ~3.3h | ~10h | ~3.5h / 10h |

## Still Running (2026-04-08 evening)

| Job ID | Mode | Model | GPU/Node | Questions | Progress | Accuracy |
|---|---|---|---|---|---|---|
| 40687 | rag_simple | Qwen3-8B | A6000 / a60-2209 | 600 | 388/600 (65%) | 52.0% |
| 40736 | snap_hyde | Gemma 4 E4B | A6000 / a60-2209 | 1195 | 627/1195 (52%) | 60.1% |

SLURM logs: `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/{jobid}.out`

## Completed Infrastructure

| Item | Status | Details |
|---|---|---|
| Embedding build (40387) | ✅ Complete | 686,324 docs, 2.2h, ChromaDB at `chroma_db/` (3.1 GB) |
| Gemma 4 E4B download | ✅ Complete | 15 GB cached at HF cache |
| Gemma 4 26B-A4B download | ✅ Complete | 49 GB cached at HF cache |
| Gemma 4 venv (vLLM nightly) | ✅ Complete | vLLM 0.19.1rc1 + transformers 5.5.0 |

## Completed Full-Set Results (N=1195 unless noted, barexam)

From `logs/experiments.jsonl` across all branches/machines:

| Model | Size | llm_only | golden_passage | rag_simple | rag_snap_hyde |
|---|---|---|---|---|---|
| deepseek-chat | — | 82.1% (N=28) | 85.7% (N=28) | — | — |
| groq-llama70b | 70B | 72.5% (N=200) | 81.0% (N=100) | 73.0% (N=100) | 76.5% (N=200) |
| or-gemma27b | 27B | 58.0% | 65.5% | 54.6% | — |
| or-qwen3-32b | 32B | 61.4% | 66.7% | 63.1% | — |
| or-qwen3-14b | 14B | 57.7% | — | — | — |
| or-qwen3-8b | 8B | 54.2% | — | — | — |
| groq-qwen3-32b | 32B | 59.3% | — | — | — |
| groq-llama-8b | 8B | 53.0% | — | — | — |
| **cluster qwen3-8b** | **8B** | **52.1%** | **60.1%** | **~52% (N=600 in prog)** | — |
| **cluster gemma4-e4b** | **8B eff** | **55.5%** | **62.2%** | **54.2%** | **~60% (in prog)** |

### Key Findings

- **Gemma 4 E4B beats Qwen3-8B across all modes**: +3.4pp llm_only, +2.1pp golden, and 6.6x faster
- **Gemma 4 E4B is the best small model tested**: 55.5% llm_only beats Qwen3-14B (57.7% via API) and approaches Gemma 27B (58.0%)
- **Golden passage adds ~7pp consistently**: Qwen +8.0pp, Gemma +6.7pp
- **RAG simple slightly hurts Gemma** (-1.3pp): consistent with models that already know the material
- **Gemma snap_hyde tracking at 60.1%** — nearly closing the gap to golden (62.2%), suggesting HyDE retrieval is effective for this model
- **Gemma generates 3.7x fewer tokens** (756 vs 2,774 avg) — more concise, still more accurate
- **Speed**: Gemma 12.5s/q vs Qwen 82.6s/q (avg latency from completed runs)

## Candidate Models for Next HPC Runs

| Model | HF ID | Size (total/effective) | GPU Needed | Measured tok/s | Est. time/q | Notes |
|---|---|---|---|---|---|---|
| Qwen3-8B | Qwen/Qwen3-8B | 8B | A40+ | 40 tok/s | ~75s | Verbose output |
| Gemma 4 E4B | google/gemma-4-E4B-it | 8B/4.5B eff | A40+ (15 GiB) | 61 tok/s | ~10s | Concise output, needs split-venv |
| Gemma 4 26B-A4B | google/gemma-4-26B-A4B-it | 25B MoE / 4B eff | A40 w/ int8 | TBD | TBD | Cached, needs quantization |
| Qwen3-14B | Qwen/Qwen3-14B | 14B | A40+ | ~25-30 est | ~110s | Not yet tested |
| Qwen3-32B | Qwen/Qwen3-32B | 32B | A100+ | ~10-15 est | ~220s | Tight on A100 |

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
