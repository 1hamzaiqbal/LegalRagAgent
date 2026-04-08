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

## Active HPC Runs (2026-04-08)

| Job ID | Mode | Model | GPU/Node | Questions | Progress | Accuracy | Submitted |
|---|---|---|---|---|---|---|---|
| 40248 | llm_only | Qwen3-8B | A40 / a40-2206 | 1195 | 980/1195 (82%) | 52.1% | 2026-04-07 |
| 40249 | golden | Qwen3-8B | A40 / a40-2206 | 1195 | 1041/1195 (87%) | 59.7% | 2026-04-07 |
| 40687 | rag_simple | Qwen3-8B | A6000 / a60-2209 | 600 | 89/600 (15%) | 51.6% | 2026-04-08 |
| 40707 | llm_only | Gemma 4 E4B | A6000 / a60-2208 | 1195 | 130/1195 (11%) | 52.3% | 2026-04-08 |
| 40731 | golden | Gemma 4 E4B | A6000 / a60-2209 | 1195 | 30/1195 (3%) | — | 2026-04-08 |
| 40732 | rag_simple | Gemma 4 E4B | A6000 / a60-2209 | 1195 | 23/1195 (2%) | — | 2026-04-08 |
| 40736 | snap_hyde | Gemma 4 E4B | A6000 / a60-2209 | 1195 | starting | — | 2026-04-08 |

SLURM logs: `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/{jobid}.out`

## Completed Infrastructure

| Item | Status | Details |
|---|---|---|
| Embedding build (40387) | ✅ Complete | 686,324 docs, 2.2h, ChromaDB at `chroma_db/` (3.1 GB) |
| Gemma 4 E4B download | ✅ Complete | 15 GB cached at HF cache |
| Gemma 4 26B-A4B download | ✅ Complete | 49 GB cached at HF cache |
| Gemma 4 venv (vLLM nightly) | ✅ Complete | vLLM 0.19.1rc1 + transformers 5.5.0 |

## Completed Full-Set Results (N=1195, barexam)

From `logs/experiments.jsonl` across all branches/machines:

| Model | Size | llm_only | golden_passage | rag_simple | rag_snap_hyde |
|---|---|---|---|---|---|
| deepseek-chat | — | 82.1% (N=28) | 85.7% (N=28) | — | — |
| groq-llama70b | 70B | 72.5% (N=200) | 81.0% (N=100) | 73.0% (N=100) | — |
| or-gemma27b | 27B | 58.0% | 65.5% | 54.6% | — |
| or-qwen3-32b | 32B | 61.4% | 66.7% | 63.1% | — |
| or-qwen3-14b | 14B | 57.7% | — | — | — |
| or-qwen3-8b | 8B | 54.2% | — | — | — |
| groq-qwen3-32b | 32B | 59.3% | — | — | — |
| groq-llama-8b | 8B | 53.0% | — | — | — |
| **cluster qwen3-8b** | **8B** | **~52% (in prog)** | **~60% (in prog)** | **~52% (in prog)** | — |
| **cluster gemma4-e4b** | **8B eff** | **~52% (in prog)** | **(in prog)** | **(in prog)** | **(in prog)** |

Notes:
- Cluster results use local vLLM on HPC, not API. Qwen validates against or-qwen3-8b API (54.2%)
- Golden passage adds ~8pp over llm_only consistently
- Gemma 4 E4B generates much shorter answers (61 tok/s, ~10s/q) vs Qwen3-8B (40 tok/s, ~75s/q)
- RAG modes require ChromaDB embeddings (built by job 40387, 686K docs)

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
