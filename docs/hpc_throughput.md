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

## Completed Infrastructure

| Item | Status | Details |
|---|---|---|
| Baseline embedding build (40387) | ✅ Complete | 686,324 docs, gte-large, 2.2h, `chroma_db/` (3.1 GB) |
| Embedding builds (40921) | ✅ Complete | 4 models built on local /tmp, copied to NFS |
| Gemma 4 E4B download | ✅ Complete | 15 GB cached at HF cache |
| Gemma 4 26B-A4B download | ✅ Complete | 49 GB cached at HF cache |
| Gemma 4 venv (vLLM nightly) | ✅ Complete | vLLM 0.19.1rc1 + transformers 5.5.0 |

## Completed Full-Set Results (N=1195 unless noted, barexam)

From `logs/experiments.jsonl` across all branches/machines:

| Model | Size | llm_only | golden_passage | rag_simple | snap_hyde |
|---|---|---|---|---|---|
| deepseek-chat | — | 82.1% (N=28) | 85.7% (N=28) | — | — |
| groq-llama70b | 70B | 72.5% (N=200) | 81.0% (N=100) | 73.0% (N=100) | 76.5% (N=200) |
| or-gemma27b | 27B | 58.0% | 65.5% | 54.6% | — |
| or-qwen3-32b | 32B | 61.4% | 66.7% | 63.1% | — |
| or-qwen3-14b | 14B | 57.7% | — | — | — |
| or-qwen3-8b | 8B | 54.2% | — | — | — |
| groq-qwen3-32b | 32B | 59.3% | — | — | — |
| groq-llama-8b | 8B | 53.0% | — | — | — |
| **cluster qwen3-8b** | **8B** | **52.1%** | **60.1%** | **36.5%** (N=600)* | — |
| **cluster gemma4-e4b** | **8B eff** | **55.5%** | **62.2%** | **54.2%** | **58.6% / 57.9%**† |

*Corrupted by concurrent ChromaDB writes during embedding builds. Not representative.
† Two completed full `rag_snap_hyde` reruns are logged for Gemma 4 E4B; the later rerun on 2026-04-13 finished at **57.9%** (`692/1195`).

### Key Findings

- **Gemma 4 E4B beats Qwen3-8B across all modes**: +3.4pp llm_only, +2.1pp golden, and 6.6x faster
- **Gemma 4 E4B is the best small model tested**: 55.5% llm_only approaches Gemma 27B (58.0%)
- **Golden passage adds ~7pp consistently**: Qwen +8.0pp, Gemma +6.7pp
- **Snap_hyde stays close to the golden ceiling**: Gemma full reruns landed at 58.6% and 57.9% vs golden 62.2% (3.6-4.3pp gap, down from 8.0pp for rag_simple)
- **RAG simple slightly hurts Gemma** (-1.3pp): consistent with models that already know the material
- **Gemma generates 3.7x fewer tokens** (756 vs 2,774 avg) — more concise, still more accurate
- **Speed**: Gemma 12.5s/q vs Qwen 82.6s/q (avg latency from completed runs)

## Embedding Model Comparison (2026-04-09)

All results: Gemma 4 E4B, N=200, seed=42, BarExam, cluster A6000.

| Embedding Model | HF ID | Params | Dim | Max Tok | rag_simple | snap_hyde |
|---|---|---|---|---|---|---|
| **gte-large (baseline)** | Alibaba-NLP/gte-large-en-v1.5 | 434M | 1024 | 8192 | 57.0% | **65.5%** |
| **legal-bert** | nlpaueb/legal-bert-base-uncased | 110M | 768 | 512 | **62.0%** | 60.0% |
| stella-400m | dunzhang/stella_en_400M_v5 | 400M | 1024 | 131K | 61.0% | 60.0% |
| bge-m3 | BAAI/bge-m3 | 568M | 1024 | 8192 | 61.0% | 60.0% |
| jina-v3 | jinaai/jina-embeddings-v3 | 570M | 1024 | 8192 | 61.5% | 64.5% |
| arctic-l-v2 | Snowflake/snowflake-arctic-embed-l-v2.0 | 568M | 1024 | 8192 | 61.5% | 64.5% |
| nomic-v2-moe | nomic-ai/nomic-embed-text-v2-moe | 475M MoE | 768 | 8192 | 61.5% | 64.5% |

### Embedding Comparison Analysis

1. **rag_simple: all alternatives beat baseline** (+4-5pp). legal-bert is best at 62.0%, while the wave-2 embedders all land at 61.5%.
2. **snap_hyde: baseline still dominates** at 65.5%. Wave-1 alternatives converge at 60.0%, while the stronger wave-2 embedders improve to 64.5%.
3. **Asymmetric behavior explained by query type**:
   - `rag_simple` embeds the raw question (question→passage similarity) — legal-bert's domain vocabulary helps here
   - `rag_snap_hyde` embeds an LLM-generated hypothetical passage (passage→passage similarity) — gte-large's general embedding quality wins
4. **Aligned reranking still collapses most embedding differences** — all six alternative embedders converge to 65.0% under `snap_hyde_aligned`, so the cross-encoder remains the dominant bottleneck.
5. **legal-bert is the smallest model** (110M) yet best on rag_simple — domain pretraining > parameter count for question-to-passage matching.
6. **Failed builds are now known, not pending** — `gte-qwen2-1.5b` and `stella-1.5b` failed on the current stack due to transformers `rope_theta` compatibility.

## Candidate Models for Additional HPC Runs

| Model | HF ID | Size (total/effective) | GPU Needed | Measured tok/s | Est. time/q | Notes |
|---|---|---|---|---|---|---|
| Qwen3-8B | Qwen/Qwen3-8B | 8B | A40+ | 40 tok/s | ~75s | Already fully benchmarked; useful as a throughput anchor |
| Gemma 4 E4B | google/gemma-4-E4B-it | 8B/4.5B eff | A40+ (15 GiB) | 61 tok/s | ~10s | Already fully benchmarked; current small-model workhorse |
| Gemma 4 26B-A4B | google/gemma-4-26B-A4B-it | 25B MoE / 4B eff | A40 w/ int8 | TBD | TBD | Cached, still untested locally |
| Qwen3-14B | Qwen/Qwen3-14B | 14B | A40+ | ~25-30 est | ~110s | API baseline exists; no local vLLM run yet |
| Qwen3-32B | Qwen/Qwen3-32B | 32B | A100+ | ~10-15 est | ~220s | API baseline exists; local run would need A100/H100 time |

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
