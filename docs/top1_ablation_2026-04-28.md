# Top-1 vs top-5 retrieval-depth ablation — running

Meeting 2026-04-27 ask #4: "Check lift between passing in top-1 retrieved vs
top-5 retrieved, see if more passages boost or hurt performance."

Implementation: `--retrieval-k` CLI flag landed in commit `b286279`. Default
behavior unchanged (k=5). Pair top-1 vs top-5 within the same provider/seed
to isolate retrieval depth from provider/randomness drift.

## TL;DR — multi-hop QA needs multiple passages, regardless of method

On Llama 3.3 70B × MuSiQue N=200 (paired McNemar within same Groq slot, seed=42):

| Method | top-5 EM | top-1 EM | Δ | McNemar p | b/c (top-5 only / top-1 only) |
|---|---:|---:|---:|---:|---:|
| `rag_simple` | 27.5% | **13.0%** | **-14.5pp** | **4.18e-07** | 32 / 3 |
| `multi_hyde_diverse` | 35.5% | **19.0%** | **-16.5pp** | **5.42e-07** | 39 / 6 |
| `rag_snap_hyde` | (24.0% baseline) | (running) | (running) | (running) | (running) |
| `rag_multi_query` | (29.0% baseline) | (queued) | (queued) | (queued) | (queued) |

Both landed methods drop **-14 to -17pp** when restricted to a single retrieved
passage. Multi-hop reasoning genuinely needs multiple passages — this is not
a method-specific quirk.

**The mhd lift over rag_simple SURVIVES at top-1 (+6pp; 19.0% vs 13.0%) but the
absolute accuracy is dominated by passage count, not method choice.** This means:

- The headline mhd finding is robust to retrieval depth (lift is preserved at
  top-1 in absolute pp terms).
- The headline finding is also somewhat illusory in a different sense: most of
  the multi-hop QA difficulty is "how many passages do I get," not "which
  retrieval method." A k=5 rag_simple beats a k=1 mhd by ~9pp.

Citation-grade rule of thumb: **passage count is a first-order knob; method
choice is a second-order knob** for multi-hop factoid QA on Llama 70B.

## Cross-dataset check (in flight)

SLURM 55452 runs `rag_simple` and `rag_snap_hyde` × {top-1, top-5} on
Gemma 4 26B-A4B × BarExam N=200 via OR-Gemma. Pair within same provider
slot (within-job McNemar). Question: does top-1 hurt legal MC the same way
it hurts multi-hop, or is the BarExam snap-first regime more robust to
retrieval depth?

Hypothesis: smaller drop on BarExam, because snap-first methods rely less
on retrieved passages. The +3pp BarExam `rag_snap_hyde` lift is largely
snap-driven (per `docs/methods_vs_golden_audit_2026-04-27.md`), so cutting
retrieval depth should hurt less. If confirmed, it's a clean cross-dataset
asymmetry: multi-hop QA wants more passages; legal MC tolerates fewer.

## Implementation notes

- `--retrieval-k` plumbed through 28 call sites in `eval/eval_harness.py` (commit
  `b286279`). `_entity_search top_k=30` and first-stage `top_k=15` are
  intentionally unaffected (different retrieval stages).
- The cross-encoder reranker still picks the top-k from a pool of `k*3` first-stage
  candidates, so top-1 means "best single passage after cross-encoder rerank,"
  not "first dense hit."
- Source data: `logs/eval_*_groq-llama70b_20260428_*detail.jsonl` for top-1;
  existing 20260427 logs for top-5 baselines.

## Source-of-truth log paths

| Run | Path |
|---|---|
| Llama 70b MuSiQue rag_simple top-5 baseline | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl` |
| Llama 70b MuSiQue rag_simple top-1 | `logs/eval_rag_simple_groq-llama70b_20260428_0011_detail.jsonl` |
| Llama 70b MuSiQue mhd top-5 baseline | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl` |
| Llama 70b MuSiQue mhd top-1 | `logs/eval_multi_hyde_diverse_groq-llama70b_20260428_0019_detail.jsonl` |
| Llama 70b MuSiQue rag_snap_hyde top-5 baseline | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl` |
| Llama 70b MuSiQue rag_snap_hyde top-1 | (running 2026-04-28) |
| Gemma 4 26B BarExam top-1 vs top-5 | (SLURM 55452, in flight) |
| Gemma 4 26B BarExam snap_hyde_2call vs rag_snap_hyde | (SLURM 55451, in flight) |
