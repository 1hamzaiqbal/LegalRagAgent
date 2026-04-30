# Top-1 vs top-5 retrieval-depth ablation — landed partial

Meeting 2026-04-27 ask #4: "Check lift between passing in top-1 retrieved vs
top-5 retrieved, see if more passages boost or hurt performance."

Implementation: `--retrieval-k` CLI flag landed in commit `b286279`. Default
behavior unchanged (k=5). Pair top-1 vs top-5 within the same provider/seed
to isolate retrieval depth from provider/randomness drift.

## TL;DR — retrieval-depth sensitivity is a task-regime signal

On Llama 3.3 70B × MuSiQue N=200 (paired McNemar within same Groq slot, seed=42):

| Method | top-5 EM | top-1 EM | Δ | McNemar p | b/c (top-5 only / top-1 only) |
|---|---:|---:|---:|---:|---:|
| `rag_simple` | 27.5% | **13.0%** | **-14.5pp** | **4.18e-07** | 32 / 3 |
| `multi_hyde_diverse` | 35.5% | **19.0%** | **-16.5pp** | **5.42e-07** | 39 / 6 |
| `rag_snap_hyde` | 24.0% | **14.0%** | **-10.0pp** | **0.0012** | 28 / 8 |
| `rag_multi_query` | 29.0% | **14.0%** | **-15.0pp** | **5.30e-06** | 37 / 7 |

**Sub-finding: snap-first methods are more retrieval-depth-robust.** `rag_snap_hyde` lost
only -10pp at top-1 vs -14.5 to -16.5pp for the three "no-snap" methods. Even on MuSiQue
(where snap_hyde is cross-domain negative at top-5), the snap-first prior absorbs the
retrieval-depth shock better. Mechanism: snap reasoning happens *before* retrieval, so
cutting passages from 5 to 1 hurts the final synthesis less when the model has already
committed to a prior answer it can defend.

**Method clustering at top-1**: all 4 methods land at 13-19% top-1 EM (vs 24-35.5% top-5).
The `mhd` lift over `rag_simple` shrinks from +8pp at top-5 to +6pp at top-1 — the
method-choice effect persists in *direction* but is dominated in *absolute terms* by
the passage-count effect. Cleanest paper framing:

> On multi-hop QA, retrieval depth (top-k) is a first-order knob; method choice
> (rag_simple / mhd / multi_query / snap_hyde) is a second-order knob. The
> headline mhd lift is robust across retrieval depths, but the absolute accuracy
> ceiling at top-1 (~19%) is far below any method's top-5 ceiling (~35%).

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

New 2026-04-30 check: LegalBench-SCALR also shows a catastrophic top-1 drop,
but with a different interpretation from MuSiQue. On Llama 70B × SCALR N=200,
`rag_simple` top-5 is 77.0% while top-1 is 59.5% (-17.5pp, McNemar p=1.05e-08,
b/c=3/38). Gold-hit rate falls from 54.0% to 32.5%. SCALR is therefore
retrieval-depth limited even though the `snap_hyde_2call` query-formulation
intervention is flat/negative on the same top-5 slice.

This matters for the taxonomy: "legal MC" is not one regime. BarExam is flat
under top-1/top-5, while SCALR holding selection needs a candidate set. The
depth ablation is not merely measuring legal-vs-nonlegal; it is detecting
whether the task is candidate-retrieval limited.

## Cross-dataset check (landed for `rag_simple`)

SLURM 55452 landed the `rag_simple` top-1/top-5 pair on Gemma 4 26B-A4B ×
BarExam N=200 via OR-Gemma. `rag_simple` is flat: top-1 83.0% versus top-5
82.5% (-0.5pp, McNemar p=1.00 NS). This is the clean cross-dataset contrast
against MuSiQue's -14.5pp `rag_simple` drop. The regular `rag_snap_hyde`
top-1 detail log is not present locally; do not cite that depth comparison
until it lands.

Related SLURM 55451 result: `snap_hyde_2call` on the same BarExam N=200 slice
landed at 85.5% versus the top-5 `rag_simple` 82.5% (+3.0pp, p=0.377 NS,
parse_ok=200/200). That preserves the direction of the Tier 3 `rag_snap_hyde`
lift but is directional, not independently significant.

LegalBench-SCALR now gives the opposite cross-dataset legal result: top-1
collapses to 59.5% from top-5 77.0% (-17.5pp, p=1.05e-08). Because
`snap_hyde_2call` is 75.0% on the same SCALR top-5 slice (-2.0pp vs
`rag_simple`, p=0.503), this isolates **retrieval depth** rather than
pseudo-document query formulation as the useful intervention.

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
| Llama 70b MuSiQue rag_snap_hyde top-1 | `logs/eval_rag_snap_hyde_groq-llama70b_20260428_0025_detail.jsonl` |
| Gemma 4 26B BarExam rag_simple top-1 | `logs/eval_rag_simple_or-gemma4-26b_20260428_0138_detail.jsonl` |
| Gemma 4 26B BarExam rag_simple top-5 | `logs/eval_rag_simple_or-gemma4-26b_20260428_0231_detail.jsonl` |
| Gemma 4 26B BarExam rag_snap_hyde top-5 | `logs/eval_rag_snap_hyde_or-gemma4-26b_20260428_0257_detail.jsonl` |
| Gemma 4 26B BarExam snap_hyde_2call N=200 | `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260428_1435_detail.jsonl` |
| Llama 70b LegalBench-SCALR rag_simple top-5 | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl` |
| Llama 70b LegalBench-SCALR rag_simple top-1 | `logs/eval_rag_simple_groq-llama70b_20260429_2159_detail.jsonl` |
| Llama 70b LegalBench-SCALR snap_hyde_2call N=200 | `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_1520_detail.jsonl` |
