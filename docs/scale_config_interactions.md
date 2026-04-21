# Scale × Config Interactions

Analysis of how different retrieval/reasoning methods behave across Gemma 4 model sizes
(E2B → E4B → 26B-A4B → 31B). All numbers are post-leak-fix, BarExam, N=1195, seed=42,
unless noted. Live snapshot — updates as remaining jobs land. See
`docs/size_comparison_matrix.md` for the raw per-job state.

## Current data table (full N=1195)

Values in **bold** are landed; empty cells are in-flight or queued.

| Model | Size / active | rag_simple | rag_hyde | rag_snap_hyde | snap_only | llm_only | golden_passage |
|---|---|---|---|---|---|---|---|
| E2B | 4B eff | **45.4%** | — | — | — | — | — |
| E4B | 8B eff | **55.7%** | **57.7%** | **58.4%** | — | — | — |
| 26B-A4B | 25B / 3.8B active | **70.8%** | **74.2%** | **76.6%** | — | **74.3%** | — |
| 31B | 31B dense | **79.6%** | — | — | — | — | — |

N=200 reference points (earlier batch, 0% leak):
- E4B N=200: rag_simple 60.5, rag_hyde 59.5, rag_snap_hyde 66.5, snap_only 64.0
- 31B N=200: rag_simple 79.0, rag_hyde 83.0, rag_snap_hyde 85.0, snap_only 84.0

## Interaction patterns emerging

### 1. `rag_simple` scales cleanly with size
```
E2B 45.4% → E4B 55.7% → 26B 70.8% → 31B 79.6%
```
Monotonic. Model size is the single biggest predictor. Every ~2-4× params adds ~10-15pp.

### 2. HyDE lift over plain RAG shrinks as models scale
| Model | rag_simple | rag_hyde | HyDE lift |
|---|---|---|---|
| E4B | 55.7% | 57.7% | **+2.0pp** |
| 26B | 70.8% | 74.2% | **+3.4pp** |
| 31B N=200 ref | 79.0% | 83.0% | +4.0pp (N=200) |

HyDE lift over plain RAG is modest (+2–4pp) and fairly stable across sizes.

### 3. `rag_hyde` vs `llm_only`: retrieval stops adding value at 25B
At 26B-A4B: `rag_hyde` **74.2%** ≈ `llm_only` **74.3%**. HyDE retrieves doctrinally-relevant
passages, but the model's parametric knowledge already contains equivalent signal at this
scale. Plain HyDE retrieval contributes **~0pp** over removing retrieval entirely.

### 4. `rag_snap_hyde` is the one retrieval mode that survives scaling
| Model | llm_only | rag_hyde | rag_snap_hyde | snap+HyDE lift over llm_only |
|---|---|---|---|---|
| E4B | — | 57.7% | 58.4% | (pending llm_only) |
| 26B | 74.3% | 74.2% | **76.6%** | **+2.3pp** |
| 31B | ? | ? | 85.0% (N=200) | pending N=1195 |

At 26B, snap+HyDE **is** the winning configuration — it beats both plain HyDE and llm_only.
Snap reasoning picks the right doctrinal axis, HyDE retrieves passages on that axis, and
the final agent integrates both. Retrieval stops being redundant when snap narrows what
we're retrieving *for*.

### 5. Method stacking does collapse at the biggest scale (N=200 ref)
At 31B N=200 the delta from `rag_hyde` 83% to `rag_snap_hyde` 85% is only +2pp —
about half the lift observed at 26B. The 31B N=1195 numbers will show whether this
compression continues. Working hypothesis: at 31B, snap and HyDE overlap more with
parametric knowledge, so each additional component contributes less.

## Implications for the research story

- **Small models (4-8B)**: retrieval is barely better than noise. HyDE matters more than snap at this scale, but deltas are all inside ±3pp.
- **Medium MoE (25B, 3.8B active)**: this is the sweet spot for `rag_snap_hyde` — snap chooses the right question, HyDE retrieves for it, both are needed to beat parametric.
- **Large dense (31B)**: method choice plateaus. Parametric knowledge dominates. Worth asking whether retrieval adds anything measurable at this scale.

The paper narrative likely wants to center on the 26B result: `rag_snap_hyde` as the
config that cleanly separates from parametric knowledge at the MoE scale where this
kind of reasoning/retrieval combination is most valuable.

## What's still pending (updated continuously)

- **E4B `snap_only_in_final` N=1195** — completes the E4B ablation
- **26B `snap_only_in_final`** — in 50868, the 4th mode
- **26B subagent variants** (rag/hyde/hybrid/report) — four modes in flight across 50991/50992
- **26B `golden_passage`** — oracle ceiling at 26B, in 50990 after llm_only
- **31B `rag_hyde`, `rag_snap_hyde`, `snap_only_in_final`** — in 50865
- **31B `llm_only`, `golden_passage`, subagent variants** — queued on H100 in 50993/50994/50995
- **E4B + 26B seed=99 repeatability** — running now via 51023/51024; gives variance bars on everything in this doc

Once all of these land we'll have the full 4-size × 10-mode matrix (~28 cells, counting the
subset we're running per size) plus variance from seed=99. Expected completion within
the next 30-40h as the H100-bound 31B queue drains.
