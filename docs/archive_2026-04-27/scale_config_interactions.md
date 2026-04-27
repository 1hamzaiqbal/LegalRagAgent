<!-- ARCHIVED 2026-04-27 — superseded by docs/compiled_results.md -->

# Scale × Config Interactions

> **2026-04-22 prompt-bug fix landed in `f95f316` + `3d5ff05`**: the BarExam
> `prompt` column (37% of questions) was being dropped by both the answer
> formatter AND 11 retrieval/rerank query sites. All numbers below dated
> before `3d5ff05` are **pre-prompt-fix references**. The first apples-to-apples
> validation at N=200 (E4B `rag_simple` 61.5% pre-fix → 61.0% post-fix)
> shows **the accuracy impact is smaller than feared** (within N=200 noise),
> likely because legal MC choices encode the doctrine and the model can
> deduce the fact pattern from them. Full N=1195 clean rerun pending at
> the `clean-rerun-v1` tag.

Analysis of how different retrieval/reasoning methods behave across Gemma 4 model sizes
(E2B → E4B → 26B-A4B → 31B). All numbers are post-leak-fix, BarExam, N=1195, seed=42,
unless noted. Live snapshot — updates as remaining jobs land. See
`docs/size_comparison_matrix.md` for the raw per-job state.

## Current data table (full N=1195)

Values in **bold** are landed; empty cells are in-flight or queued.

| Model | Size / active | rag_simple | rag_hyde | rag_snap_hyde | snap_only | llm_only | golden_passage |
|---|---|---|---|---|---|---|---|
| E2B | 4B eff | **45.4%** | **43.7%** | — | — | — | — |
| E4B | 8B eff | **55.7%** | **57.7%** | **58.4%** | — | — | — |
| 26B-A4B | 25B / 3.8B active | **70.8%** | **74.2%** | **76.6%** | — | **74.3%** | **75.0%** |
| 31B | 31B dense | **79.6%** | **80.4%** | — | — | — | — |

Additional 26B-A4B modes: `subagent_rag` **75.7%**, `subagent_hybrid` **73.4%**.

Seed=99 repeatability (landed so far):
- 26B rag_simple seed=99 = **71.8%** (+1.0pp vs seed=42 70.8%). Tight variance.

N=200 reference points (earlier batch, 0% leak):
- E4B N=200: rag_simple 60.5, rag_hyde 59.5, rag_snap_hyde 66.5, snap_only 64.0
- 31B N=200: rag_simple 79.0, rag_hyde 83.0, rag_snap_hyde 85.0, snap_only 84.0

## Interaction patterns emerging

### 1. `rag_simple` scales cleanly with size
```
E2B 45.4% → E4B 55.7% → 26B 70.8% → 31B 79.6%
```
Monotonic. Model size is the single biggest predictor. Every ~2-4× params adds ~10-15pp.

### 2. HyDE lift is inverted-U across scale
| Model | rag_simple | rag_hyde | HyDE lift |
|---|---|---|---|
| E2B (4B) | 45.4% | 43.7% | **-1.7pp** (HyDE hurts) |
| E4B (8B) | 55.7% | 57.7% | **+2.0pp** |
| 26B (25B) | 70.8% | 74.2% | **+3.4pp** (peak) |
| 31B (31B) | 79.6% | 80.4% | **+0.8pp** |

Clear inverted-U shape. At 4B, the model can't meaningfully use the
retrieved passages and HyDE actively hurts. At 26B, HyDE reaches peak
effectiveness. At 31B, parametric knowledge saturates and the HyDE delta
compresses again.

### 3. `rag_hyde` vs `llm_only`: retrieval stops adding value at 25B
At 26B-A4B: `rag_hyde` **74.2%** ≈ `llm_only` **74.3%**. HyDE retrieves doctrinally-relevant
passages, but the model's parametric knowledge already contains equivalent signal at this
scale. Plain HyDE retrieval contributes **~0pp** over removing retrieval entirely.

### 3b. The ceiling is MUCH lower than we thought — at 26B

- 26B `golden_passage` (oracle, gets the exact gold passage as context): **75.0%**
- 26B `llm_only` (no retrieval): **74.3%**
- Gap: **+0.7pp**. The gold passage barely contributes over parametric knowledge.

This reframes the "retrieval bottleneck" story. At 25B scale, the model already
knows most of the doctrine the gold passage teaches. Retrieval is not the
bottleneck — the problem is the *interpretation/reasoning* phase, not the
information access phase.

Yet `rag_snap_hyde` lands at **76.6%** — **+1.6pp above the oracle ceiling**.
That's only possible if snap+HyDE is adding reasoning structure beyond what a
single gold passage provides: snap forces an explicit doctrinal analysis, HyDE
retrieves multiple topically-aligned passages, the final agent integrates both.

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

### 5. Method stacking collapses at the biggest scale — confirmed at full N=1195
- 31B N=200: rag_simple 79.0% → rag_hyde 83.0% = **+4.0pp**
- 31B N=**1195**: rag_simple 79.6% → rag_hyde 80.4% = **+0.8pp**

The N=200 HyDE lift at 31B was mostly noise. At full scale, HyDE adds basically
nothing to 31B — consistent with the 26B finding that retrieval has almost no
ceiling to climb against parametric knowledge.

Method stacking full picture (so far):
- **E4B**: rag_simple → rag_hyde → rag_snap_hyde = +2.0pp, +0.7pp
- **26B**: +3.4pp, +2.4pp (snap is the only thing that breaks past the parametric ceiling)
- **31B**: +0.8pp, ? (rag_snap_hyde still running)

## Qualitative cross-scale reasoning patterns (2026-04-21)

Codex compared reasoning traces across E2B/E4B/26B/31B on the same BarExam
questions (same seed=42 subset). Three representative cases:

**Case A (mbe_1 evidence law, collateral-matter rule)** — 26B/31B correct (D);
E2B/E4B wrong (B). Smaller models used the loose heuristic "witness opened
the door, attack credibility." Bigger models invoked the specific
"extrinsic-evidence-on-collateral-matter" ban. Scale unlocks doctrinal precision.

**Case B (mbe_1014)** — all 4 sizes wrong with DIFFERENT predictions (D/B/C/C;
correct A). **Root cause is a dataset bug**: the question stem is literally
`"Is Farmer obligated to make the $4,000 payment?"` — missing the whole fact
pattern. Each model hallucinates different missing facts.

**Case C (mbe_1004 lost-volume-seller)** — E2B correct (C); E4B/26B/31B all
wrong (A). Bigger models applied the clean resale-offset formula
("resold at same price → no loss"), missing the lost-volume-seller exception.
**Smaller model caught a cue the bigger models pattern-matched past.**

Read: scale monotonically improves doctrinal framing + rule invocation, but
can overcommit to a clean rule when an exception applies. The rag_snap_hyde
mode likely helps here — snap reasoning raises alternative frames before
HyDE retrieval narrows the passage pool.

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
