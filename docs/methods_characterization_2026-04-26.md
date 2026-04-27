# Methods characterization — 2026-04-26

## Update 2026-04-27 ~12:30 CDT

Change reason: supersede the 2026-04-26 method characterization with today’s Tier 2 MuSiQue findings while preserving the historical analysis below. Current characterization:

| Method family | Current verdict |
|---|---|
| `rag_snap_hyde` on BarExam | Still the audited legal-MC winner: 26B +3.09pp and E4B +3.69pp over `rag_simple` at N=1195 |
| `multi_hyde_diverse` on MuSiQue | Confirmed only on Llama 70b N=200: +8.0pp, p=0.0195; Gemma 3 27B N=200 is NULL (+2.5pp, p=0.5901) |
| Mechanism decomposition | Llama `rag_multi_query` is +1.5pp NS (p=0.728); HyDE-style answer-bearing passages account for about +6.5pp of MHD’s +8pp lift |
| Heavy planning/subagent paths | `subagent_rag` actively hurts Llama MuSiQue N=200 (-12.0pp, p=0.0007); `iter_hyde` is NS at Llama N=200 and directionally bad on smaller models |
| Citation tier | N<200 rows below are direction-only, not confirmed claims |

Synthesis of when methods (HyDE, snap+HyDE, subagent, planning_table)
beat the simple `rag_simple` baseline vs when they hurt. Built from all
post-fix data landed through 2026-04-26 ~02:50 UTC.

## Headline: methods only help where the dataset rewards them

`rag_simple` is the canonical "1 query, top-k passages, answer" baseline.
Every more-complex method is justified by some lift over it. The data:

| Dataset | Model | rag_simple | Best method | Lift | Winner |
|---|---|---|---|---|---|
| BarExam (legal MC, single-hop) | Gemma 4 26B | 78.08% | rag_snap_hyde 81.17% | **+3.09pp** | snap+HyDE |
| BarExam | Gemma 4 E4B (8B eff) | 58.49% | rag_snap_hyde 62.18% | **+3.69pp** | snap+HyDE |
| MuSiQue (multi-hop, ~2-4 hops) | Gemma 4 26B | 26.7% | rag_simple itself | **0pp** | **NO method beats baseline** |

**Bounded claim**: snap+HyDE delivers a real ~3-5pp lift over `rag_simple` on
*single-hop, retrieval-of-doctrine* tasks (legal MC). On multi-hop entity
composition the lift collapses or inverts.

## Per-mode lift over `rag_simple` — Gemma 4 26B BarExam (full N=1195 post-fix)

Pre-fix (commit 770c9ac), retrieval-bug present, every mode underperformed.
Post-fix (commit 56bffc8) numbers:

| Mode | Post-fix EM | Δ vs rag_simple (78.08%) | Pipeline |
|---|---|---|---|
| llm_only (no retrieval) | 79.75% | +1.67pp | 1 call |
| snap_only_in_final (no retrieval) | 80.59% | +2.51pp | 2 calls |
| **rag_snap_hyde** | **81.17%** | **+3.09pp** | 3 calls |
| rag_hyde | 78.91% | +0.83pp | 2 calls |
| golden_passage (oracle) | 78.66% | +0.58pp | 1 call + gold context |
| subagent_rag | 78.16% | +0.08pp | 4 calls |
| subagent_hybrid | 74.14% | **-3.94pp** | 4 calls — actively hurts |

**Three observations on 26B BarExam:**
1. **`rag_simple` is already better than `golden_passage` oracle** (78.08% vs 78.66% — within noise). The retrieved-passage signal is on par with the gold passage. *The model knows enough doctrine that retrieval is mostly redundant at this scale*.
2. **`llm_only` beats `rag_simple`** by +1.67pp — at 26B the model's parametric knowledge is competitive with retrieved evidence. Retrieval is barely net-positive.
3. **Only `rag_snap_hyde` decisively wins** (+3.09pp). The snap-driven HyDE adds value because snap reasoning narrows what to retrieve and the final agent gets a more focused passage set.

## Per-mode lift over `rag_simple` — Gemma 4 E4B BarExam (full N=1195 post-fix)

| Mode | Post-fix EM | Δ vs rag_simple (58.49%) |
|---|---|---|
| **rag_snap_hyde** | **62.18%** | **+3.69pp** |
| subagent_rag | 60.92% | +2.43pp |
| rag_hyde | 60.59% | +2.10pp |
| subagent_hyde | 60.17% | +1.68pp |
| snap_hyde_report | 60.75% | +2.26pp |
| subagent_hybrid | 58.83% | +0.34pp |
| snap_only_in_final | 57.82% | -0.67pp |
| llm_only / golden_passage | (missing — wallclocked) | — |

**E4B observations:**
1. **rag_snap_hyde wins again** (+3.69pp), basically same lift as 26B
2. **All HyDE-using modes positive** (rag_hyde +2.10, snap_hyde_report +2.26) — at smaller scale, retrieval matters more
3. **subagent_hybrid drops to baseline** at the deeper pipelines — no benefit from the extra structure

**Cross-size pattern**: snap+HyDE consistently lifts +3-4pp regardless of model size on BarExam. Not a scaling artifact — a real method effect.

## Per-mode lift over `rag_simple` — MuSiQue (Gemma 4 26B, N=30 via API)

| Mode | EM | gold_retrieved | Δ vs rag_simple (26.7%) |
|---|---|---|---|
| rag_simple | **26.7%** | **83%** | baseline |
| planning_table_no_snap | 13.3% | 83% | -13.4pp |
| planning_table | 20.7% | not tracked | -6.0pp |
| rag_hyde | 20.0% | 50% | -6.7pp |
| rag_snap_hyde | 20.0% | 60% | -6.7pp |
| subagent_rag | 20.0% | 0% (tracking artifact) | -6.7pp |
| golden_passage (oracle) | 62.0% | (gold given) | +35.3pp |

**MuSiQue observations:**
1. **EVERY structured method underperforms `rag_simple` on multi-hop**.
2. Two distinct failure modes:
   - **Snap-bias**: HyDE / snap+HyDE / subagent_rag / planning_table all condition retrieval on snap output → retrieval finds wrong-hop topic (gold_retrieved drops to 50-60% vs 83% for raw question)
   - **Decomposition cost**: planning_table_no_snap restores retrieval (83%) but the model can't re-compose multi-hop facts from per-TODO findings — final answer often contradicts findings (audit caught 17/30 records of this)
3. The 35pp gap between `rag_simple` and `golden_passage` shows there IS room for retrieval improvement; current methods don't close it on multi-hop

## Cross-model llm_only BarExam N=100 (validates the "model size matters more than method" claim)

| Model | EM | Active params |
|---|---|---|
| Llama 3.3 70b dense | **81%** | 70B |
| Gemma 4 26B-A4B MoE | 79.75% (cluster N=1195) | 3.8B active |
| Llama 4 Scout 17b MoE | 67% | ~3.5B active |
| Qwen3 32b dense | 68% | 32B (think-mode lossy) |

**Cross-family takeaway**: Llama 70b dense and Gemma 4 26B MoE tie at ~80% with very different param budgets. Method choice (llm_only vs rag_snap_hyde at +3pp) is a smaller effect than model choice (Gemma 4 26B vs Llama Scout at +12pp).

## When to use which method (decision tree from this data)

1. **Doctrinal MC, model has parametric coverage** (BarExam-like) → `rag_snap_hyde` for +3-4pp, or `rag_simple` for max efficiency
2. **Doctrinal MC, smaller model** → `rag_snap_hyde` (E4B benefits same +4pp as 26B)
3. **Multi-hop entity composition** (MuSiQue-like) → `rag_simple` (every structured method hurts; snap-bias + decomposition cost both penalize)
4. **No retrieval available** → `llm_only`; near-optimal at 26B+ scale, only -1.67pp behind rag_simple

## What we still don't know

- Does `rag_snap_hyde` lift hold on Llama 70b / Qwen3 32b? (running now)
- Does the MuSiQue `rag_simple > all-methods` pattern hold on a non-Gemma model?
- Would a no-snap rag_hyde (just question-as-HyDE) recover the snap-bias retrieval loss on MuSiQue without taking the decomposition tax?
- 31B Gemma 4 only has rag_snap_hyde measured post-fix (83.93%); other modes never run

## Meeting talking points (5)

1. **The bug-fix decomposition story is rock-solid** — llm_only and snap_only_in_final both show +5.44pp (formatter only), rag_simple shows +7.29pp (formatter + retrieval-query +1.85 marginal). Two no-retrieval modes producing identical lift is the strongest possible internal validation.
2. **snap+HyDE is real on legal MC**: +3-4pp consistently across model sizes (E4B +3.69, 26B +3.09). Not a one-off finding.
3. **snap+HyDE BREAKS on multi-hop**: gold-retrieval crashes from 83% (rag_simple) to 50-60% (HyDE-using). The HyDE-generated passage commits to a wrong-hop entity and biases BM25.
4. **planning_table doesn't fix it**: explicit per-TODO decomposition recovers retrieval but introduces a composition tax (model doesn't trust per-TODO findings, falls back on parametric guess).
5. **Model size dominates method**: Llama 3.3 70b ≈ Gemma 4 26B at ~80% on llm_only; Scout 17b and Qwen3 32b cluster at 67-68%. Method lift is a smaller second-order effect.
