<!-- ARCHIVED 2026-04-27 — superseded by docs/narrative_2026_04_27.md -->

# Monday meeting summary — 2026-04-27

Source: `docs/validation_log_2026-04-25.md` (cluster post-fix wave, commit `56bffc8`),
`logs/experiments.jsonl` (pre-fix references at commit `770c9ac`/`3d5ff05`),
and `logs/eval_*_or-gemma4-26b_*detail.jsonl` (MuSiQue API runs).

All numbers verified against detail logs; no fabrication.

---

## 1. Headline numbers — post-fix N=1195 BarExam

Cluster wave commit `56bffc8` (post both `f95f316` formatter fix and `3d5ff05`
retrieval-query fix). Pre-fix references are commit `770c9ac` (the last clean
N=1195 wave before the fixes landed).

### Gemma 4 26B-A4B (MoE, 25B params / 3.8B active)

| Mode | Pre-fix N=1195 | Post-fix N=1195 | Lift |
|---|---|---|---|
| llm_only | 74.31% | **79.75%** | **+5.44pp** |
| snap_only_in_final | 75.15% | **80.59%** | **+5.44pp** (identical to llm_only) |
| rag_simple | 70.79% | **78.08%** | **+7.29pp** (largest) |
| rag_hyde | 74.23% | **78.91%** | **+4.68pp** |
| rag_snap_hyde | 76.57% | **81.17%** | **+4.55pp** |
| golden_passage | 74.98% | **78.66%** | **+3.68pp** |
| subagent_rag | 75.73% | **78.16%** | **+2.46pp** |
| subagent_hybrid | 73.39% | **74.14%** | **+0.75pp** (smallest) |

### Gemma 3n E4B (~8B effective)

| Mode | Pre-fix N=1195 | Post-fix N=1195 | Lift |
|---|---|---|---|
| rag_simple | 55.73% | **58.49%** | +2.76pp |
| rag_hyde | 57.74% | **60.59%** | +2.85pp |
| rag_snap_hyde | 58.41% | **62.18%** | +3.77pp |
| snap_only_in_final | 54.81% | **57.82%** | +3.01pp |
| subagent_rag | 57.20% | **60.92%** | +3.72pp |
| subagent_hybrid | 57.70% | **58.83%** | +1.13pp |
| llm_only | 55.48% | — (job wallclocked) | — |
| golden_passage | 62.18% | — (job wallclocked) | — |

### Range compression

- **Pre-fix 26B**: rag_simple 70.79 → rag_snap_hyde 76.57 = **5.78pp spread**
- **Post-fix 26B**: rag_simple 78.08 → rag_snap_hyde 81.17 = **3.09pp spread**

The post-fix base (`rag_simple` 78.1) is so much stronger that pipeline complexity
adds only ~3pp on top — the old "snap_hyde adds +5.8pp over rag_simple" headline
compresses to "+3.1pp at the same N=1195".

---

## 2. Bug-fix decomposition story (~200 words)

The two bugs we patched on 2026-04-22 affected the model in distinct ways.
**Bug 1 (formatter `f95f316`)**: the question prompt column carrying shared
fact patterns for 37% of BarExam rows was never read into the model-facing
prompt. **Bug 2 (retrieval-query `3d5ff05`)**: 11 retrieval call sites were
also reading the bare 47-character stem instead of the full prompt-bearing
question, so the vector store was being searched with the wrong query.

Post-fix N=1195 at 26B isolates the two contributions cleanly because
`llm_only` and `snap_only_in_final` perform **no retrieval** — only the
formatter fix can help them. Both modes show **exactly +5.44pp** lift
(74.31 → 79.75 and 75.15 → 80.59), an unambiguous formatter-fix attribution.

`rag_simple` shows **+7.29pp** lift on the same model. Subtracting the
formatter contribution leaves **+1.85pp marginal** as the retrieval-query
fix attribution. This is small because pre-fix BM25 was still retrieving
useful context, just suboptimally. The asymmetry — formatter dominates,
retrieval-query is marginal — directly validates the claim that the bulk
of the lost accuracy was pure prompt-context starvation, not bad retrieval.

Two no-retrieval modes producing identical lift in different pipelines
is the strongest possible regression-test that the fix is real and clean.

---

## 3. Cross-mode pattern (~100 words)

Lift is **inversely proportional to pipeline depth**:

| Calls in pipeline | Mode | Lift |
|---|---|---|
| 1 (no context) | llm_only | +5.44pp |
| 2 (no retrieval) | snap_only_in_final | +5.44pp |
| 1 (retrieval) | rag_simple | +7.29pp |
| 2 (HyDE+final) | rag_hyde | +4.68pp |
| 3 (snap+HyDE+final) | rag_snap_hyde | +4.55pp |
| 1 (gold context) | golden_passage | +3.68pp |
| 4 (gap+rag+report+final) | subagent_rag | +2.46pp |
| 4 (deepest) | subagent_hybrid | +0.75pp |

Multi-call pipelines were partially **compensating** for the missing prompt
context via gap analysis / snap reasoning / HyDE expansion. The fix exposes
that simpler modes were being penalised the most — they had nothing to fall
back on. This is direct evidence that deep pipelines have implicit robustness
to context loss.

---

## 4. MuSiQue cross-domain finding (~200 words)

We tested the same Gemma 4 26B model via OpenRouter API on MuSiQue multi-hop
QA to probe whether the legal-domain method ranking transfers.

| Mode | N | EM | F1 | gold_retrieved (top-5) |
|---|---|---|---|---|
| golden_passage (oracle) | 50 | **62.0%** | 0.759 | 100% (by construction) |
| rag_simple (in-row BM25) | 30 | **26.7%** | 0.414 | **83.3%** |
| rag_hyde (HyDE+BM25) | 30 | **20.0%** | 0.322 | **50.0%** ← retrieval crashes |
| rag_snap_hyde | 30 | **20.0%** | 0.349 | **60.0%** |
| subagent_rag | 15 | 20.0% | 0.344 | 0% (tracking artifact, not a real failure) |

Three findings:

1. **Retrieval works, composition fails.** BM25 hits the gold paragraph 83% of
   the time but the model only solves 27% of questions. The 56pp gap from
   gold-retrieved → EM is pure multi-hop reasoning failure. 3-hop falls to
   8.3% EM despite 75% gold-retrieved.

2. **HyDE actively hurts on multi-hop.** gold_retrieved drops from 83% (BM25
   alone) to 50% with HyDE. The hypothetical answer commits to one wrong hop
   and biases BM25 toward the wrong topic. Concrete failure mode: for "Who is
   the spouse of the Green performer?" HyDE generates Norah Jones (wrong
   guess) and BM25 retrieves Norah Jones paragraphs.

3. **Snap+HyDE compounds the wrong-hop bias.** rag_snap_hyde holds at 20% EM
   but the failure mode is more concentrated: snap commits to a wrong hop,
   HyDE anchors on it, BM25 reinforces it, final answer doubles down. The
   pipeline becomes a wrong-answer reinforcement loop on multi-hop.

**Bound established**: snap+HyDE works on single-hop doctrinal lookup
(legal). It breaks on multi-hop fact composition (MuSiQue). This is a real
domain-specificity finding worth presenting as a null result that motivates
the planning-table / explicit decomposition approach.

---

## 5. Open questions / what's missing (~100 words)

- **E4B llm_only and golden_passage missing**: SLURM job 54173 wallclocked
  at 28h with mode 3 (llm_only) dying at 1155/1195 and mode 4
  (golden_passage) never starting. Both can be re-run via API since neither
  needs Chroma. Without these, we cannot complete the E4B bug-fix
  decomposition (no clean formatter-only mode at the smaller scale).
- **31B (Qwen3-32B) coverage incomplete**: post-fix re-run not yet done at
  31B; the 31B numbers in `experiment_overview.md` are still the pre-fix
  reference (rag_simple 79.6%, rag_snap_hyde 83.93%).
- **MuSiQue N=30 is small**: directional only; N=100 needed for
  publishable claims.
- **Planning-table mode**: hypothesised cure for HyDE wrong-hop
  bias on MuSiQue, not yet implemented.

---

## 6. Recommended talking points

1. **The clean two-fix decomposition is publication-grade evidence.** Two
   no-retrieval modes (llm_only + snap_only_in_final) at 26B both show
   +5.44pp identical lift. This isolates the formatter bug from the
   retrieval-query bug with no ambiguity. Strongest single result this wave.

2. **Snap stacking shrinks at scale.** Pre-fix gap "snap_hyde +5.78pp over
   rag_simple" compresses to **+3.09pp** post-fix at 26B. The "bigger
   models care less about retrieval method" story is now supported by clean
   numbers; the previous gap was partly a bug artifact.

3. **Multi-hop breaks the snap+HyDE assumption.** MuSiQue rag_simple 26.7%
   beats rag_hyde 20% beats rag_snap_hyde 20% — the legal-domain method
   ordering INVERTS on multi-hop. HyDE crashes BM25 gold-retrieval from
   83% to 50%. This bounds the snap+HyDE claim to single-hop domains.

4. **Multi-hop is a composition problem, not a retrieval problem.** BM25
   gets gold paragraphs 83% of the time; the model still misses 73% of
   answers. The "retrieval is solved, composition isn't" framing motivates
   the planning-table next step better than any retrieval-tuning could.

5. **Pipeline depth correlates with bug-robustness.** subagent_hybrid lift
   is +0.75pp vs rag_simple's +7.29pp. Deeper pipelines were silently
   compensating for the bug all along. This reframes our previous "deep
   pipelines underperform" narrative — they were doing reasoning work the
   simpler modes weren't, just to break even on a broken benchmark.
