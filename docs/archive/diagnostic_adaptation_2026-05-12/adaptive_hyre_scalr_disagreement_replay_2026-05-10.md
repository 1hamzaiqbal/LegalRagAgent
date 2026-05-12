# LegalBench-SCALR Disagreement Replay

Date: 2026-05-10

Purpose: test a cached, disagreement-only adaptive layer over existing N=200
SCALR detail logs. The method keeps the shared answer when completed methods
agree and spends one extra LLM call only when their predictions disagree.

## Setup

- Script: `scripts/replay_disagreement_arbitrator.py`
- Dataset: `legalbench_scalr`
- Provider: `or-gemma4-26b`
- Source methods:
  - `rag_simple`
  - `rag_snap_hyde_2call`
  - `adaptive_snap_hyre_frontier`
- Full detail log:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_disagreement_arbitrator_or-gemma4-26b_20260510_scalr_n200_detail.jsonl`

## Result

The replay completed without API or parse errors.

| Method | Correct | Accuracy | Avg calls |
|---|---:|---:|---:|
| `rag_simple` | 148 / 200 | 74.0% | 1.00 |
| `rag_snap_hyde_2call` | 152 / 200 | 76.0% | 2.00 |
| `adaptive_snap_hyre_frontier` | 153 / 200 | 76.5% | 2.00 |
| `adaptive_snap_hyre_disagreement_replay` | 155 / 200 | 77.5% | 0.19 arbitration calls |

Paired comparisons against the replay:

| Baseline | Delta | b/c | p |
|---|---:|---:|---:|
| `rag_simple` | +3.5pp | 13 / 6 | 0.1671 |
| `rag_snap_hyde_2call` | +1.5pp | 8 / 5 | 0.5811 |
| `adaptive_snap_hyre_frontier` | +1.0pp | 6 / 4 | 0.7539 |

On the 38 rows that required arbitration, the replay answered 22/38 correctly.
The source methods on the same slice were:

| Method | Arbitration-slice correct |
|---|---:|
| `rag_simple` | 15 / 38 |
| `rag_snap_hyde_2call` | 19 / 38 |
| `adaptive_snap_hyre_frontier` | 20 / 38 |

## Interpretation

This is a small but real directional win for the adaptive framing on SCALR:
the useful extra budget is spent only when methods disagree, and the replay
beats the strongest source method on that disagreement slice.

The effect is not statistically reliable at N=200. Treat it as a candidate
controller direction, not a final claim. It does, however, justify a more
careful disagreement arbitrator prompt or a cheaper deterministic proxy for
when to trust raw RAG, two-call Snap-HyDE, or the frontier route.
