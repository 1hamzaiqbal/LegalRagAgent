# Validation log — coverage wave starting 2026-04-25

Live audit log for the E4B+26B coverage wave (jobs 54173–54179).
Each landed mode gets a row in the table below + sample inspection notes.
Updated continuously by the babysit loop.

## Wave summary

- 7 SLURM jobs submitted at ~2026-04-25 21:00 UTC (Sat afternoon local)
- All on `general-gpu` partition, 1 GPU each
- Cluster repo at commit `adee3ae`
- Tests at submission: 17/17 pass (test_sanitizer + test_formatter)

| Job | Modes | Status | First detail-log audit |
|---|---|---|---|
| 54173 E4B-1 | rag_simple, rag_hyde, llm_only, golden_passage | RUNNING (mbe_24, 1.5q/min) | — |
| 54174 E4B-2 | rag_snap_hyde, snap_only_in_final | RUNNING (mbe_12, ~0.7q/min) | — |
| 54175 E4B-3 | subagent_rag, subagent_hyde | RUNNING (mbe_13, ~0.7q/min) | — |
| 54176 E4B-4 | subagent_hybrid, snap_hyde_report | RUNNING (mbe_13, ~0.7q/min) | — |
| 54177 26B-1 | rag_simple, rag_hyde, llm_only, golden_passage | RUNNING (mbe_75, ~4q/min) | — |
| 54178 26B-2 | rag_snap_hyde, snap_only_in_final | RUNNING (mbe_30, ~1.8q/min) | — |
| 54179 26B-3 | subagent_rag, subagent_hybrid | RUNNING (mbe_30, ~1.8q/min) | — |

**21:14 UTC**: jobs submitted, all PENDING(Priority) blocked by my own RL queue
**21:30 UTC**: cancelled 14 RL/autowatch jobs — gemma4 jobs leapfrog into a40-2205 (4×) + a100s-2305 (3×)
**21:32 UTC**: all 7 jobs RUNNING; vLLM startup
**21:39 UTC**: vLLM ready on all 7; eval calls in flight, no errors

ETA for first detail log: 26B-1 rag_simple ~02:14 UTC (5h from job start).

## Pre-wave evidence (audit reference)

| Log | Mode | N | Structured leak in HyDE |
|---|---|---|---|
| 2026-04-13 (meeting-flagged) | rag_snap_hyde | 1195 | **74.4%** ⚠️ |
| 2026-04-22 post-fix | rag_snap_hyde | 1195 | 0.00% ✓ |
| 2026-04-22 post-fix | subagent_hyde | 1195 | 0.00% ✓ |
| 2026-04-22 post-fix | snap_hyde_report | 1195 | 0.00% ✓ |

The fix landed cleanly. New runs are expected to maintain ~0% structured leak.
The babysit loop will record each landed mode's leak rate here as it lands.

## Per-mode audits (filled as modes land)

(empty — modes will populate this section as they finish)

## Anomalies / things to investigate

(empty — populated when audits flag something)

## Phase 2 — multi-hop benchmark survey (landed 2026-04-25)

Background subagent compared HotpotQA, MuSiQue, 2WikiMultihopQA. Spec cards
in `docs/multihop_benchmark_survey_2026-04-25.md`. **Recommendation:
lead with MuSiQue** — cleanest schema, hardest narrative, decomposition
field doubles as `decompose_rag` diagnostic. HotpotQA second for breadth.
