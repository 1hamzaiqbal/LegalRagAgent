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
**22:01 UTC** (47 min in): all 7 healthy, 0 errors. **26B running 5-9pp HIGHER pass-rate than pre-fix at the early-question subset** — bug-fix lift visible in real time. Sample pass rates so far:
  - E4B rag_simple 73/131 = 55.7% (matches pre-fix 55.7%)
  - E4B rag_snap_hyde 39/63 = 61.9% (vs pre-fix 58.4%, +3.5pp)
  - 26B rag_simple 310/389 = **79.7%** (vs pre-fix 70.8%, **+8.9pp**)
  - 26B rag_snap_hyde 148/185 = **80.0%** (vs pre-fix 76.6%, +3.4pp)
  - 26B subagent_rag 139/179 = 77.7% (vs pre-fix 75.7%, +2.0pp)

Caveat: early questions only; final accuracies will narrow (the first 200-400q are not a random sample). But large enough early signal to suspect the bug impact was UNDERSTATED at N=200 post-fix smoke (E4B was within noise) — the prompt-fix lift at 26B may be ~5-8pp at full N. Will land definitively when modes complete.

**22:04 UTC**: submitted MuSiQue embed job (54190) — building `musique_passages` collection on a40-2206 (idle), ~10 min ETA.
**22:48 UTC**: 54190 silently FAILED — `Unknown corpus: musique`. Root cause: slurm_embed_musique.sh runs from `/engrfs/.../LegalRagAgent` (data repo) while my fast_embed.py update was pushed to `/engrfs/.../LegalRagAgent-clean` (code repo). Manually scp'd fixed fast_embed.py + CSVs across, resubmitted as **54218** (PENDING — a40-2206 now in use by boundclip).
**22:48 UTC progress**: 26B-1 (rag_simple) at **839/1195 (70.2%) done**, 78.3% pass rate (657/839). Final lands in ~25 min at ~23:25 UTC.

ETA for first detail log: **26B-1 rag_simple ~23:25 UTC**.

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

### 🎯 26B rag_simple post-fix (54177 mode 1) — landed 2026-04-26 01:20 UTC

**Headline: 78.08% (933/1195)** vs pre-fix 70.79% = **+7.29pp bug-fix lift**

| Metric | Value |
|---|---|
| Records | 1195 |
| Final accuracy | 0.7808 (933/1195) |
| Pre-fix reference | 0.7079 (846/1195) at commit 770c9ac |
| Bug-fix lift | **+7.29pp** |
| Avg latency | 5.9s/q |
| Avg LLM calls | 1.0 |
| HyDE artifact leaks (top-level) | **0** |
| Report artifact leaks | **0** |
| Knowledge artifact leaks | **0** |
| Call traces present | 1195/1195 |
| Code commit | 56bffc8 |

By subject (post-fix):
- nan (prompt-bearing 601 Qs): 80.4% (483/601) ← biggest lift target, the 37% of Qs that needed the prompt fix
- CONST. LAW: 90.5% (86/95)
- TORTS: 77.7% (87/112)
- REAL PROP.: 75.0% (69/92)
- CONTRACTS: 72.6% (82/113)
- EVIDENCE: 72.0% (67/93)
- CRIM. LAW: 66.3% (59/89)

Sample inspection (3 random records, idx mbe_1179/700/1034): all clean — full
question/answer text, structured legal analysis in final answer, correct
predictions, no leakage patterns observable. Question field is [:500]-truncated
in log but model received the full text via `_fmt(row, config)`.

**Verdict:** CLEAN. The +7.29pp lift is real bug-fix signal, exactly the
pattern predicted by the asymmetric-impact hypothesis (retrieval modes most
affected). The post-fix accuracy lands above golden_passage pre-fix (75.0%)
for rag_simple — strongly suggests our pre-fix 26B numbers underestimated
the true model capability.

## Anomalies / things to investigate

(empty — populated when audits flag something)

## Phase 2 — multi-hop benchmark survey (landed 2026-04-25)

Background subagent compared HotpotQA, MuSiQue, 2WikiMultihopQA. Spec cards
in `docs/multihop_benchmark_survey_2026-04-25.md`. **Recommendation:
lead with MuSiQue** — cleanest schema, hardest narrative, decomposition
field doubles as `decompose_rag` diagnostic. HotpotQA second for breadth.
