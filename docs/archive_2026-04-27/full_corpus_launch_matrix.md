# ARCHIVED 2026-04-27 — superseded by docs/signoff_log.md
# Full Corpus Launch Matrix (snapshot 2026-04-27 ~03:30 CDT)

This snapshot plans the next full-corpus wave around methods that already survived paired checks, log-quality review, and the current empty-retrieval / preflight guardrails in HEAD (`c8bcd05`). MuSiQue full validation means 2400 questions; BarExam full validation means 1195 questions.

## Sample-size citation tiers (paper-grade calibration, 2026-04-27)

User calibration: N<200 numbers cannot be trusted as legitimate performance numbers.
Concrete demonstration in this batch: Gemma 27B `rag_simple` jumped from 22% (N=100) to 28.5% (N=200) — a 6.5pp swing from sample variance alone.

- **Tier 0 (N=30)**: smoke / triage / direction signal only. NEVER cite.
- **Tier 1 (N=100)**: paired McNemar OK for "trending" language but treat as preliminary.
- **Tier 2 (N=200+)**: minimum citeable for paper.
- **Tier 3 (full corpus, 1195 BarExam / 2400 MuSiQue)**: paper headline.

What we have at Tier 2+ right now (citeable):
- BarExam `rag_snap_hyde` cross-size: Gemma 4 26B-A4B +3.09pp (N=1195) and E4B +3.69pp (N=1195).
- (Everything MuSiQue is currently Tier 1 or below; mhd Gemma 27B N=200 is the first Tier 2 attempt.)

## Robustness gates: must pass BEFORE launching full-corpus

For each method x model: passes if (a) >=2 cross-family validations at N>=200, (b) clean log audit, (c) silent-empty bug fixes in HEAD.

Currently passing:
- BarExam `rag_snap_hyde` (N=1195 Gemma 4 26B-A4B + N=1195 Gemma 4 E4B).
- `mhd` x {Llama 70b, Gemma 27B} on MuSiQue is N=100 only — needs N=200 confirmation before Tier 3 launch (Gemma N=200 in flight).

Baseline findings:
- `multi_hyde_diverse` (`mhd`) is the proven robust method on MuSiQue multi-hop:
  - Llama 3.3 70b dense: +12pp significant at N=100 (33/100 vs 21/100, McNemar p=0.0227/0.023).
  - Gemma 3 27B dense: +8pp trending at N=100 (30/100 vs 22/100, McNemar p=0.1338/0.134); N=200 power-up is running.
  - Llama 4 Scout 17b MoE: -1pp flat at N=100 (29/100 `mhd` vs 30/100 `rag_simple`), a capacity-floor signal rather than a launch candidate.
  - Qwen3 30B MoE: TBD; N=100 pair is still running.
- `rag_snap_hyde` is the proven robust method on BarExam:
  - Gemma 4 26B-A4B: +3.09pp at N=1195 (`rag_snap_hyde` 81.17% vs `rag_simple` 78.08%).
  - Gemma 4 E4B: +3.69pp at N=1195 (`rag_snap_hyde` 62.18% vs `rag_simple` 58.49%).
- Cross-domain: `rag_snap_hyde` x Llama 70b MuSiQue N=100 = 21.0%, identical to `rag_simple` 21.0%. `rag_snap_hyde` is BarExam-specific; `mhd` is multi-hop-specific.

Source notes:
- Verified directly from the last 10 `logs/experiments.jsonl` lines: Scout N=100 flatline (30/100 `rag_simple`, 29/100 `mhd`); Llama 70b MuSiQue `rag_snap_hyde` 21/100; Gemma 27B N=200 `rag_simple` half landed at 57/200 while the `mhd` half is still in flight.
- Derived from `docs/audit_log.md` and `docs/validation_log_2026-04-25.md`: Llama 70b `mhd` +12pp p=0.023; Gemma 27B `mhd` +8pp p=0.134; BarExam `rag_snap_hyde` +3.09pp / +3.69pp; Qwen3 30B MoE BarExam 70% clean baseline.
- `docs/run_audit_2026-04-27.md` marks the relevant N=100 MuSiQue rows clean, and `docs/log_quality_audit_2026-04-27.md` found no truncation / empty-output corruption in sampled latest logs, with only interpretive groundedness caveats.

## Tier 2 (N=200) follow-ups: queue these BEFORE jumping to Tier 3 full-corpus

| Mode | Model | Provider | Dataset | Trigger |
|---|---|---|---|---|
| `mhd` + `rag_simple` pair | Llama 70b | `groq-llama70b` | MuSiQue | After Groq daily TPD reset (19:00 CDT). Convert N=100 +12pp to N=200 paired McNemar. |
| `mhd` + `rag_simple` pair | Qwen3 30B MoE | `or-qwen3-30b-moe` | MuSiQue | After current N=100 pair lands (~50 min). Confirms or refutes capacity-floor on Qwen MoE. |
| `mhd` + `rag_simple` pair | Llama 4 Scout | `groq-scout` | MuSiQue | If we want to confirm flat finding — low priority since direction is clear. |
| `rag_multi_query` + `rag_simple` pair | Llama 70b | `groq-llama70b` | MuSiQue | After current N=100 ablation lands (~10 min). Confirms diversity-vs-HyDE mechanism. |

Tier 2 is the gate to Tier 3. Don't launch Tier 3 corpus jobs on a method/model pair until paired Tier 2 N=200 lands clean.

## Tier 3 — full-corpus runs to fire when Tier 2 N=200 confirms

| Order | Mode | Model | Provider | Dataset | N | ETA | Where | Why |
|---:|---|---|---|---|---:|---|---|---|
| 1 | `mhd` | Llama 70b | `groq-llama70b` | MuSiQue | 2400 (full validation) | ~6-8 hr | local | Convert N=100 +12pp finding to full corpus |
| 2 | `mhd` | Gemma 4 26B-A4B | `or-gemma4-26b` | MuSiQue | 2400 | ~6-8 hr | cluster | Cluster headline model on multi-hop, never tested at scale |
| 3 | `rag_simple` | Llama 70b | `groq-llama70b` | MuSiQue | 2400 | ~3 hr | local | Paired baseline for #1 |
| 4 | `rag_simple` | Gemma 4 26B-A4B | `or-gemma4-26b` | MuSiQue | 2400 | ~3 hr | cluster | Paired baseline for #2 |
| 5 | `rag_snap_hyde` | Llama 3.3 70b | `cluster-vllm` | BarExam | 1195 | ~3 hr | cluster | Extends BarExam winner to non-Gemma cluster model |

## Tier 2 — secondary full-corpus runs (after Tier 1 lands)

| Order | Mode | Model | Provider | Dataset | N | Trigger | Where | Why |
|---:|---|---|---|---|---:|---|---|---|
| 6 | `mhd` | Gemma 3 27B dense | `or-gemma27b` | MuSiQue | 2400 | Launch only if the N=200 power-up converts the current +8pp trend into a clean/significant result | local/API | N=100 is promising but not yet significant; full corpus should wait for the in-flight power check |
| 7 | `mhd` | Qwen3 30B MoE | `or-qwen3-30b-moe` | MuSiQue | 2400 | Launch only if the current N=100 pair lands positive and clean | local/API | Qwen is the MoE architecture check; current status is TBD/running, so no full-corpus launch before audit |
| 8 | `rag_snap_hyde` | Gemma 4 26B-A4B | `cluster-vllm` | BarExam | 1195 | Re-run only if launch branch, extractor, or prompt changes after the clean 81.17% audit | cluster | This is already the BarExam headline; re-validation is useful only to refresh provenance, not to discover the effect |

Do not launch Scout full-corpus `mhd` from current evidence: N=100 is flat (-1pp vs `rag_simple`) and the log-quality audit adds groundedness caveats for Scout raw answers.

## Rate-limit budget

- `groq-llama70b`: 1000 RPD / 100K TPD daily (resets 19:00 CDT)
- `groq-scout`: 1000 RPD / 500K TPD
- `or-*` (per-model): generous, low risk
- cluster: 1 SLURM job at a time per priority queue

## Failure modes to monitor in adaptive loop

- Empty retrieval (gate fires `SystemExit(4)`)
- Truncation at 2048-token cap (Qwen 32b had 13/100 truncated)
- API auth fail (deepseek hit this; preflight gate caught it)
- Out-of-memory (cluster vLLM 26B-A4B needs A100 80GB, NOT a40 44GB; see SLURM 55018 incident)

## When to relax adaptive loop

Once Tier 1 jobs are submitted and running, the loop slows to ~hourly spot-checks rather than 30-min cadence.

## Cluster status

Attempted `ssh wustl 'squeue -u hiqbal'` for live queue status. The sandbox could not use the configured SSH path (`Control socket ... Operation not permitted`; hostname resolution failed), so cluster availability is unknown in this snapshot and no queue state is reported.

## Currently running (peeked at /tmp/*.log)

- `/tmp/mhd_pair_gemma_n200.log`: Gemma 27B N=200 power-up is in flight. Last `experiments.jsonl` line shows paired `rag_simple` landed at 57/200 = 28.5%; temp log shows the `mhd` half had reached 35/200 with no summary yet.
- `/tmp/mhd_pair_qwen.log`: Qwen3 30B MoE N=100 pair is in flight; tail shows `rag_simple` progress through 60/100, with no completed N=100 headline yet.
- `/tmp/iter_hyde_qwen_n30.log` and `/tmp/iter_hyde_scout_n30.log`: secondary diagnostics are live/incomplete; not Tier 1 launch blockers.
- `/tmp/mhd_pair_deepseek.log`: deepseek auth failed in preflight and aborted before logging garbage. This confirms the auth gate behavior and should not be cited as a result.
- `/tmp/snap_hyde_llama_musique.log`: completed `rag_snap_hyde` Llama 70b MuSiQue N=100 at 21/100, matching `rag_simple`; this supports the cross-domain "does not carry to multi-hop" conclusion.
- `/tmp/mhd_pair_scout.log`: completed Scout pair at 30/100 `rag_simple` vs 29/100 `mhd`; no Scout full-corpus launch.
