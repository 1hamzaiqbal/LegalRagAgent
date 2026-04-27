# Compiled Results — paper-grade, audit-referenced

Last updated: 2026-04-27 ~10:40 CDT
Branch: hpc-setup, HEAD: 6b58ddb547d481e1d000b5f1c163174856fec9bd
Source-of-truth for cited numbers: docs/audit_log.md (post-fix authoritative for BarExam Tier 3),
docs/mcnemar_2026-04-27.md (paired tests), logs/experiments.jsonl (raw run summaries).

Notation: `exp row` gives the `logs/experiments.jsonl` run-id timestamp prefix when present. `post-fix re-scored` means the value is verified from the detail log plus `docs/audit_log.md`, but no matching `experiments.jsonl` row exists for that post-fix detail log.

## Section 1 — Tier 3 / Full corpus (cite-able for paper)

### 1.1 BarExam cross-size rag_snap_hyde lift

| Model | rag_simple | rag_snap_hyde | Lift | n | Detail logs | exp row | Audit doc | Commit |
|---|---:|---:|---:|---:|---|---|---|---|
| Gemma 4 26B-A4B | 933/1195 = 78.08% | 970/1195 = 81.17% | +3.09pp | 1195 | `logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl`; `logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md`; `docs/full_corpus_launch_matrix.md` | audit/result commit `8bbf0e7`; extractor audit `ed15eb7` |
| Gemma 4 E4B | 699/1195 = 58.49% | 743/1195 = 62.18% | +3.69pp | 1195 | `logs/eval_rag_simple_cluster-vllm_20260426_0020_detail.jsonl`; `logs/eval_rag_snap_hyde_cluster-vllm_20260426_0614_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md`; `docs/full_corpus_launch_matrix.md` | audit/result commit `8bbf0e7`; extractor audit `ed15eb7` |

Discrepancy to preserve: `logs/experiments.jsonl` has older full-corpus rows for analogous BarExam runs (for example E4B `20260421_0812`/`20260421_1402` and 26B `20260421_1615`/`20260421_2234`) whose percentages differ. Do not silently substitute those rows for the post-fix values above; cite the rows above as `post-fix re-scored`.

### 1.2 Other Tier 3 BarExam method coverage on Gemma 4 26B-A4B

| Mode | EM | n | Detail log | exp row | Audit doc | Commit |
|---|---:|---:|---|---|---|---|
| `golden_passage` | 940/1195 = 78.66% | 1195 | `logs/eval_golden_passage_cluster-vllm_20260426_0224_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `llm_only` | 953/1195 = 79.75% | 1195 | `logs/eval_llm_only_cluster-vllm_20260426_0027_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `rag_hyde` | 943/1195 = 78.91% | 1195 | `logs/eval_rag_hyde_cluster-vllm_20260425_2240_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `rag_simple` | 933/1195 = 78.08% | 1195 | `logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `rag_snap_hyde` | 970/1195 = 81.17% | 1195 | `logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `snap_only_in_final` | 963/1195 = 80.59% | 1195 | `logs/eval_snap_only_in_final_cluster-vllm_20260426_0154_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `subagent_hybrid` | 887/1195 = 74.23% | 1195 | `logs/eval_subagent_hybrid_cluster-vllm_20260426_0254_detail.jsonl` | absent; post-fix re-scored; raw detail stored 886/1195 = 74.14% | `docs/audit_log.md` | audit/result `8bbf0e7`; extractor `ed15eb7` |
| `subagent_rag` | 934/1195 = 78.16% | 1195 | `logs/eval_subagent_rag_cluster-vllm_20260425_2234_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |

### 1.3 Other Tier 3 BarExam method coverage on Gemma 4 E4B

| Mode | EM | n | Detail log | exp row | Audit doc | Commit |
|---|---:|---:|---|---|---|---|
| `rag_simple` | 699/1195 = 58.49% | 1195 | `logs/eval_rag_simple_cluster-vllm_20260426_0020_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `rag_hyde` | 724/1195 = 60.59% | 1195 | `logs/eval_rag_hyde_cluster-vllm_20260426_0714_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `rag_snap_hyde` | 743/1195 = 62.18% | 1195 | `logs/eval_rag_snap_hyde_cluster-vllm_20260426_0614_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `snap_hyde_report` | 726/1195 = 60.75% | 1195 | `logs/eval_snap_hyde_report_cluster-vllm_20260426_1713_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `snap_only_in_final` | 691/1195 = 57.82% | 1195 | `logs/eval_snap_only_in_final_cluster-vllm_20260426_1512_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `subagent_hybrid` | 703/1195 = 58.83% | 1195 | `logs/eval_subagent_hybrid_cluster-vllm_20260426_0545_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `subagent_hyde` | 719/1195 = 60.17% | 1195 | `logs/eval_subagent_hyde_cluster-vllm_20260426_1739_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |
| `subagent_rag` | 728/1195 = 60.92% | 1195 | `logs/eval_subagent_rag_cluster-vllm_20260426_0545_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` |

## Section 2 — Tier 2 / N=200 paired-McNemar (cite-able for paper)

### 2.1 Llama 70b MuSiQue method matrix (TODAY)

| Mode | EM | n | Δ vs rag_simple | McNemar p | Detail log | Commit |
|---|---:|---:|---:|---:|---|---|
| `rag_simple` | 55/200 = 27.5% | 200 | baseline | — | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; exp row `20260427_0952`; doc `docs/mcnemar_2026-04-27.md` | row `31e69db`; result doc `6b58ddb` |
| `multi_hyde_diverse` | 71/200 = 35.5% | 200 | +8.0pp | 0.0195 SIG | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl`; exp row `20260427_1010`; doc `docs/mcnemar_2026-04-27.md` | row `31e69db`; result doc `3ab2f51`/`6b58ddb` |
| `rag_snap_hyde` | 48/200 = 24.0% | 200 | -3.5pp | 0.36 NS | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl`; exp row `20260427_1019`; doc `docs/mcnemar_2026-04-27.md` | row `3ab2f51`; result doc `21e687a` |
| `iter_hyde` | 49/200 = 24.5% | 200 | -3.0pp | 0.47 NS | `logs/eval_iter_hyde_groq-llama70b_20260427_1036_detail.jsonl`; exp row `20260427_1036`; doc `docs/mcnemar_2026-04-27.md` | row `21e687a`; result doc `6b58ddb` |

### 2.2 Gemma 3 27B MuSiQue (cross-family check, NULL'd)

| Mode | EM | n | Δ vs rag_simple | McNemar p | Detail log | Commit |
|---|---:|---:|---:|---:|---|---|
| `rag_simple` | 57/200 = 28.5% | 200 | baseline | — | `logs/eval_rag_simple_or-gemma27b_20260427_0309_detail.jsonl`; exp row `20260427_0309`; docs `docs/mcnemar_2026-04-27.md`, `docs/meeting_2026_04_27_brief_v2.md` | row `c8bcd05`; result doc `83fb2fc` |
| `multi_hyde_diverse` | 62/200 = 31.0% | 200 | +2.5pp | 0.5901 NS | `logs/eval_multi_hyde_diverse_or-gemma27b_20260427_0404_detail.jsonl`; exp row `20260427_0404`; docs `docs/mcnemar_2026-04-27.md`, `docs/meeting_2026_04_27_brief_v2.md` | row `a3aee05`; result doc `83fb2fc` |

### 2.3 Llama 4 Scout MuSiQue baseline confirmations (sub-Tier 2 — kept for context but per user, dropping Scout going forward)

| Mode | EM | n | Interpretation | Detail log | Commit |
|---|---:|---:|---|---|---|
| `rag_simple` | 30/100 = 30.0% | 100 | Tier 1 context; paired with N=100 mhd | `logs/eval_rag_simple_groq-scout_20260427_0246_detail.jsonl`; exp row `20260427_0246`; docs `docs/run_audit_2026-04-27.md`, `docs/log_quality_audit_2026-04-27.md` | row `46fe19b` |
| `multi_hyde_diverse` | 29/100 = 29.0% | 100 | Tier 1 flat vs baseline | `logs/eval_multi_hyde_diverse_groq-scout_20260427_0249_detail.jsonl`; exp row `20260427_0249`; docs `docs/run_audit_2026-04-27.md`, `docs/log_quality_audit_2026-04-27.md` | row `46fe19b` |
| `rag_simple` | 60/200 = 30.0% | 200 | Tier 2 baseline context | `logs/eval_rag_simple_groq-scout_20260427_0459_detail.jsonl`; exp row `20260427_0459`; doc `docs/meeting_2026_04_27_brief_v2.md` | row `6b7a922` |
| `rag_multi_query` | 61/200 = 30.5% | 200 | Multi-query N=100 dip was noise | `logs/eval_rag_multi_query_groq-scout_20260427_0332_detail.jsonl`; exp row `20260427_0332`; doc `docs/meeting_2026_04_27_brief_v2.md` | row `a3aee05` |

## Section 3 — Mechanism decomposition (preliminary, multi-source verified)

| Model | rag_simple N=200 | rag_multi_query N=200 | mhd N=200 | Diversity comp | HyDE comp | Direct refs |
|---|---:|---:|---:|---:|---:|---|
| Llama 70b | 27.5% (`20260427_0952`) | pending at N=200; N=100 was 25.0% (`20260427_0325`) | 35.5% (`20260427_1010`) | N=100 only: +4pp | N=100 only: +8pp beyond multi-query | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; `logs/eval_rag_multi_query_groq-llama70b_20260427_0325_detail.jsonl`; `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl`; docs `docs/mhd_mechanism_2026-04-27.md`, `docs/mcnemar_2026-04-27.md`; commits `31e69db`, `77dd9da`, `3ab2f51`, `6b58ddb` |
| Gemma 3 27B | 28.5% (`20260427_0309`) | 28.5% (`20260427_0536`) | 31.0% (`20260427_0404`) | 0.0pp observed | +2.5pp observed, but McNemar p=0.5901 NULL | `logs/eval_rag_simple_or-gemma27b_20260427_0309_detail.jsonl`; `logs/eval_rag_multi_query_or-gemma27b_20260427_0536_detail.jsonl`; `logs/eval_multi_hyde_diverse_or-gemma27b_20260427_0404_detail.jsonl`; docs `docs/mcnemar_2026-04-27.md`, `docs/meeting_2026_04_27_brief_v2.md`; commits `c8bcd05`, `0d51b36`, `a3aee05`, `83fb2fc` |
| Llama 4 Scout | 30.0% (`20260427_0459`) | 30.5% (`20260427_0332`) | pending at N=200; N=100 was 29.0% (`20260427_0249`) | +0.5pp observed at N=200 | pending | `logs/eval_rag_simple_groq-scout_20260427_0459_detail.jsonl`; `logs/eval_rag_multi_query_groq-scout_20260427_0332_detail.jsonl`; `logs/eval_multi_hyde_diverse_groq-scout_20260427_0249_detail.jsonl`; docs `docs/meeting_2026_04_27_brief_v2.md`, `docs/log_quality_audit_2026-04-27.md`; commits `6b7a922`, `a3aee05`, `46fe19b` |
| Qwen3 30B MoE | N=100 only: 24.0% (`20260427_0334`) | pending | N=100 only: 28.0% (`20260427_0448`) | pending | N=100 +4pp total; not decomposed | `logs/eval_rag_simple_or-qwen3-30b-moe_20260427_0334_detail.jsonl`; `logs/eval_multi_hyde_diverse_or-qwen3-30b-moe_20260427_0448_detail.jsonl`; doc `docs/meeting_2026_04_27_brief_v2.md`; commit `a3aee05` |

Treat this section as mechanism evidence, not a final causal decomposition.

## Section 4 — Friend/foe attribution probe

Brief: 4/30 Gemma + 6/30 Llama outcome changes; reference `docs/friend_foe_bias_analysis_2026-04-27.md`.

| Model | Accuracy | Kept snap self/foe/control | Outcome changes | Detail log | exp row | Commit |
|---|---:|---:|---:|---|---|---|
| Gemma 3 27B | 3/30 = 10.0% | 27/30 / 27/30 / 27/30 | 4/30 = 13.3% | `logs/eval_friend_foe_attribution_or-gemma27b_20260427_0249_detail.jsonl`; doc `docs/friend_foe_bias_analysis_2026-04-27.md` | `20260427_0249` | row `46fe19b`; analysis doc `6b7a922` |
| Llama 70b | 4/30 = 13.3% | 25/30 / 25/30 / 22/30 | 6/30 = 20.0% | `logs/eval_friend_foe_attribution_groq-llama70b_20260427_0305_detail.jsonl`; doc `docs/friend_foe_bias_analysis_2026-04-27.md` | `20260427_0305` | row `393e12f`; analysis doc `6b7a922` |

## Section 5 — Negative findings (citeable as 'method does not transfer')

| Finding | Verified value | Citation status | Direct refs |
|---|---|---|---|
| `iter_hyde` hurts/underperforms small-model MuSiQue settings | Gemma 27B: `iter_hyde` 2/30 = 6.7%; Scout: 5/30 = 16.7%; Qwen3 30B MoE: 2/30 = 6.7% | Direction only; not a definitive result. The requested -13 to -17pp statement uses mixed N=100 baselines for Scout/Qwen and Gemma; same-N Gemma N=30 baseline gives -20.0pp. | Detail logs `logs/eval_iter_hyde_or-gemma27b_20260427_0034_detail.jsonl`, `logs/eval_iter_hyde_groq-scout_20260427_0320_detail.jsonl`, `logs/eval_iter_hyde_or-qwen3-30b-moe_20260427_0347_detail.jsonl`; baseline logs `logs/eval_rag_simple_or-gemma27b_20260426_2355_detail.jsonl`, `logs/eval_rag_simple_groq-scout_20260427_0246_detail.jsonl`, `logs/eval_rag_simple_or-qwen3-30b-moe_20260427_0334_detail.jsonl`; exp rows `20260427_0034`, `20260427_0320`, `20260427_0347`; docs `docs/audit_log.md`, `docs/meeting_2026_04_27_brief_v2.md`; commits `4d06a34`, `c8bcd05`, `a3aee05` |
| `iter_hyde` does not help Llama 70b at Tier 2 | 49/200 = 24.5% vs `rag_simple` 55/200 = 27.5%; -3.0pp; p=0.47 NS | Citeable as neutral/not significant, not as active harm | Detail logs `logs/eval_iter_hyde_groq-llama70b_20260427_1036_detail.jsonl`, `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; exp rows `20260427_1036`, `20260427_0952`; doc `docs/mcnemar_2026-04-27.md`; result commit `6b58ddb` |
| `rag_snap_hyde` does not carry from BarExam to MuSiQue | MuSiQue Llama 70b N=200: 48/200 = 24.0% vs 55/200 = 27.5%; -3.5pp; p=0.36 NS | Citeable as method-specificity evidence | Detail logs `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl`, `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; exp rows `20260427_1019`, `20260427_0952`; doc `docs/mcnemar_2026-04-27.md`; result commit `21e687a` |
| BarExam `rag_snap_hyde` remains positive where it belongs | Gemma 4 26B-A4B +3.09pp; Gemma 4 E4B +3.69pp | Citeable Tier 3, but post-fix re-scored from audit/detail logs, not experiments rows | Detail logs from Section 1.1; docs `docs/audit_log.md`, `docs/full_corpus_launch_matrix.md`; audit/result commit `8bbf0e7` |

## Section 6 — Methodology hardening shipped this week

- Pre-flight gate, circuit breaker, summary guard, think-tag strip: `171c2c4`; documented in `docs/audit_log.md`.
- Empty-retrieval guard / FAILED-EMPTY-RETRIEVAL protection: `5f8b723`; caught `20260426_2242_advisor_planning_table_groq-llama70b_api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL` in `logs/experiments.jsonl`.
- Extractor fallback and `_run_gap` routed_to marker: `ed15eb7`; `docs/audit_log.md` records the +1 subagent_hybrid recovery and silent-fallback audit.
- mhd / iter_hyde silent-empty fallback bug fixes: `393e12f`; implementation risks documented in `docs/method_impl_audit_2026-04-27.md`.
- Tier system / N<200 citation discipline: `800c454`; `docs/full_corpus_launch_matrix.md`.
- McNemar paired-test infrastructure and result docs: `83fb2fc`, `3ab2f51`, `21e687a`, `6b58ddb`; `docs/mcnemar_2026-04-27.md`.

## Section 7 — Currently in flight

| Run | Status | Detail log | Source |
|---|---|---|---|
| `gemma4_full` mhd-pair × Gemma 4 26B-A4B × N=2400 MuSiQue | In flight; `/tmp/mhd_pair_gemma4_full.log` tail showed `[324/2400]` (~13.5%) on the current local read, close to the requested ~12% snapshot | pending | `/tmp/mhd_pair_gemma4_full.log`; `docs/mcnemar_2026-04-27.md` lists Gemma 4 26B-A4B full MuSiQue in flight |
| `qwen_full` mhd-pair × Qwen3 30B MoE × N=2400 MuSiQue | In flight; `/tmp/mhd_pair_qwen_full.log` tail showed `[791/2400]` (~33.0%), close to the requested ~30% snapshot | pending | `/tmp/mhd_pair_qwen_full.log`; `docs/mcnemar_2026-04-27.md` lists Qwen3 30B MoE full MuSiQue in flight |
| SLURM BarExam mhd+iter_hyde × Gemma 4 26B-A4B N=200 | Pending/unverified locally. User supplied SLURM `55107`; docs conflict with older `55040` / `55094`, and live `ps` was blocked by sandbox. | pending | `docs/mcnemar_2026-04-27.md`; `docs/meeting_2026_04_27_brief_v2.md`; user-provided current snapshot |
| `subagent_rag` × Llama 70b N=200 | No longer in flight in local sources: `/tmp/captain_llama70b_subagent.log` completed 31/200 = 15.5%, with detail log and dirty `logs/experiments.jsonl` row `20260427_1044`; no paired McNemar/audit doc yet, so do not promote to paper-grade. | `logs/eval_subagent_rag_groq-llama70b_20260427_1044_detail.jsonl` | `/tmp/captain_llama70b_subagent.log`; `logs/experiments.jsonl` row `20260427_1044`; row commit `6b58ddb` |

## Section 8 — What NOT to cite (failed runs / contaminated rows)

- All N=30 runs as `result`; use only as smoke/direction, even when the direction is consistent.
- All N=100 runs as definitive; at most Tier 1 directional unless reinforced by N=200+.
- The `advisor_planning_table` BarExam N=50 FAILED-EMPTY-RETRIEVAL row in `logs/experiments.jsonl`: `20260426_2242_advisor_planning_table_groq-llama70b_api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL`; detail log `logs/eval_advisor_planning_table_groq-llama70b_20260426_2242_detail.jsonl`; doc `docs/audit_log.md`; commit `45f1e03`; empty retrieval 50/50.
- Pre-fix BarExam numbers before formatter/retrieval-query fixes `f95f316` and `3d5ff05`; use post-fix `docs/audit_log.md` values.
- `subagent_rag` Llama 70b N=200 15.5% as paper-grade until a paired comparison/audit doc exists and the dirty `logs/experiments.jsonl` row is committed or otherwise archived.

## Reproducibility appendix

- Current HEAD: `6b58ddb` (`analysis: Llama 70b N=200 Tier 2 method matrix complete`).
- Recent result commits from `git log --oneline -30`: `6b58ddb`, `21e687a`, `3ab2f51`, `83fb2fc`, `6b7a922`, `a3aee05`, `800c454`, `77dd9da`, `393e12f`, `5f8b723`, `8bbf0e7`.
- Historical hardening commits required for BarExam interpretation: `f95f316` (prompt column in BarExam formatting), `3d5ff05` (prompt column in retrieval/rerank query paths), `ed15eb7` (extractor fallback + routed_to marker), `171c2c4` (pre-flight/circuit/summary/think-tag guard).
- Data-state caveat: `logs/experiments.jsonl` is dirty in the current worktree. The latest MuSiQue rows are directly verifiable locally, but the JSONL edits themselves have not all been committed. The post-fix Tier 3 BarExam detail logs are under ignored `logs/` paths and absent from `logs/experiments.jsonl`; the committed audit trail for those values is `docs/audit_log.md`.
