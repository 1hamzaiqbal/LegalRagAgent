# Audit log — every cited number → its verification

Independent verification trail for every headline number cited in the meeting docs. Each row links to: (a) the detail log on disk, (b) the audit subagent verdict (if any), (c) any caveat.

## Cluster post-fix N=1195 BarExam (commit 56bffc8)

Re-scored 2026-04-26 with current extractor (commit ed15eb7 — added "last standalone A-D letter" fallback). All 12 cluster modes verified ROBUST: 11 of 12 unchanged at the pp level, 1 of 12 (`subagent_hybrid` 26B) shifted +0.08pp = +1 question recovered. **NO HEADLINE NUMBERS CHANGE meaningfully.**

| Mode (cluster) | Stored | Re-scored | Δ | Audit |
|---|---|---|---|---|
| 26B golden_passage | 78.66% | 78.66% | 0.00pp | `docs/post_fixes_audit_2026-04-26.md` (CLEAN) |
| 26B llm_only | 79.75% | 79.75% | 0.00pp | CLEAN |
| 26B rag_hyde | 78.91% | 78.91% | 0.00pp | CLEAN |
| 26B rag_simple | 78.08% | 78.08% | 0.00pp | CLEAN |
| 26B rag_snap_hyde | 81.17% | 81.17% | 0.00pp | CLEAN |
| 26B snap_only_in_final | 80.59% | 80.59% | 0.00pp | CLEAN |
| 26B subagent_hybrid | 74.14% | **74.23%** | **+0.08pp** | +1 record recovered (extractor fallback) |
| 26B subagent_rag | 78.16% | 78.16% | 0.00pp | CLEAN |
| E4B rag_simple | 58.49% | 58.49% | 0.00pp | CLEAN |
| E4B rag_hyde | 60.59% | 60.59% | 0.00pp | CLEAN |
| E4B rag_snap_hyde | 62.18% | 62.18% | 0.00pp | CLEAN |
| E4B snap_hyde_report | 60.75% | 60.75% | 0.00pp | CLEAN |
| E4B snap_only_in_final | 57.82% | 57.82% | 0.00pp | CLEAN |
| E4B subagent_hybrid | 58.83% | 58.83% | 0.00pp | CLEAN |
| E4B subagent_hyde | 60.17% | 60.17% | 0.00pp | CLEAN |
| E4B subagent_rag | 60.92% | 60.92% | 0.00pp | CLEAN |

Plus `_run_gap` silent-fallback rate verified: cluster subagent_rag/hybrid/hyde all 0-0.7% → headline numbers NOT contaminated by silent llm_only fallback (see `docs/silent_fallback_audit_2026-04-26.md` Finding 1.2).

## Cross-family BarExam llm_only N=100 (API)

| Model | EM | Audit verdict |
|---|---|---|
| Llama 3.3 70b | **81%** | CLEAN — 0 errors, 0 None preds |
| Gemma 4 26B-A4B | 79.75% (cluster N=1195) | per cluster table above |
| Qwen3 30B MoE | 70% (N=100) | CLEAN, audit subagent verdict (see commit `86ec6f7`) |
| Qwen3 32b dense | 68% | CLEAN AFTER think-tag strip (was 1/5 broken pre-fix) |
| Gemma 3 27b | 68% | CLEAN |
| Llama 4 Scout 17b | 67% | CLEAN |

## MuSiQue API runs N=30 (Gemma 4 26B + Llama 70b)

All re-scored 2026-04-26 with current extractor. Llama 70b runs that PRE-DATE the `<span>` extractor fix (commit 97c204a) had stored em=6.7% but rescore to 20.0% — uniformly applied across all comparisons.

| Mode | Gemma 4 26B | Llama 70b | gold_ret | Audit |
|---|---|---|---|---|
| rag_simple | 26.7% | 20.0% | 83% / 87% | CLEAN |
| rag_multi_query | 23.3% | 20.0% | 87% / 83% | CLEAN |
| ptable_no_snap_v2 | 23.3% | 20.0% | 83% / 90% | `docs/planning_table_audit_2026-04-26.md` CLEAN |
| ptable_with_snap_v2 | 16.7% | n/a | 93% / n/a | CLEAN — snap-bias mechanism per-record (`docs/post_fixes_audit_2026-04-26.md`); -6.6pp directional, p=0.73 at N=30 |
| iter_ptable | 20.0% | **23.3%** | 77% / 93% | Llama: directional +3.3pp, McNemar p=1.0 (not sig); Gemma: -6.7pp |
| advisor_ptable | (running) | **23.3%** | 83% (Llama) | Llama: +3.3pp same as iter; Gemma in flight |
| rag_snap_hyde | 20.0% | 13.3% | 60% / 50% | CLEAN |
| golden_passage (oracle) | 62% | n/a | n/a | CLEAN |

## Confounder fixes applied

1. **Extractor fix** (commit `97c204a`): MuSiQue extractor strips wrapping HTML tags (`<span>...</span>`). Caught Llama 70b emitting literal HTML wrappers from prompt placeholder.
2. **Extractor fallback** (commit `ed15eb7`): `extract_answer_mc` now falls back to "last standalone A-D letter" when no `Answer:` marker. +1 record on subagent_hybrid 26B.
3. **`_run_gap` routed_to marker** (commit `ed15eb7`): silent fallback to llm_only/snap_only now emits `routed_to` field. Verified existing cluster runs are 0-0.7% fallback rate.
4. **Pre-flight smoke gate** (commit `171c2c4`): aborts on auth/404 in seconds.
5. **Per-question circuit breaker** (commit `171c2c4`): 5 consecutive errors → SystemExit.
6. **Summary-write guard** (commit `171c2c4`): error_rate > 50% → tag as `_FAILED-do-not-use`.
7. **Think-tag strip** (commit `171c2c4`): strips closed `<think>...</think>` before answer extraction. Unlocked Qwen3 (was 1/5 broken pre-fix).

## Confounders that DON'T affect us

- **Finding 2.2 (multi-gold gold_retrieved on MuSiQue)**: only affects modes that bypass `_retrieve_musique_in_row` helper. None of our cited MuSiQue runs do; gold_retrieved figures are correct.
- **Finding 1.2 (silent llm_only fallback)**: cluster subagent_* runs verified 0-0.7%.
- **Finding 3.2 (extract_answer_mc fallback)**: re-score confirms only +0.08pp impact on 1 mode.

## Confounders deferred (LOW/MEDIUM, no current impact)

- **HyDE sanitizer fallback to question text** (Finding 1.1): no current impact; counter not yet added.
- **Open-ended same-model judge bias** (Finding 2.4): only affects legal_rag/australian datasets; not in our current headline.
- **SEED only affects question sampling** (Finding 5.3): noted in CLAUDE.md.
- **Summary-guard threshold 50% too lax** (Finding 7.3): no row currently between 30-49% errors.

## Audit pattern going forward

For every NEW result that lands, the workflow is:
1. **Re-score** with current extractor (handles `<span>`, fallback letter, etc.)
2. **Spawn audit subagent** with concrete checklist (errors, None preds, sample records, comparison vs baseline, statistical significance if claiming a lift)
3. **Save audit verdict** to a doc
4. **Cite from validation_log** with link to audit
5. **Commit + push** with the audit reference in the commit message

This pattern was applied to: ptable_no_snap_v2 (audit `a5d0f6457732180b9`), ptable_with_snap_v2 (audit `ad761cf8dc0f64234`), iter_ptable Llama (audit `aa74ab56dcabb1d29`), advisor Llama (audit `a52e7244f936cc5c8` running), Qwen3 30B MoE (audit `a82dc3b08ebe4203a`).

## MuSiQue N=100 paired anchor — advisor pattern vs rag_simple (Llama 70b)

Audited 2026-04-26 for the "advisor pattern (cheap LLM plans/finds, strong LLM synthesizes)" claim. Both runs at N=100, same shared idx set (100/100 overlap).

| Mode | Stored EM | Re-scored EM | Errors | None preds | gold_retrieved | routed_to | Verdict |
|---|---|---|---|---|---|---|---|
| `advisor_planning_table` (advisor=groq-llama8b, main=groq-llama70b) | 23/100 | **23/100** | 0 | 0 | 88% | 0 | CLEAN |
| `rag_simple` (groq-llama70b) | 21/100 | **21/100** | 0 | 0 | 83% | 0 | CLEAN |

- **Provider integrity (advisor log)**: `advisor_provider='groq-llama8b'` and `provider='groq-llama70b'` on all 100 rows. `advisor_model='llama-3.1-8b-instant'`, todos_count avg ~3, planning_table populated with `{todo, finding, evidence_ids}`. Cheap-plan / strong-synthesize claim holds row-by-row.
- **Spot-check (5 rows from each, seed=42)**: extractor agrees with human judgment. One borderline advisor row (`2hop__51113_84616`) returned `None` because synthesis went meandering — defensible miss, not extractor bug. Rag_simple emits clean `Answer: …` markers; advisor emits longer chain-of-thought, both extracted correctly.
- **Paired McNemar (N=100)**: both_right=12, both_wrong=68, b(adv+/rag-)=11, c(adv-/rag+)=9 → **two-sided exact p = 0.824**. Differential is well within sampling noise.
- **Bootstrap 95% CI on (advisor_em - rag_em)**: +0.02 with **CI [-0.07, +0.11]** (10K resamples, paired). Includes 0.

**Verdict: CLEAN, but NOT statistically significant.** Both numbers are correctly extracted, both runs healthy (0 errors, 0 None preds, 0 silent fallbacks). The +2pp advisor lift is directional only — at N=100 paired, p=0.82 and CI spans zero. Cite as "advisor pattern matches rag_simple within noise at N=100; cheap planner ≠ accuracy regression" rather than as a win. If the paper claims a lift, scale up N or reframe as cost/latency parity.

Files:
- `/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_advisor_planning_table_groq-llama70b_20260426_2229_detail.jsonl`
- `/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_rag_simple_groq-llama70b_20260426_2226_detail.jsonl`
