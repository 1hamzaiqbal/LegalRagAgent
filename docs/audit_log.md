# Audit log — every cited number → its verification

## Update 2026-04-27 ~12:30 CDT

Change reason: append today’s Tier 2 McNemar verdicts without rewriting older audit history. Cross-check source IDs in `logs/experiments.jsonl`: Llama rows `20260427_0952`, `20260427_1010`, `20260427_1019`, `20260427_1036`, `20260427_1044`, `20260427_1112`; Gemma rows `20260427_0309`, `20260427_0404`, `20260427_0536`.

| Finding | Verified verdict |
|---|---|
| Llama 70b MuSiQue `multi_hyde_diverse` N=200 | 35.5% vs `rag_simple` 27.5%; +8pp; McNemar p=0.0195 SIG; paper headline |
| Gemma 3 27B MuSiQue `multi_hyde_diverse` N=200 | 31.0% vs 28.5%; +2.5pp; p=0.5901 NULL |
| Llama 70b mechanism control | `rag_multi_query` 29.0%; +1.5pp; p=0.728 NS, so HyDE-style passages explain about +6.5pp of the +8pp MHD lift |
| Llama 70b negatives | `rag_snap_hyde` -3.5pp p=0.36 NS; `iter_hyde` -3.0pp p=0.47 NS; `subagent_rag` real -12.0pp p=0.0007 SIG negative with gap-routing over-abstention caveat |
| Friend/foe attribution probe | real but limited: 10/60 outcome changes; see `docs/friend_foe_bias_analysis_2026-04-27.md` |

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

## Multi-method × multi-dataset landings 2026-04-26 ~22:42 UTC

Two new landings audited together: a MuSiQue golden_passage Llama-vs-Gemma comparison, and a BarExam advisor_planning_table paired against llm_only Llama. Both were re-scored with current extractors, provider/model integrity verified, and per-record sampled.

### Landing 1: MuSiQue Llama 70b golden_passage N=30 vs Gemma 4 26B golden_passage N=50

| Metric | Llama 70b N=30 | Gemma 4 26B N=50 |
|---|---|---|
| Stored / Re-scored EM | 14/30 = 46.67% (rescored matches) | 31/50 = 62.0% (rescored matches) |
| Avg F1 | 0.648 | 0.759 |
| F1≥0.7 lenient rate | 50.0% (15/30) | 64.0% (32/50) |
| Errors / None preds | 0 / 0 | 0 / 0 |
| gold_passage present | 30/30 | 50/50 |
| Provider/Mode/Dataset | groq-llama70b / golden_passage / musique (clean) | or-gemma4-26b / golden_passage / musique (clean) |

Same-idx overlap (30 shared questions): both EM=13, only Llama EM=1, only Gemma EM=4, neither=12. F1≥0.7 lenient on overlap: Llama 50% vs Gemma 60%. The 5 cells where one model wins differ across the two models (`2hop__82858_654855`, `3hop1__491648_339990_15538`, `3hop1__773338_42197_18397`, `3hop2__230_89048_66294`, `4hop1__166471_49925_13759_736921`) — failures aren't on the same questions, so this is a real model-capability difference, not a single hard subset.

Of Llama's 16 EM=False: 7 prefix/article mismatches (e.g., `Rabbi Dovber Schneuri` vs `Dovber Schneuri`, `Koh Phi Phi` vs `island Koh Phi Phi`, `to the Politburo` vs `the Politburo`), 1 different-paraphrase mid-F1, 8 genuinely wrong (`17th century` vs `13 December 1642`, etc.). Gemma's 19 EM=False: 10 prefix/article, 1 paraphrase, 8 genuinely wrong — same shape, more correct ones underneath.

**Verdict: REAL ceiling difference.** Strict-EM normalization quirk explains ~3-7pp on each side equally; Gemma's lead survives F1≥0.7 lenient (60% vs 50% on the 30-overlap, 64% vs 50% overall). The "compositional bottleneck differs by model" claim is supported. Safer cite is "F1≥0.7 lenient: Gemma 64% / Llama 50% on shared idxs" alongside the strict EM figures rather than just 62% vs 46.7% headline. Llama's gap is REAL, not normalization noise.

### Landing 2: BarExam advisor_planning_table Llama 70b N=50 (paired vs llm_only)

| Metric | advisor_planning_table | llm_only |
|---|---|---|
| Stored / Re-scored is_correct | 36/50 = 72.0% (matches) | 42/50 = 84.0% (paired N=50) |
| Errors / None preds / extraction failures | 0 / 0 / 0 | 0 / 0 / 0 |
| Provider | groq-llama70b (synth), groq-llama8b (advisor, llama-3.1-8b-instant) — verified all 50 rows | groq-llama70b |
| **`retrieved_ids`** | **EMPTY on 50/50 rows** | n/a |
| **`gold_retrieved=True`** | **0/50 rows** | n/a |
| **`evidence_store`** | **EMPTY on 50/50 rows** | n/a |
| Per-step `evidence_ids` | EMPTY on 150/150 step entries | n/a |
| Findings text | "No answer available" / "No information is available to answer this question" on 150/150 step findings | n/a |

Cross-check: same advisor harness on MuSiQue (`...2229_detail.jsonl` N=100) shows `gold_retrieved=True` 88/100 and zero empty `retrieved_ids`. The retrieval failure is **BarExam-specific in this run** — `_retrieve_and_format` at line 893 of eval_harness.py routes BarExam through ChromaDB `legal_passages`, but every call returned 0 docs.

McNemar paired (50 overlap): both right 32, b(advisor+/llm-)=4, c(advisor-/llm+)=10, both wrong 4. Continuity-corrected χ²=1.79, p≈0.18. Sampling 3 advisor-wrong/llm-right rows (`mbe_377`, `mbe_900`, `mbe_761`): in all 3 the advisor's planning_table findings were ALL "No answer available", and the strong synth still had to guess from doctrine — drifted to a wrong letter. The 4 advisor-right/llm-wrong rows look like dice rolls, not method wins (same empty findings).

**Verdict: ARTIFACT, not a real method failure.** The -12pp gap is driven by a broken retrieval pipeline (50/50 rows zero passages on legal_passages collection in this run), not by the cheap-LLM 8B systematically misleading the synthesizer. Findings carried zero signal — cheap LLM correctly said "no information available" — and the 70B synth faced the SAME doctrinal-only task as llm_only but with ~150 useless "no answer" lines as distractor context. This degrades the synth's accuracy. **DO NOT CITE the -12pp legal-MC regression.** Re-run `advisor_planning_table` on barexam after diagnosing why `_retrieve_and_format` returned empty on legal_passages here (env var or session-level chroma issue at 22:42 UTC; the 22:29 MuSiQue run on same provider/same harness retrieved fine, so the harness code is OK — likely a transient ChromaDB / collection state issue). Until then the only safe BarExam claim is "advisor pattern not yet measured on BarExam at N=50 with retrieval working".

Files:
- `/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_golden_passage_groq-llama70b_20260426_2239_detail.jsonl`
- `/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_golden_passage_or-gemma4-26b_20260426_0007_detail.jsonl`
- `/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_advisor_planning_table_groq-llama70b_20260426_2242_detail.jsonl`
- `/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_llm_only_groq-llama70b_20260426_1930_detail.jsonl`
- Comparison advisor with retrieval working: `/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_advisor_planning_table_groq-llama70b_20260426_2229_detail.jsonl` (N=100 musique, 88/100 gold_retrieved)

Audit agent: `audit_landings_2026-04-26_2242`

### Root cause + scope (2026-04-26 ~22:50 UTC, post-audit)

**Root cause confirmed**: this Mac's `chroma_db/` has `legal_passages` collection with **0 docs** (verified via `chromadb.PersistentClient(path='./chroma_db').get_collection('legal_passages').count() == 0`). This is NOT a transient issue — the local corpus is missing entirely. All cluster runs are unaffected (cluster has its own `chroma_db/` populated).

**Scope of contamination on local Mac BarExam runs** (audit script run against `logs/eval_*_barexam_detail.jsonl`):
- Confirmed bogus: `eval_advisor_planning_table_groq-llama70b_20260426_2242_detail.jsonl` (50/50 empty_ret, 72.0% — tagged `_FAILED-EMPTY-RETRIEVAL` in experiments.jsonl)
- All other recent local BarExam runs that retrieve: 0.2-4% empty rows = within normal noise (queries that genuinely don't match anything in the corpus); these come from CLUSTER, where the corpus IS populated

**False positive**: `golden_passage_cluster-vllm_20260421_1658_detail.jsonl` flagged 1195/1195 empty_ret — but `golden_passage` mode INJECTS the row's `golden_passage` field directly into the prompt instead of retrieving. By design. Added `golden_passage`/`golden_arbitration`/`golden_arb_conservative` to `_NO_CHROMA_MODES` to suppress this false positive in future audits.

**Harness fixes applied (commit pending)**:
1. **Pre-flight collection check** (`eval/eval_harness.py` ~line 4509): before iterating questions, fetch the configured ChromaDB collection and abort with `SystemExit(4)` if `count() == 0` and the mode requires retrieval. Prints `[preflight] FAILED: collection 'X' is EMPTY (0 docs)`. Skipped for `_NO_CHROMA_MODES` (golden, llm_only, vectorless, etc.) and for `dataset == "musique"` (in-row BM25, no Chroma).
2. **Empty-retrieval summary guard** (`eval/eval_harness.py` ~line 4798): at end-of-run, if a RAG mode produced `retrieved_ids == []` on >50% of rows, tag the summary `_FAILED-EMPTY-RETRIEVAL` with `empty_retrieval_rate` and `empty_retrieval_count` recorded. Mirrors the existing `_FAILED-do-not-use` error-rate guard.
3. **`_NO_CHROMA_MODES` set** (module-level constant): now includes the golden modes plus the historical no-retrieval list, so both pre-flight and post-hoc guards skip them correctly.

**Recommended action**: any local-Mac RAG run before commit (this commit) is suspect. If you need legal_passages locally, rebuild via `uv run python utils/fast_embed.py barexam` (~2.2 hr on RTX 3070, longer on Mac CPU). For the meeting, all BarExam claims must come from cluster runs (they have a populated corpus); local Mac is multi-hop / API-only territory.

**Codex independent verification**: requested 2026-04-26 ~22:55 UTC, agent `ab52ee7515e1ad06d`. Status as of ~01:00 UTC next day: codex processes from that timestamp are no longer in `ps`, no commits, no logs, no branches, no stashes left. **Codex agent failed silently**, no artifacts produced. Internal verification + Haiku review (agents `aa510f848aebbc319`, `abc5c719d7d81d4fe`, `aded790c4e293d756`) covered the same ground: Haiku independently reviewed multi_hyde_diverse code (caught 3 issues, all fixed in commit `5f8b723`); audit subagents verified every cited cluster headline number, advisor N=100 paired comparison, golden_passage Llama, and Phase 13 multi_hyde_diverse cross-model. Pre-flight collection check + empty-retrieval summary guard now prevent the empty-retrieval contamination class harness-wide. Conclusion: we proceeded with internal verification only — no loss of rigor since the same checks were performed by Haiku/audit subagents.

A second codex dispatch for friend/foe attribution mode implementation (agent `ad1bfb685a6e6feb3`) similarly failed silently — no commits, no artifacts. Friend/foe mode design sketch is captured in task #32 / validation_log; will rebuild manually if time permits before the meeting. A third codex dispatch for iter_hyde code review (agent `a3b049be08e681d7d`) at ~00:13 UTC was still active in `ps` as of ~01:00 UTC — may yet deliver.

**Update 2026-04-27 ~01:00 UTC**: ROOT CAUSE of all the silent codex failures was the codex CLI version. CLI 0.124.0 / 0.125.0 default model `gpt-5.5` was rejected by upstream API ("model requires newer Codex"). Companion CLI test directly with `--model spark --effort low` works. Codex broker for LegalRagAgent (PID 70634, 6+ hours stale) restarted; 13 zombie codex tasks (5-14 days old) bulk-cancelled; CLI upgraded to 0.126.0-alpha.4 with `gpt-5.5` working as default. **Codex now usable via codex-rescue agent with no extra flags** — defaults to `gpt-5.5 + xhigh + fast + write-mode + fresh` per `~/.codex/config.toml`. Verified by dispatching iter_hyde audit (agent `a58e57ee8200008e5` / task `task-mogsjzg4-5okahl`): completed in 15m 42s with substantive verdict.

## iter_hyde Gemma 3 27B N=30 — independent CODEX audit (2026-04-27 ~01:15 UTC)

Codex (model gpt-5.5, xhigh, read-only) independently verified the Phase 14 negative finding. Verdict: **REAL_FINDING**. Not a code bug, not a prompt issue, not a Gemma quirk — a genuine method limitation.

Re-score: stored 2/30 = re-scored 2/30 (no extractor disagreements). 30/30 have `Answer:` marker; 28/30 `gold_retrieved=True`. Round distribution: 3 records ran 1 round, 3 ran 2, 24 ran 3 (decider firing 20% — 6/30 early-exits). Failure-mode breakdown on 5 sampled em=False rows:

| Cause | Count |
|---|---|
| (a) chain found right span, synth compressed/lost it | 1 |
| (b) chain confabulated wrong entity, synth followed | 3 |
| (c) chain found right info, synth abstained | 0 |
| (d) chain split across rounds, synth picked wrong thread | 1 |

**The smoking-gun comparison** (codex's most powerful finding): on shared `idx=2hop__622308_61845` (gold "Mido"):
- **iter_hyde** round 2 finding contained "Ray Stewart and Mido"; round 3 HyDE conditioned only on Ray Stewart, narrowing the focus → synth followed → wrong answer "Ray Stewart"
- **multi_hyde_diverse** (same model, same question) pooled retrieval kept BOTH candidates; synth directly compared sources and picked "Mido" — correct

**Mechanism**: iter_hyde's serial-conditioned-on-prior-findings architecture is BOTH a feature (chain coherence, narrowing search) AND a liability (early-round drift narrows the synth's option set; once committed, the chain pulls toward wrong answers). mhd's parallel pooling preserves multiple candidate paths until the synth chooses.

**Implication for paper**: cite iter_hyde Gemma 3 27B -20pp finding as "serial chain drift" mechanism, complementary to mhd's "diversity preserves option set" mechanism. Frame both as the FIRST principled MuSiQue method-comparison (mhd > iter_hyde > rag_simple is the wrong cross-family ordering for Gemma 3 27B; mhd > iter_hyde and mhd > rag_simple is right).

Codex audit task: `task-mogsjzg4-5okahl`. Output saved automatically to codex job log.

## Qwen3-32b cited 68% has 13/100 truncated — caveat needed (2026-04-26 ~23:00 UTC)

Audit of `logs/eval_llm_only_groq-qwen_20260426_1941_detail.jsonl` (Qwen3-32b dense, the "Qwen3 32b dense | 68%" row in the cross-family BarExam llm_only board) found:

- **13/100 records have `predicted_answer=None`** (effectively wrong by default, contributing -13pp to the cited 68%)
- **All 13 had `output_tokens` ≥ 2046** (max ≈ 2049) — they hit the **2048-token max_tokens cap**
- Inspection: each None-pred record starts with `<think>\nOkay, let's try to figure...` and ends mid-sentence in the reasoning, never closing the `</think>` tag and never reaching `Answer: X`. The think-tag-strip helper (commit `171c2c4`) only fires on CLOSED `<think>...</think>` blocks, so unclosed truncated reasoning falls through and `extract_answer_mc` finds no `Answer:` marker.
- Other models in the same board emit ~500-2000 tokens for an MC answer; Qwen3-32b averages 4943 chars (~1500 tokens) on OK records and the 13 truncated records spent the full 2048 tokens reasoning before emitting any conclusion.

**Implication**: Qwen3-32b's "true" llm_only accuracy is likely **70-78%** (depending on how the 13 truncated reason chains would have resolved). Our cited 68% understates the model and is a methodology artifact (output-token cap, not model capability).

**Verdict**: KEEP-WITH-CAVEAT. Update validation_log + experiment_overview rows for "Qwen3 32b dense | 68%" to read "Qwen3 32b dense | 68% (13/100 truncated mid-`<think>` at 2048 tokens; true score likely higher)". Re-running with `max_tokens=4096` is a one-shot fix but eats Groq TPD budget; defer unless Qwen becomes a headline claim.

**Other models in the cross-family board verified CLEAN** (0 None preds): Llama 70b (81%), Gemma 27b (68%), Llama 4 Scout (67%), Qwen3-30B-MoE (70%), GPT-5.4-mini (74%), Gemma 4 26B cluster (79.7%). The truncation issue is Qwen3-dense-specific (verbose `<think>` reasoning).

Files:
- `/Users/hamzaiqbal/grad/LegalRagAgent/logs/eval_llm_only_groq-qwen_20260426_1941_detail.jsonl`
- Cross-check at-cap rate: 13/13 None-preds had output_tokens ≥ 2046

## Phase 12 cluster verification — every cited cluster headline is bulletproof (2026-04-26 ~23:15 UTC)

Independent re-score of every cited cluster BarExam number under the current extractor (handles `<span>` wrap-strip, last-standalone-A-D fallback, think-tag strip). Every cluster claim in CLAUDE.md / RESEARCH.md / experiment_overview.md / validation_log_2026-04-25.md verified directly from disk:

### Cross-size `rag_simple` BarExam scaling (N=1195 each, all clean)

| Run | Stored = Re-scored | Cited model | Errors | None |
|---|---|---|---|---|
| `rag_simple_cluster-vllm_20260421_0802` | 542/1195 = **45.4%** | E2B | 0 | 0 |
| `rag_simple_cluster-vllm_20260421_0812` | 666/1195 = **55.7%** | E4B | 0 | 0 |
| `rag_simple_cluster-vllm_20260421_0857` | 846/1195 = **70.8%** | 26B-A4B | 0 | 0 |
| `rag_simple_cluster-vllm_20260421_1203` | 951/1195 = **79.6%** | 31B | 0 | 0 |

Monotonic scaling story: every step of the cross-size cited table re-scores identically.

### Gemma 4 26B-A4B Phase 12 modes (cluster, N=1195 each)

| Mode | Stored | Re-scored | Cited | Notes |
|---|---|---|---|---|
| `rag_snap_hyde` | 970/1195 = 81.17% | **81.17%** | 81.17% | ✓ headline winner |
| `snap_only_in_final` | 963/1195 = 80.59% | **80.59%** | 80.59% | ✓ |
| `subagent_rag` | 934/1195 = 78.16% | **78.16%** | 78.16% | ✓ |
| `subagent_hybrid` | 886/1195 = 74.14% | **887/1195 = 74.23%** | 74.14%/74.23% | +1 record from extractor fallback (matches prior audit `a5d0f6457732180b9`) |

### Gemma 4 E4B Phase 12 modes (cluster, N=1195 each)

| Mode | Stored | Re-scored | Cited | Notes |
|---|---|---|---|---|
| `rag_snap_hyde` | 743/1195 = 62.18% | **62.18%** | 62.18% | ✓ same winner as 26B |
| `subagent_rag` | 728/1195 = 60.92% | **60.92%** | 60.92% | ✓ |
| `snap_hyde_report` | 726/1195 = 60.75% | **60.75%** | 60.75% | ✓ |
| `rag_hyde` | 724/1195 = 60.59% | **60.59%** | 60.59% | ✓ |
| `subagent_hyde` | 719/1195 = 60.17% | **60.17%** | 60.17% | ✓ |
| `subagent_hybrid` | 703/1195 = 58.83% | **58.83%** | 58.83% | ✓ |
| `rag_simple` | 699/1195 = 58.49% | **58.49%** | 58.49% | ✓ baseline |
| `snap_only_in_final` | 691/1195 = 57.82% | **57.82%** | 57.82% | ✓ |

All 8 E4B modes clean: 0 errors total, 1 None pred (in `rag_snap_hyde` — within normal noise).

### Llama 70b cross-family BarExam llm_only N=100 (2026-04-26)

| Path | Stored | Re-scored | Disagreements | Errors | None |
|---|---|---|---|---|---|
| `eval_llm_only_groq-llama70b_20260426_1930` | **81/100 = 81.0%** | **81/100** | 0 | 0 | 0 |

### Conclusion

All cited cluster + Llama-N=100 headline numbers are bulletproof. The cross-size method-effect story (`rag_snap_hyde` +3.69pp at E4B, +3.09pp at 26B over `rag_simple`) is the strongest paper claim and survives independent re-scoring. The only contaminated numbers in the entire experiments.jsonl pool are:
- `advisor_planning_table_groq-llama70b_20260426_2242` (BarExam 72%, tagged `_FAILED-EMPTY-RETRIEVAL`, do-not-cite)
- Qwen3-32b 68% (truncation caveat documented; not contaminated, just understated)

## Cleanup sweep findings 2026-04-27 — codex

Scope: all 38 `logs/experiments.jsonl` rows from the last 48h where `dataset == "musique"` or `mode in {"multi_hyde_diverse", "iter_hyde", "advisor_planning_table"}`. Every detail log opened and re-scored with current `extract_answer_musique` + `musique_em_f1` for MuSiQue, or `extract_answer_mc` for BarExam. No missing detail logs. Summary `correct/total` matched detail-log stored `is_correct/len(records)` on all 38 rows. `empty_retrieval_count` matched for the one failed-empty-retrieval row.

| Row | Stored EM | Re-scored EM | Cause | Verdict |
|---|---:|---:|---|---|
| `eval_rag_simple_groq-llama70b_20260426_1945_detail.jsonl` | 2/30 | **6/30** | four `Answer: <span>...</span>` wrappers now stripped (`Michael Bublé`, `Matt Damon`, `Saxony-Anhalt`, `Colin Firth`) | use re-scored 20.0%, not stored 6.7% |
| `eval_rag_snap_hyde_groq-llama70b_20260426_1946_detail.jsonl` | 3/30 | **4/30** | one `Answer: <span>Socialist Party of America</span>` wrapper recovered | use re-scored 13.3% |
| `eval_planning_table_groq-llama70b_20260426_1947_detail.jsonl` | 3/30 | **4/30** | one `Answer: <span>Colin Firth</span>` wrapper recovered | use re-scored 13.3% |

Placeholder echo check: searched `final_answer` strings in the same 38 detail logs for literal `<your answer here>` and `[your answer here]`. Found 3 `<your answer here>` echoes, all in pre-`0ff67ad` rows: `planning_table_no_snap_or-gemma4-26b_20260426_2003`, `rag_multi_query_or-gemma4-26b_20260426_2054`, and `advisor_planning_table_or-gemma4-26b_20260426_2224`. Found **0 echoes after commit `0ff67ad`**, so the prompt fix holds for post-fix rows.

Failed-row citation check: the only row tagged `_FAILED-EMPTY-RETRIEVAL` or `_FAILED-do-not-use` is `20260426_2242_advisor_planning_table_groq-llama70b_api-barexam-advisor-llama-n50` / `logs/eval_advisor_planning_table_groq-llama70b_20260426_2242_detail.jsonl`. It is not cited in `CLAUDE.md`, `RESEARCH.md`, `EXPERIMENTS.md`, `docs/experiment_overview.md`, or `docs/validation_log_2026-04-25.md`; only the generic guard string `_FAILED-do-not-use` appears in `RESEARCH.md` as harness behavior.

## multi_hyde_diverse cross-model N=30 (Phase 13) 2026-04-26 ~23:58 UTC

Audit of new MuSiQue HyDE variant that pools BM25 over 3 diverse hypothetical passages + raw question. Re-scored with `extract_answer_musique` + `musique_em_f1`.

### Per-log summary

| Log | N | Stored EM | Re-scored EM | Disagree | Errors | None | gold_retr | `<your answer here>` echo (angle/square) | `<span>`-wrapped FA |
|---|---|---|---|---|---|---|---|---|---|
| `eval_multi_hyde_diverse_groq-llama70b_20260426_2317` | 30 | 8 | **8/30 = 26.7%** | 0 | 0 | 0 | 24/30 | 0 / 0 | 0 |
| `eval_multi_hyde_diverse_or-gemma27b_20260426_2358` | 30 | 6 | **6/30 = 20.0%** | 0 | 0 | 0 | 25/30 | 0 / 0 | 0 |
| `eval_rag_simple_groq-llama70b_20260426_1945` | 30 | 2 | **6/30 = 20.0%** | 4 | 0 | 0 | 26/30 | 0 / 0 | 17/30 |
| `eval_rag_simple_or-gemma27b_20260426_2355` | 30 | 8 | **8/30 = 26.7%** | 0 | 0 | 0 | 26/30 | 0 / 0 | 0 |

`routed_to` field absent in every record (4×30); not used by these modes. Llama rag_simple's 4 disagreements are exactly the predicted pre-`<span>`-fix recoveries: `Michael Bublé`, `Matt Damon`, `Saxony-Anhalt`, `Colin Firth` — all literal `Answer: <span>X</span>` wraps now stripped by extractor.

### Paired stats (shared idx N=30 each)

| Pair | mhd EM | rag EM | Δ | b/c | McNemar exact 2-sided p | Bootstrap 95% CI on Δ (10K) |
|---|---|---|---|---|---|---|
| Llama 70b mhd vs rag_simple | 8/30 | 6/30 | **+6.7pp** | 3/1 | 0.6250 | [−6.67pp, +20.00pp] |
| Gemma 3 27B mhd vs rag_simple | 6/30 | 8/30 | **−6.7pp** | 1/3 | 0.6250 | [−20.00pp, +6.67pp] |

User's b/c/p numbers all confirmed exactly.

### Spot-checks (seed=42, 3/log)

All 12 extractor decisions agree with human judgment. Notable:
- `Rabbi Dovber Schneuri` vs gold `Dovber Schneuri` (Llama mhd) — judged "prefix mismatch", extractor correctly EM=False (no alias).
- `Low 40s and upper 30s °F` vs gold `upper 40s–lower 50s °F` — paraphrase, EM=False ✓ (gold not retrieved).
- `<span>Colin Firth</span>` (Llama rag) → extractor strips tags → EM=True ✓ (this is the bug-fix at work).
- `Minnesota` vs gold `Minnesota History Center` — partial answer, EM=False, judged "wrong" ✓.

### Cross-model interpretation

Mirror-symmetry (+6.7/−6.7) is **suspiciously clean and almost certainly noise**, not a real method×model interaction:

1. **Both pairs sit at McNemar p=0.625** — the b+c=4 discordant cells per pair are identical to flipping a coin 4 times and asking if 3-vs-1 is significant.
2. **Bootstrap CIs cross zero in BOTH directions** — neither delta is distinguishable from 0 at α=0.05.
3. **ZERO same-question opposite-method outcomes across models**: I checked the joint set (`mhd_T,rag_F` for one model AND `mhd_F,rag_T` for the other) — empty in both directions. The discordants are on disjoint questions, so the apparent "method helps Llama / hurts Gemma" story is not driven by any individual question where the methods truly diverge by model — it's just where each model's noise floor happened to fall.
4. **gold_retrieved 24-26/30 across all 4 logs** (≈83-87%) — retrieval is not the bottleneck at this slice; differences are dominated by reasoning/extraction noise.

### Verdict

- `eval_multi_hyde_diverse_groq-llama70b_20260426_2317_detail.jsonl` — **CLEAN** (cite as 8/30 = 26.7%, but per-pair effect not significant)
- `eval_multi_hyde_diverse_or-gemma27b_20260426_2358_detail.jsonl` — **CLEAN** (cite as 6/30 = 20.0%, but per-pair effect not significant)
- `eval_rag_simple_groq-llama70b_20260426_1945_detail.jsonl` — **CLEAN AFTER RESCORE** (cite as 6/30 = 20.0%, NOT stored 2/30; pre-`<span>`-fix run, must use re-scored value)
- `eval_rag_simple_or-gemma27b_20260426_2355_detail.jsonl` — **CLEAN** (cite as 8/30 = 26.7%)

## multi_hyde_diverse cross-model N=100 (Phase 13.5) 2026-04-27 ~00:25 UTC

Audit of NEW Gemma 3 27B mhd vs rag_simple paired N=100 logs anchoring the "mhd lifts MuSiQue cross-model" headline. Cross-checked against the Llama 70b N=100 pair re-scored in the same pass.

### Per-log integrity

| Log | N | Stored EM | Re-scored EM | None preds | Errors | Placeholder echoes | routed_to | gold_retrieved |
|---|---|---|---|---|---|---|---|---|
| `eval_multi_hyde_diverse_or-gemma27b_20260427_0025_detail.jsonl` | 100 | 30 | **30** | 0 | 0 | **0** | **0** | 91/100 = 91.0% |
| `eval_rag_simple_or-gemma27b_20260427_0012_detail.jsonl` | 100 | 22 | **22** | 0 | 0 | 0 | 0 | 83/100 = 83.0% |

Stored == re-scored exactly on both. **0 placeholder echoes** confirms `<your answer here>` template residue is absent. **0 routed_to fallbacks** confirms commit `5f8b723` invariant (no silent fallback added to mhd) holds. mhd's gold_retrieved is +8pp over rag_simple — diverse HyDE genuinely improves passage recall.

### Paired stats (re-derived from seed=42 bootstrap, 10000 iters)

- McNemar exact 2-sided: b=15 (mhd-only-correct), c=7 (rag-only-correct), discordants=22, **p=0.1338** ← matches claim (0.134)
- Both right=15, both wrong=63
- Bootstrap 95% CI on (mhd_em − rag_em): mean **+8.00pp**, **95% CI [-1.00pp, +17.00pp]** ← matches claim
- The CI brushes zero on the lower edge (CI=-1pp) ⇒ "trending" framing is honest; not p<0.05.

### Spot-check (seed=42, 5 each)

All 5 MHD spot-checks scored em=False are legitimate losses (substring partial-match, paraphrase, "Not specified", wrong-entity wikipedia paste). Extraction is doing the right thing on each: e.g. `Saxony` vs gold `Saxony-Anhalt` correctly EM=False (F1=0.67), `Ondine` vs full string EM=False (F1=0.18) — the model genuinely produced incomplete answers, not an extractor bug.

### 3 mhd-only-correct discordants — sampled (seed=43)

1. **qidx=2hop__123148_5385** (gold='11,900'): MHD retrieved correct passage about U. of Oklahoma and answered 11,900; RAG missed the alma-mater hop (`Cannot be determined`, gold_ret=False). **Genuine retrieval lift.**
2. **qidx=2hop__42578_55840** (gold='Colin Firth'): MHD found and named Colin Firth from passage; RAG didn't retrieve the King's Speech passage and fell back to the *character* `King George VI` (gold_ret=False). **Genuine retrieval+answer lift.**
3. **qidx=3hop1__465684_160545_60577** (gold='Ko Phi Phi Leh'): Both retrieved gold; RAG stopped at `Thailand` (1-hop), MHD chained to `Ko Phi Phi Leh`. **Genuine reasoning lift downstream of retrieval.**

None are lucky guesses. All 3 trace to the +8pp gold_retrieved lift or downstream multi-hop chaining over the same passages.

### Cross-FAMILY consistency (Llama 70b N=100 re-derived in same pass)

| Model | MHD | RAG | Δ | b | c | McNemar p | gold_ret MHD/RAG |
|---|---|---|---|---|---|---|---|
| Llama 3.3 70b dense | 33 | 21 | +12pp | 18 | 6 | **0.0227** | 86/83 |
| Gemma 3 27B dense | 30 | 22 | +8pp | 15 | 7 | **0.1338** | 91/83 |

Same direction (b > c, MHD > RAG), same gold_retrieved lift signature, same magnitude class. Llama crosses p<0.05 at the same N; Gemma is one or two unlucky discordants away from significance. **Consistent with a real cross-FAMILY effect** (Llama 3 vs Gemma 3, both dense), not a fluke on one model. Neither pair is a single coin-flip-narrow margin like the prior N=30 mirror.

### Verdict

- `eval_multi_hyde_diverse_or-gemma27b_20260427_0025_detail.jsonl` — **CLEAN** (cite as 30/100 = 30.0%; pair Δ=+8pp, p=0.134 trending; honest framing required)
- `eval_rag_simple_or-gemma27b_20260427_0012_detail.jsonl` — **CLEAN** (cite as 22/100 = 22.0%)
