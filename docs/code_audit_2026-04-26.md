# Code audit — eval/eval_harness.py + eval/eval_config.py — 2026-04-26

Empirical audit before scaling. **Verdict: SAFE TO SCALE 5-10x** with two
small fixes (both applied below).

## Truncation findings

- **No model-input truncation found.** All `[:N]` slices in eval_harness.py
  (lines 3953-3990) write to per-question detail records only — `record["question"]`,
  `record["final_answer"]`, etc. get `[:500]`. Model-facing prompts
  (`question = _fmt(...)`, `user = f"## Retrieved Passages..."`) are NEVER
  sliced.
- **No `max_tokens` cap on model output** — `ChatOpenAI` constructed without it.
- `_preview_text` (limit=1200) is log-only.
- List slicing (`[:3]` for gaps/TODOs, `[:5]` for keywords) is intentional.
- 4 sample detail logs across modes — all `final_answer` text complete, no mid-token cuts.

## Dead code / dedup candidates

| Mode | Status | Recommendation |
|---|---|---|
| `vectorless_keyword` | 0 runs | DELETE — bypasses _retrieve_and_format with inconsistent code path |
| `snap_hyde_report` + `snap_hyde_report_snap` | 95% identical bodies | CONSOLIDATE — single `_run_snap_hyde_report(include_snap)` |
| `ce_threshold` + `ce_threshold_k3` | Differ only in k=5 vs k=3 | CONSOLIDATE — pass k as parameter |
| All 6 subagent_* variants | Already share `_run_gap` | KEEP — clean ablation grid |
| All 8 vectorless_* | Mostly share `_run_vectorless` | KEEP |
| All gap_* / golden_* / entity_* | All actively used | KEEP |

Dedup wins available: collapsing 2 pairs above removes ~120 LOC.

## Inconsistent fields

- **5 modes never return retrieval fields**: `run_decompose`, `run_llm_only`,
  `run_self_verify`, `run_snap_debate`, `run_snap_only_in_final`. Harness loop
  fills with `setdefault` defaults at lines 4000-4002 → silently False/empty.
  **Safe but invisible** in analysis.
- **MuSiQue gold check is dataset-aware** in `_retrieve_musique_in_row` but
  bare `gold_idx in retrieved_ids` in shared retrieval helpers. Fine for
  current paths since MuSiQue routes through the BM25 helper, but if a mode
  ever uses a non-musique branch on MuSiQue the gold check would silently
  return False on hits.

## 🔴 Sanitization gaps (FIXED THIS COMMIT)

Two real leaks where snap output was passed downstream without
`_strip_answer_line`. Every other snap-aware mode strips. Fixed:

1. **`run_snap_rag` (line 1900)**: `f"## Your Initial Answer\n{snap_answer}\n\n"` → fixed to use `_strip_answer_line(snap_answer)` and renamed header to "Your Initial Reasoning".
2. **`run_vectorless_hybrid` (line 2135)**: `f"## Student's Initial Analysis\n{snap_answer}\n\n"` → fixed to use `_strip_answer_line(snap_answer)`.

Both modes have only 1-2 historical runs (low blast radius), but if anyone
re-runs them post-fix the result will not be inflated by snap-letter echo.

## Schema consistency

- Cluster runs carry `call_trace` + `trace_events` (env var enables);
  Mac-local API runs do not. Downstream analysis must check key presence,
  not assume schema.
- One detail log row had model emitting `Answer: (C)` twice — `extract_answer_mc`
  returns the LAST match, so correctly scored. No-op for current grading.

## Other gotchas

- **`seed=42` not propagated to LLM calls** — only to `pd.DataFrame.sample(random_state=...)`.
  At temp=0 most providers are deterministic. Groq adds jitter (visible in
  confidence_gated logs). For vLLM at temp=0, `confidence_gated` will route
  ~100% to skip_rag, defeating the gating premise.
- **`--questions curated --dataset housing` would hard-fail** at int cast.
  Fail-loud is fine.
- `gold_idx` and `retrieved_ids` are str-cast everywhere. Consistent.
- **End-of-run write only** — already documented in CLAUDE.md.
- `_llm_call` retry layer already added (commit `ffe767a`) — catches
  JSONDecodeError/Connection/Timeout once.

## Verdict

**Safe to scale 5-10x.** No model-input truncation, no max_tokens cap, schema
consistent (defaults backfilled), final_answer text complete in all sampled
logs.

Pre-flight fixes applied this audit:
- Wrap snap_answer in `_strip_answer_line(...)` at lines 1900 + 2135 (snap_rag, vectorless_hybrid)

Optional cleanup deferred:
- Delete `vectorless_keyword` (never used)
- Consolidate `ce_threshold*` and `snap_hyde_report*` pairs

Skip on temp=0 vLLM:
- `confidence_gated` and `double_snap` (no per-call randomness → degenerate)
