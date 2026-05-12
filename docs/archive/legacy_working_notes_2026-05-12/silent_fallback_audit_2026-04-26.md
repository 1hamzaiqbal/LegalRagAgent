# Silent-fallback / confounder hunt — 2026-04-26 (round 4)

Subagent-produced deep audit specifically targeting silent fallbacks
that could quietly inflate or deflate results. Top 3 confounders flagged
for immediate fix.

## Top 3 confounders worth fixing before meeting

### 1. `_run_gap` no-gap fallback to plain `llm_only` (HIGH)

When `_gap_analysis` returns `[]` (model says NONE or fails to parse),
the runner silently falls back to a fresh `llm_only` call but logs the
result as the original mode (e.g. `subagent_rag`).

**Empirical impact**: `eval_subagent_rag_or-gemma4-26b_20260426_0501_detail.jsonl`
shows **6/15 (40%)** records hit the no-gap branch on this small-model run.
Cluster runs with the bigger model are <1%, but any "subagent_X scored Y"
comparison across model sizes is partly `llm_only` math wrapped in a
subagent label.

Fix: add `routed_to: "llm_only_fallback"` field; either error-out or
split summary by route.

### 2. MuSiQue multi-gold `gold_retrieved` check (HIGH)

`_retrieve_and_format` line 929 does `gold_idx in retrieved_ids` — fine
for BarExam (single-id strings) but broken for MuSiQue (comma-separated
multi-hop). The musique_in_row helper splits correctly, but
`run_full_pipeline`, `run_vectorless_keyword`, and entity_search bypass
that helper and silently report False on MuSiQue.

Fix: centralize gold_retrieved into a helper that splits on commas.

### 3. `extract_answer_mc` has no "last standalone letter" fallback (MEDIUM)

housing extractor has a fallback to last standalone Y/N; MC has no
equivalent for last standalone A-D letter. Models that answer in prose
("After review, B.") are auto-FAILed.

Possibly a partial explanation for cross-size scaling gaps — smaller
models drop the `Answer:` marker more often. One-line fix.

## Other findings (LOW/MEDIUM, defer)

- **1.1 (MEDIUM)**: HyDE sanitizer falls back to question-text on full strip.
  Currently 0% firing on Llama 70b but small models could trigger this silently.
  Add summary counter: `silent_hyde_fallback_n`.
- **1.4 (MEDIUM)**: `_retrieve_musique_in_row` returns top-5 by index when
  all queries are empty (deterministic position-0-4 retrieval). Add assertion.
- **1.5 (MEDIUM)**: Entity-search graph-missing degrades silently to
  `llm_only`. Add `routed_to` marker.
- **2.3 (LOW)**: 71.9% of MuSiQue rows have empty `answer_aliases` —
  semantically correct, but EM scoring is more lenient on the 28% with aliases.
- **2.4 (MEDIUM)**: Open-ended evaluators use SAME provider for both
  answer + judge → same-model bias. Pin a held-out judge.
- **5.3 (MEDIUM)**: `SEED=42` only affects question sampling, NOT LLM
  calls. Reproducibility claim should be qualified.
- **7.3 (MEDIUM)**: Summary-guard fires at >50% error rate; 30-49% slips
  through silently. Drop threshold to 20%.

## False alarms

- BarExam `gold_idx` populated in 100% of rows (no missing-gold issue)
- `get_llm()` LRU cache cleared on provider switch (correct)
- `_CALL_TRACE` reset per question (correct)
- `experiments.jsonl` is append-only, parse-clean (no partial writes)

## Action plan

Applying fixes 1 + 2 + 3 immediately. Auditing existing subagent_rag
cluster logs to quantify how much of the 26B subagent_rag (78.16%) and
26B subagent_hybrid (74.14%) are actually llm_only fallbacks.
