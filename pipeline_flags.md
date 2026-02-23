# Pipeline Flags (2026-02-23)

Systematic audit of yellow/red flags across the codebase. Each flag has severity, evidence from traces, and proposed fixes.

## RED FLAGS

### R1. Verifier always passes (verify_answer_node)
- **Evidence**: 8/8 traced queries passed first attempt. Corrective-step path is dead code in practice.
- **Root cause**: Skill says "pass by default", "when in doubt PASS", and verifier sees the *same evidence* the synthesizer used — it's checking whether the LLM contradicted itself, which it rarely does.
- **Impact**: 1 wasted LLM call/query. False "verified" label on every answer.
- **Fix options**: (A) Give verifier independently-retrieved evidence. (B) Remove verifier, save the call. (C) Make verifier adversarial — but risk false failures.
- **Status**: Open

### R2. QA memory cache serves stale answers
- **Evidence**: Hit in testing — cached answer returned wrong MC letter (B) when correct was (C). Had to manually clear cache.
- **Root cause**: No invalidation on skill/model/corpus changes. Writes at confidence >= 0.45 (everything passes). Caches full final answer including `**Answer: (X)**` MC selection.
- **Impact**: Silent wrong answers in production/demo use.
- **Fix options**: (A) Disable by default, opt-in via env var. (B) Version cache with hash of skills+model. (C) Don't cache MC selections, only research.
- **Status**: Open

### R3. Silent exception swallowing in rag_utils.py
- **Evidence**: 6 instances of `except Exception: pass` in source-diverse retrieval paths.
- **Root cause**: Intended to handle ChromaDB filter errors for missing source types, but catches everything.
- **Impact**: Low today (source-diverse is off by default), but any ChromaDB error (disk, corruption) would be silently ignored.
- **Fix options**: Catch specific exceptions + log warnings.
- **Status**: Open

## YELLOW FLAGS

### Y1. Confidence score is a rubber stamp (evaluator_node)
- **Evidence**: 0/24 steps failed across 8 traces. Observed range: 0.709-0.804. Threshold: 0.6.
- **Root cause**: Threshold too low for gte-large. Metric (mean cosine) measures query-doc proximity, not answer quality.
- **Impact**: Evaluator is a no-op. Every step passes regardless of quality.
- **Fix options**: (A) Raise threshold to 0.70+. (B) Use max instead of mean. (C) Use cross-encoder score instead of bi-encoder cosine.
- **Status**: FIXED — raised default to 0.70

### Y2. Classifier always picks multi_hop for MC
- **Evidence**: 6/6 MC queries classified multi_hop. Only out-of-corpus got simple.
- **Root cause**: Skill says "when in doubt, classify as multi_hop."
- **Impact**: 2.4x more LLM calls for simple questions (12 vs 5). Not a correctness issue.
- **Fix options**: Adjust skill guidance, or accept cost for safety.
- **Status**: Accepted (safety margin worth the cost)

### Y3. No cross-step passage deduplication
- **Evidence**: Contracts had 7/15 unique docs (53% overlap across 3 steps).
- **Root cause**: Each step retrieves independently. Replanner doesn't know what was retrieved.
- **Impact**: Wastes retrieval capacity. Redundant content in final answer.
- **Fix options**: Pass `exclude_ids` from prior steps to `retrieve_documents_multi_query()`.
- **Status**: FIXED — executor_node gathers prior doc_ids, passes as exclude_ids

### Y4. Raw answer concatenation (no merge/dedup)
- **Evidence**: Final answer is 3 step answers joined by `---`. Source citations (Source 1, Source 2...) restart per step, creating ambiguity.
- **Root cause**: `_aggregate_completed_answers()` does simple string join.
- **Impact**: MC selector and verifier see redundant/overlapping content. Citation references are ambiguous across steps.
- **Fix options**: (A) Global citation renumbering. (B) Lightweight merge/dedup LLM call. (C) Accept as-is (MC selector handles it).
- **Status**: Open

### Y5. Single-quote JSON fix corrupts apostrophes
- **Evidence**: Line 90 of main.py: `fixed = fixed.replace("'", '"')` — replaces ALL single quotes.
- **Root cause**: Last-resort heuristic for malformed JSON from LLM.
- **Impact**: Probably never fires. If it did, would corrupt English text containing apostrophes.
- **Fix options**: Remove the line. If JSON is that broken, let it fail and count as parse failure.
- **Status**: FIXED — removed

### Y6. String-matching for retryable errors
- **Evidence**: `"connection" in err_str` could match non-transient errors mentioning "connection."
- **Root cause**: LangChain wraps HTTP errors as generic exceptions with string messages.
- **Impact**: No observed false matches, but fragile.
- **Fix options**: Tighten patterns or check exception types.
- **Status**: Low priority

### Y7. step_id is a float
- **Evidence**: All observed step_ids are x.0 (1.0, 2.0, 3.0). Never fractional.
- **Root cause**: JSON parsing returns numbers as float by default.
- **Impact**: Semantically wrong type, cosmetic only.
- **Fix options**: Cast to int at PlanStep construction.
- **Status**: Low priority

### Y8. Stale "expectation" key in fallback plan
- **Evidence**: `skill_plan_synthesis` fallback dict (line 236) still has `"expectation"` key. PlanStep no longer has this field.
- **Root cause**: Missed during dead-state cleanup.
- **Impact**: Silently ignored by Pydantic v2. Would break on `extra = "forbid"`.
- **Fix options**: Remove the key from fallback dict.
- **Status**: FIXED — removed

### Y9. Graph rebuilds per query in eval_trace.py
- **Evidence**: `build_graph()` called inside `trace_full_pipeline()` — 8 compilations vs 1.
- **Root cause**: Different pattern from eval_comprehensive.py which builds once.
- **Impact**: Milliseconds. Not meaningful.
- **Status**: Low priority

### Y10. Count-based skip in load_passages_to_chroma
- **Evidence**: If existing collection has 20K docs and you try to load 1.5K curated, it skips (20K >= 1.5K).
- **Root cause**: Only compares count, not content.
- **Impact**: Must manually clear when switching from larger to smaller corpus.
- **Fix options**: Store source config in collection metadata, auto-clear on mismatch.
- **Status**: Low priority (known workflow)

### Y11. Evidence stored as raw strings with no per-doc metadata
- **Evidence**: `step.execution["sources"]` is a list of strings parallel-indexed with `retrieved_doc_ids`.
- **Root cause**: Simple design — metadata lives in a separate list.
- **Impact**: Fragile parallel indexing. Would break if lists got out of sync.
- **Fix options**: Store as list of `{"idx": ..., "text": ..., "source": ...}` dicts.
- **Status**: Low priority
