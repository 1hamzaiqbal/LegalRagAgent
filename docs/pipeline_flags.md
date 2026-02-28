# Pipeline Flags (2026-02-23)

Systematic audit of yellow/red flags across the codebase. Each flag has severity, evidence from traces, and proposed fixes.

**Latest eval baseline** (8-query trace, Gemma 3 27B, 20K passages, threshold 0.70, cross-step dedup ON, verifier removed):
- MC accuracy: 4/6 (torts Y, contracts Y, crimlaw N, evidence Y, constlaw N, realprop Y)
- LLM calls: 11/query multi_hop MC, 10 non-MC, 4 simple
- Passage diversity: 100% unique docs on every query
- Citation format: `[Query X][Source Y]` with step headers

---

## FIXED

### F1. Confidence threshold was a no-op (was Y1)
- **Problem**: Threshold 0.6, observed range 0.709-0.804. 0/24 steps failed.
- **Fix**: Raised default to 0.70. Now 3/26 steps fail, triggering replanner retries.
- **Result**: constlaw 3c/2f (18 LLM calls), crimlaw 3c/1f (15 calls). Others unchanged at 12.

### F2. No cross-step passage deduplication (was Y3)
- **Problem**: Contracts had 7/15 unique docs (53% overlap). Avg ~73% unique.
- **Fix**: `exclude_ids` param on `retrieve_documents_multi_query()`. Executor gathers prior doc_ids.
- **Result**: 100% unique docs on every query. MC accuracy rose from 4/6 to 5/6 (evidence query flipped correct).

### F3. Single-quote JSON hack (was Y5)
- **Problem**: `fixed.replace("'", '"')` would corrupt apostrophes in text values.
- **Fix**: Removed the line. Broken JSON now falls through to parse failure counter honestly.

### F4. Stale "expectation" key in fallback (was Y8)
- **Problem**: Fallback plan dict still had `"expectation"` key removed from PlanStep model.
- **Fix**: Removed the key.

### F5. Verifier always passes (was R1)
- **Problem**: 8/8 traced queries pass first attempt. Corrective-step path never triggered. 1 wasted LLM call/query.
- **Fix**: Removed verifier LLM call. Auto-pass verification. Kept MC selection in verify_answer_node. Removed dead corrective-step retry code and `verification_retries` state field.
- **Result**: 11 LLM calls/query (down from 12). Skill file `verify_answer.md` retained for future use.

### F6. QA memory cache serves stale MC answers (was R2)
- **Problem**: Cached full answer including MC letter selection. Write threshold 0.45 (everything qualifies).
- **Fix**: Strip MC selection (`**Answer: (X)**` block) before caching. Raise write threshold to 0.70.
- **Result**: MC selection re-runs fresh each time. Only high-confidence answers cached.

### F7. Silent exception swallowing (was R3)
- **Problem**: 6 instances of `except Exception: pass` in source-diverse retrieval paths.
- **Fix**: Added `import logging` and `logger = logging.getLogger(__name__)` to rag_utils.py. All catches now log `logger.warning("retrieval pool error: %s", e)`.

### F8. Raw answer concatenation (was Y4)
- **Problem**: Source citations restarted per step. `[Source 1]` in step 1 vs step 2 were different passages.
- **Fix**: `_aggregate_completed_answers()` now adds `### Step N: {phase}` headers and rewrites `[Source N]` → `[Query X][Source N]`.

---

## YELLOW FLAGS (open)

### Y2. Classifier always picks multi_hop for MC
- **File**: `main.py` classifier_node, `skills/classify_and_route.md`
- **Evidence**: 6/6 MC queries classified multi_hop. Only out-of-corpus got simple.
- **Root cause**: Skill says "when in doubt, classify as multi_hop."
- **Impact**: Every MC query costs 11 LLM calls instead of ~4. Not a correctness issue — may even help.
- **Recommendation**: Accept. Safety margin is worth the cost on free-tier models.
- **Status**: Accepted

### Y6. String-matching for retryable errors
- **File**: `main.py` `_llm_call()` lines 188-192
- **Evidence**: `"connection" in err_str` could match "connection type is invalid" (not transient).
- **Impact**: No observed false matches. Worst case: 2 extra retries with delay on a non-transient error.
- **Recommendation**: Low priority. Could tighten to check for specific HTTP status codes in the string.
- **Status**: Low priority

### Y7. step_id is a float
- **File**: `main.py` PlanStep model
- **Evidence**: All observed values are x.0. JSON parsing returns floats by default.
- **Impact**: Cosmetic. Works fine.
- **Recommendation**: Low priority. `int(s.get("step_id", ...))` at construction time.
- **Status**: Low priority

### Y9. Graph rebuilds per query in eval_trace.py
- **File**: `eval_trace.py` `trace_full_pipeline()`
- **Evidence**: `build_graph()` called inside the function. 8 compilations vs 1.
- **Impact**: Milliseconds. Not meaningful vs 60-80s of LLM calls.
- **Recommendation**: Low priority. Move `build_graph()` to `main()` and pass app as arg.
- **Status**: Low priority

### Y10. Count-based skip in load_passages_to_chroma
- **File**: `rag_utils.py` `load_passages_to_chroma()` line 108-110
- **Evidence**: `if existing_count >= len(documents): skip`. Switching from 20K to 1.5K curated silently keeps 20K.
- **Impact**: Must manually clear when switching to smaller corpus. Known workflow.
- **Recommendation**: Low priority. Could store source CSV hash in collection metadata.
- **Status**: Low priority

### Y11. Evidence stored as raw strings with no per-doc metadata
- **File**: `main.py` executor_node, `step.execution["sources"]`
- **Evidence**: Parallel lists: `sources` (text strings) and `retrieved_doc_ids` (idx strings). If they desync, no way to trace which passage supported which claim.
- **Impact**: Fragile but works. Would matter if we added per-passage provenance tracking.
- **Recommendation**: Low priority. Merge into list of `{"idx": str, "text": str}` dicts when needed.
- **Status**: Low priority
