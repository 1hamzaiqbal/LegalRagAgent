# Pipeline Flags (2026-02-23)

Systematic audit of yellow/red flags across the codebase. Each flag has severity, evidence from traces, and proposed fixes.

**Latest eval baseline** (8-query trace, Gemma 3 27B, 20K passages, threshold 0.70, cross-step dedup ON):
- MC accuracy: 5/6 (83%)
- Passage diversity: 100% unique docs on every query
- Steps with failures: 3/26 (evaluator now rejects borderline steps)
- Avg LLM calls: 11.9/query

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

---

## RED FLAGS (open)

### R1. Verifier always passes
- **File**: `main.py` verify_answer_node, `skills/verify_answer.md`
- **Evidence**: 8/8 traced queries pass first attempt (both before and after threshold/dedup changes). Corrective-step path has never triggered in any traced eval.
- **Root cause**: Skill says "pass by default", "when in doubt PASS". Verifier sees the same evidence the synthesizer used — it's checking whether the LLM contradicted itself.
- **Impact**: 1 wasted LLM call/query (~8% of total). False "verified" label. Dead code in corrective-step path.
- **Fix options**:
  - **(A) Independent evidence**: Re-retrieve for the *answer text* (not the original question) and verify against those passages. Catches claims not grounded in any retrievable passage. Adds 1 retrieval call but no LLM call.
  - **(B) Remove verifier**: Save the LLM call. Accept that synthesis quality is the synthesizer's job. Simplest change, saves ~5s/query.
  - **(C) Adversarial verifier**: Change skill to "try to find problems." Risk: false failures increase LLM calls via retries.
- **Recommendation**: (B) is safest. The verifier adds no signal. If we want verification later, (A) is the right architecture.

### R2. QA memory cache serves stale answers
- **File**: `rag_utils.py` check_memory/write_to_memory, `main.py` planner_node/memory_writeback_node
- **Evidence**: Caused wrong answer in testing (cached MC letter from prior model/skill config). Now mitigated by clearing cache at eval start.
- **Root cause**: No invalidation on skill/model/corpus changes. Writes at confidence >= 0.45 (everything qualifies). Caches full answer including MC selection.
- **Impact**: Mitigated for eval (auto-clear). Still a risk for demo/production use where cache persists across sessions.
- **Fix options**:
  - **(A) Strip MC from cached answer**: Only cache the research portion (`_aggregate_completed_answers`), not the `--- **Answer: (X)**` MC selection. MC selection re-runs each time.
  - **(B) Version the cache**: Store a hash of skill file contents + model name in metadata. Invalidate on mismatch.
  - **(C) Raise write threshold**: From 0.45 to 0.70 (match eval threshold). Fewer but higher-quality cache entries.
- **Recommendation**: (A) + (C) together. Strip MC, raise write threshold.

### R3. Silent exception swallowing in rag_utils.py
- **File**: `rag_utils.py` lines 164-169, 170-176, and 4 more in multi-query source-diverse path
- **Evidence**: 6 instances of `except Exception: pass` in source-diverse retrieval paths.
- **Root cause**: Intended for ChromaDB filter errors when a source type doesn't exist. Catches everything.
- **Impact**: Low today (source-diverse is off by default). Any ChromaDB error (disk full, corruption) is silently swallowed.
- **Fix options**: Replace bare `except Exception: pass` with `except Exception as e: logger.warning("retrieval pool error: %s", e)`. No behavior change, just visibility.
- **Recommendation**: Trivial fix. Just add logging.

---

## YELLOW FLAGS (open)

### Y2. Classifier always picks multi_hop for MC
- **File**: `main.py` classifier_node, `skills/classify_and_route.md`
- **Evidence**: 6/6 MC queries classified multi_hop. Only out-of-corpus got simple.
- **Root cause**: Skill says "when in doubt, classify as multi_hop."
- **Impact**: Every MC query costs 12+ LLM calls instead of 5. Not a correctness issue — may even help (more research = more evidence for MC selector).
- **Recommendation**: Accept. Safety margin is worth the cost on free-tier models.
- **Status**: Accepted

### Y4. Raw answer concatenation (no merge/dedup)
- **File**: `main.py` `_aggregate_completed_answers()`
- **Evidence**: Final answer is 3 step answers joined by `---`. Source citations restart per step ([Source 1] in step 1 vs [Source 1] in step 2 are different passages).
- **Impact**: MC selector sees overlapping content. Citation references are ambiguous across steps. With dedup fix (F2), content overlap is reduced but citation numbering still restarts.
- **Fix options**:
  - **(A) Global citation renumbering**: Post-process the joined answer to renumber [Source N] sequentially across steps. No LLM call.
  - **(B) Section headers**: Prefix each step's answer with `### Step N: {phase}` so the MC selector can distinguish.
  - **(C) Accept as-is**: MC selector at 83% accuracy handles it well enough.
- **Recommendation**: (A) is clean and mechanical. Could combine with (B).

### Y6. String-matching for retryable errors
- **File**: `main.py` `_llm_call()` lines 190-193
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

---

## Priority queue (next changes)

1. **R3** — Add logging to silent exception catches (trivial, pure safety)
2. **R1** — Remove verifier or give it independent evidence (saves 1 LLM call/query, removes false confidence)
3. **R2** — Strip MC from cached answers + raise write threshold (prevents stale MC answers)
4. **Y4** — Global citation renumbering in aggregated answer (cleaner MC selector input)
