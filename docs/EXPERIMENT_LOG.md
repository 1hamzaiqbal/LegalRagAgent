# Experiment Log

Chronological record of changes, experiments, and findings.

---

## Phase 1: Bootstrapping (2026-02-21)

### `fb436c1` Initial commit
- Empty repo scaffolding.

### `2788d6f` LangGraph + eval implementation
- First working pipeline: classify → plan → execute → evaluate loop.
- Basic eval framework (`eval_comprehensive.py`) with placeholder queries.

### `e4023f0` Wire real LLM integration + 5 skill prompts
- Connected to OpenAI-compatible API via `langchain-openai`.
- Added skills: `classify_and_route`, `plan_synthesis`, `synthesize_and_cite` (originally separate skills), `verify_answer`, `detect_prompt_injection`.
- Demo queries: simple (negligence), multi_hop (constitutional rights), medium (preliminary injunction).

### `6f54a40` Adaptive replanner + caching + observability
- Added `adaptive_replan.md` skill for multi-step research.
- LLM client singleton (`@lru_cache`), skill file caching.
- Prefix cache metrics logging for providers that support it (DeepSeek, vLLM, OpenAI).

### `e5de933` Implement remaining skills + fix 12 pipeline issues
- All 7 skills now functional.
- Fixed routing bugs, JSON parsing edge cases, state initialization issues.

### `6c66dfa` Batch test audit, migrate to uv, switch to Cerebras
- Migrated from pip to `uv` for dependency management.
- Switched default provider from OpenAI to Cerebras free tier (14K RPD).
- Fixed issues found during batch testing.

### `b0ea1c0` Calibrate thresholds + comprehensive eval
- Phase 1 eval: Recall@5/MRR on 953 QA pairs (retrieval only, no LLM).
- Phase 2 eval: full pipeline on diverse query set.
- Calibrated evaluator confidence threshold to MiniLM corpus characteristics.

**Baseline metrics (1K passages, MiniLM-L6-v2):**
| Metric | Value |
|--------|-------|
| Corpus size | 1,000 passages |
| Recall@5 | 16.4% (953 QA pairs) |
| MRR | 0.095 |
| Avg confidence | 0.446 |

---

## Phase 2: Retrieval Quality (2026-02-21 – 2026-02-22)

### Manual walkthrough findings (documented in DIAGNOSIS.md)
- Ran 5 bar exam questions manually, acting as the LLM.
- All 5 answered correctly by the LLM, but retrieval was the bottleneck.
- Identified 3 failure modes: keyword noise, terminological gaps, adversarial retrieval.
- See DIAGNOSIS.md for full per-question analysis.

### `09de734` Load full 220K corpus + hybrid retrieval
- Expanded from 1K to 220K passages.
- **Problem**: Caselaw (98.2%) drowned out MBE study material (1.1%).
- **Fix**: Source-aware hybrid retrieval — fetch from MBE/wex and caselaw pools separately, interleave 3 study + 2 caselaw.

### `98d95a8` Cross-encoder reranker (branch: `feature/corpus-expansion-and-tools`)
- Added `cross-encoder/ms-marco-MiniLM-L-6-v2` for two-stage retrieval.
- Bi-encoder over-retrieves 4x candidates, cross-encoder reranks to top k.
- **Qualitative win**: IIED/"murderer" keyword-noise query went from 2/5 → 5/5 relevant passages.
- **Recall@5 A/B test** (96 queries): bi-encoder 6.25% vs reranker 5.21%. Metric understates improvement — cross-encoder selects equally relevant but different passages than the gold.

### `fc3f44f` Merge feature/corpus-expansion-and-tools into main

**Metrics after Phase 2 (220K passages, MiniLM + cross-encoder):**
| Metric | Value |
|--------|-------|
| Corpus size | 220,000 passages |
| Recall@5 | 5.2% (down from 16.4% — dilution from larger corpus) |
| MRR | 0.021 |
| Avg confidence | 0.492 |

---

## Phase 3: Pipeline Quality (2026-02-22)

### `a42c6ff` Cleanup + merge synthesis/citation + multi-query retrieval
- Merged separate synthesis and citation skills into `synthesize_and_cite.md` (single-pass).
- Added `query_rewrite.md` skill: rewrites query into primary + 2 alternatives.
- Added `retrieve_documents_multi_query()`: pools bi-encoder candidates across all query variants, deduplicates, cross-encoder reranks.
- Bridges terminological gaps (e.g., "cancellation clause" also retrieves "illusory promise").

### `73b7fa0` Replanner guardrails + anti-fabrication
- Hard cap: 3 completed steps maximum for multi-hop queries.
- Stagnation detection: 3+ consecutive failures with similar low scores → stop.
- Added anti-fabrication rules to `synthesize_and_cite.md`: only state facts from evidence passages.

### `2e51030` Stale cleanup, parameterize vectorstore, WSL notes
- Parameterized ChromaDB collection name for flexibility.
- Added WSL setup instructions to CLAUDE.md.
- Cleaned up stale comments and dead code.

### Embedding model upgrade: MiniLM → gte-large-en-v1.5 (uncommitted at the time)
- Switched from `all-MiniLM-L6-v2` (384d, 22M params) to `Alibaba-NLP/gte-large-en-v1.5` (1024d, 434M params, 8192 token context).
- Corpus reduced to 20K passages for faster iteration during development.

### Provider registry expansion (uncommitted at the time)
- Expanded from 3-env-var config to 19-provider registry (Google AI Studio, Groq, OpenRouter, Cerebras, Ollama).
- Default switched to Gemma 3 27B via Google AI Studio (14.4K RPD, free).

### `94d2f12` MC correctness checking + memory cache clearing
- Fixed MC correctness checker: added `**Answer: (X)**` pattern (strict MC format from `skill_select_mc_answer`).
- Fixed case-sensitivity bug: uppercase pattern matched against lowered text → 2 correct answers were being missed.
- Phase 2 eval now clears QA memory cache before runs.
- MC answer choices now injected into `global_objective` so the pipeline can see them.
- Pinned Python < 3.14 in pyproject.toml.

**Metrics after Phase 3 (20K passages, gte-large, Gemma 3 27B):**
| Metric | Value |
|--------|-------|
| Corpus size | 20,000 passages |
| Recall@5 | 6.3% (60/953) |
| Avg confidence | 0.71 |
| MC accuracy | 3/5 (60%) on 6-query trace |

Key finding: Pipeline compensates for retrieval gaps. Multiple traced queries got MC correct despite gold passage NOT being in top-5 retrieval (0% Recall@5 on all 6 traced queries).

---

## Phase 4: Resilience & Efficiency (2026-02-22 – 2026-02-23)

### 6-query traced experiment audit
- Ran 6 bar exam questions through `eval_trace.py` with detailed diagnostics.
- Identified 4 improvements; 3 implemented (final-answer re-synthesis dropped — concat format already works).

### `6540460` Connection resilience, configurable threshold, MC classification
- `_llm_call`: now retries on connection/timeout errors (not just rate limits).
- `replanner_node`: try/except with graceful fallback to `complete` on persistent failure.
- Evaluator threshold configurable via `EVAL_CONFIDENCE_THRESHOLD` (default 0.6 for gte-large).
- `classify_and_route.md`: MC-specific guidance — single-concept MC → simple, multi-concept → multi_hop.
- `adaptive_replan.md`: aligned "don't re-ask" threshold from 0.4 → 0.6.

---

## Branches

| Branch | Status | Summary |
|--------|--------|---------|
| `main` | Active | All core work |
| `feature/corpus-expansion-and-tools` | Merged + deleted | Cross-encoder reranker + full corpus |
| `feature/musique-eval` | Unmerged (remote only) | MuSiQue multi-hop QA eval (550 lines, 2 new files). Self-contained, no conflicts expected. |

---

## Next Steps

### High Priority
1. **Commit uncommitted files** — `llm_config.py`, `rag_utils.py`, `eval_comprehensive.py`, `eval_trace.py` contain the provider registry, embedding upgrade, retrieval refactor, and trace evaluator. The committed `main.py` has a broken import without these.
2. **Retrieval quality** — Recall@5 of 6.3% is the main bottleneck. The pipeline compensates via multi-query + synthesis, but better retrieval would directly improve MC accuracy. Options:
   - Expand corpus back to 220K+ (or full 686K) with gte-large (requires ~3.5hr+ embed time)
   - Hybrid search: combine dense (embedding) with sparse (BM25) retrieval
   - Fine-tune embedding model on legal domain
3. **Merge `feature/musique-eval`** — Self-contained MuSiQue multi-hop QA eval. Provides a general-domain benchmark alongside the bar-exam-specific eval.

### Medium Priority
4. **Run full eval sweep** — After committing all changes, run `eval_comprehensive.py` (both phases) and `eval_trace.py` to establish post-improvement baselines.
5. **Evaluator threshold tuning** — The new 0.6 default is theoretical. Run eval with `EVAL_CONFIDENCE_THRESHOLD=0.5`, `0.6`, `0.7` to find the sweet spot where failed steps actually get useful retries.
6. **Classification effectiveness** — Track how many MC questions get classified as `simple` vs `multi_hop` after the new guidance. Measure LLM call savings.

### Lower Priority
7. **Unit tests** — No test suite exists. Key functions to test: `_parse_json`, `_check_mc_correctness`, routing functions, skill fallbacks.
8. **Web search fallback** — For topics absent from the corpus entirely (identified in DIAGNOSIS.md as failure mode #3).
9. **Rate limit awareness** — Eval scripts don't pace themselves. Long runs on low-RPD providers (Groq: 1K RPD) can exhaust quota mid-eval.
