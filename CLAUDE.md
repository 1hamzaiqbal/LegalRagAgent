# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Legal RAG system built on LangGraph. It uses a classify-plan-execute-evaluate loop to answer legal research questions by retrieving passages from a ChromaDB vector store of bar exam materials. LLM calls are made via `langchain-openai`'s `ChatOpenAI`, compatible with any OpenAI-compatible API (Groq, Ollama, OpenAI, etc.).

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run the agent with a demo query
uv run python main.py simple       # Single-concept question (negligence elements)
uv run python main.py multi_hop    # Multi-concept constitutional rights scenario
uv run python main.py medium       # Preliminary injunction standard

# Run comprehensive evaluation (retrieval + full pipeline)
uv run python eval_comprehensive.py

# Run retrieval A/B test (bi-encoder vs cross-encoder reranking)
uv run python eval_reranker.py

# Verify LLM config
uv run python -c "from llm_config import get_llm; print(get_llm())"
```

## Environment Setup

Copy `.env.example` to `.env` and set your API key. Default uses Cerebras (generous free tier — 14K requests/day, 1M tokens/day):

```bash
cp .env.example .env
# Edit .env with your Cerebras API key from https://cloud.cerebras.ai
```

## Architecture

### LangGraph Workflow (main.py)

Nine-node state machine with adaptive replanning, injection detection, answer verification, QA memory, and observability:

1. **detect_injection_node** — Screens input for adversarial prompt injection using `detect_prompt_injection.md` skill. Skippable via `SKIP_INJECTION_CHECK=1` env var (saves 1 LLM call for eval/testing). Unsafe inputs are rejected and routed to observability for metrics.
2. **classifier_node** — Classifies objective as `simple` or `multi_hop` using `classify_and_route.md` skill
3. **planner_node** — Checks QA memory first (cosine similarity >= 0.92); on cache hit, short-circuits to memory writeback. Otherwise generates initial plan. For `multi_hop`, emits only the first step; replanner handles the rest adaptively.
4. **executor_node** — For each pending step: rewrites query into primary + 2 alternatives (`query_rewrite.md`, JSON output), multi-query retrieves from ChromaDB (`retrieve_documents_multi_query`), synthesizes answer with inline `[Source N]` citations in a single pass (`synthesize_and_cite.md`), computes confidence via cosine similarity
5. **evaluator_node** — Marks steps completed (confidence >= 0.6) or failed (< 0.6). Accumulates step summaries into `accumulated_context`. Sets explicit failure message if all steps fail.
6. **replanner_node** — (multi_hop only) Receives objective + accumulated evidence, decides: `next_step` (add new research step), `retry` (rephrase failed step), or `complete` (aggregate final answer).
7. **verify_answer_node** — Cross-checks final answer against evidence using `verify_answer.md` skill. On first failure, adds a corrective step using the LLM's `suggested_query` (a proper legal question). Second failure terminates cleanly without orphaned steps.
8. **memory_writeback_node** — Persists successful QA pairs (avg confidence >= 0.7, verified) to ChromaDB `qa_memory` collection (cosine distance) for future cache hits. Skips write if verification failed.
9. **observability_node** — Aggregates and prints run metrics: LLM calls, char usage, parse failures, steps completed/failed, memory hit status, verification retries, has_answer, injection status.

Routing:
- After injection check: 2-way — `classifier` (safe) | `observability` (unsafe, for metrics)
- After planner: 2-way — `executor` (no memory hit) | `memory_writeback` (memory hit)
- After evaluator: 3-way — `executor` (pending steps) | `replanner` (multi_hop, all done, no prior correction) | `verify_answer` (simple done, iteration limit, or correction done)
- After replanner: 2-way — `executor` (new step added) | `verify_answer` (complete)
- After verify: 2-way — `executor` (not verified, first failure) | `memory_writeback` (verified or retry exhausted)
- After memory writeback: fixed — `observability → END`

Graph: `detect_injection → {classifier | observability}`; `classifier → planner → {executor | memory_writeback}`; `executor → evaluator → {executor | replanner | verify_answer}`; `replanner → {executor | verify_answer}`; `verify_answer → {executor | memory_writeback}`; `memory_writeback → observability → END`

### Shared State (`AgentState`)

TypedDict with `global_objective`, `planning_table` (list of `PlanStep`), `query_type` ("simple"/"multi_hop"), `final_cited_answer`, `accumulated_context` (step summaries for replanner), `iteration_count` (loop guard, max 4), `injection_check` (safety result), `verification_result` (answer verification with `suggested_query`), `verification_retries` (1 corrective retry max), `memory_hit` (QA cache result, threshold 0.92), and `run_metrics` (observability data including parse failures and has_answer).

### Skill System (skills/)

7 markdown prompt files cached at first load via `@lru_cache` in `load_skill_instructions()`:
- `classify_and_route.md` — classify query complexity
- `plan_synthesis.md` — generate research plan
- `query_rewrite.md` — rewrite query into primary + 2 alternatives (JSON output for multi-query retrieval)
- `synthesize_and_cite.md` — synthesize answer with inline `[Source N]` citations and `## Sources` map in a single pass
- `adaptive_replan.md` — decide next research step based on accumulated evidence
- `detect_prompt_injection.md` — screen input for adversarial prompts (fail-open, skippable via `SKIP_INJECTION_CHECK=1`)
- `verify_answer.md` — cross-check answer against evidence for unsupported claims, contradictions, and missing info

### LLM Config (llm_config.py)

- Single `get_llm()` function returning a cached `ChatOpenAI` singleton (`@lru_cache`)
- Configured via env vars: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- Default: Cerebras with `llama-3.3-70b` (14K RPD, 1M TPD free tier)
- Also supports: Google AI Studio, Groq, DeepSeek, vLLM with automatic prefix-cache hit logging
- Rate limit retry with exponential backoff (3 attempts, 2s/4s/8s delays)

### RAG / Retrieval (rag_utils.py)

- Embeddings: HuggingFace `all-MiniLM-L6-v2` (bi-encoder)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (cross-encoder, cached singleton)
- Vector store: ChromaDB persisted to `./chroma_db/`, 220K passages (98% caselaw, 1% MBE, 1% wex)
- Two-stage retrieval in `retrieve_documents(query, k=5)`:
  1. **Bi-encoder over-retrieval**: Fetch 4x candidates from MBE/wex and caselaw pools separately (source-filtered)
  2. **Source-aware cross-encoder rerank**: Rerank within each pool, then interleave (3 study + 2 caselaw) to preserve source diversity
  - For small corpora (<5K), falls back to simple retrieval + rerank
- Multi-query retrieval in `retrieve_documents_multi_query(queries, k=5)`:
  - Pools bi-encoder candidates from multiple query variants (primary + alternatives) across both source pools
  - Deduplicates by `idx`, then cross-encoder reranks the full pool against the primary query
  - Source-aware interleave preserves diversity; bridges terminological gaps (e.g., "cancellation clause" also retrieves "illusory promise")
- `compute_confidence(query, docs)` returns mean cosine similarity between query and doc embeddings
- `load_passages_to_chroma(csv_path, max_passages=0)` loads passages with batch progress reporting
- Singletons: `get_vectorstore()`, `get_memory_store()`, `get_embeddings()`, `get_cross_encoder()` all cached
- QA Memory (separate `qa_memory` collection, cosine distance, same ChromaDB persist dir):
  - `check_memory(query, threshold=0.92)` returns `{"found": bool, "answer": str, "confidence": float, "question": str}`
  - `write_to_memory(question, answer, confidence)` stores QA pair with timestamp metadata

### Evaluation

- `eval_comprehensive.py` — Two-phase eval: Phase 1 retrieval-only (953 QA pairs), Phase 2 full pipeline (26 diverse queries with grading)
- `eval_reranker.py` — A/B comparison of bi-encoder-only vs cross-encoder reranking on stratified 96-query sample
- `eval_musique.py` — MuSiQue multi-hop QA eval: Phase 1 retrieval (Recall@5, MRR by hop count), Phase 2 full pipeline (exact match, F1 token overlap)
- `load_corpus.py` — Load full 220K passage corpus: `uv run python load_corpus.py [count|status]`
- `load_corpus_musique.py` — Load MuSiQue paragraphs from HuggingFace into `musique_passages` collection

## Data Directories (gitignored)

- `datasets/barexam_qa/` — passage CSVs and QA dataset CSVs
- `chroma_db/` — persisted ChromaDB vector store (220K passages, ~1.4GB)

## Key Dependencies

`langgraph`, `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-openai`, `chromadb`, `pandas`, `pydantic`, `tqdm`, `numpy`, `python-dotenv`, `sentence-transformers`

## WSL Setup

```bash
# Clone + install
git clone <repo-url> && cd LegalRagAgent
uv sync

# Copy .env (same API keys work on both machines)
cp .env.example .env  # then fill in API keys

# Rebuild ChromaDB (not committed to git — ~20min for 220K passages)
uv run python load_corpus.py

# GPU note: sentence-transformers will auto-detect CUDA for embeddings/reranker.
# No code changes needed — just ensure torch has CUDA support:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```
