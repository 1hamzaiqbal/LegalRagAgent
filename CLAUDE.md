# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Legal RAG system built on LangGraph. It uses a classify-plan-execute-evaluate loop to answer legal research questions by retrieving passages from a ChromaDB vector store of bar exam materials. LLM calls are made via `langchain-openai`'s `ChatOpenAI`, compatible with any OpenAI-compatible API (Groq, Ollama, OpenAI, etc.).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent with a demo query
python main.py simple       # Single-concept question (negligence elements)
python main.py multi_hop    # Multi-concept constitutional rights scenario
python main.py medium       # Preliminary injunction standard

# Run retrieval evaluation (Recall@5, MRR) on bar exam QA dataset
python eval.py

# Verify LLM config
python -c "from llm_config import get_llm; print(get_llm())"
```

## Environment Setup

Copy `.env.example` to `.env` and set your API key. Default uses Groq free tier (no credit card needed):

```bash
cp .env.example .env
# Edit .env with your Groq API key from https://console.groq.com
```

## Architecture

### LangGraph Workflow (main.py)

Nine-node state machine with adaptive replanning, injection detection, answer verification, QA memory, and observability:

1. **detect_injection_node** — Screens input for adversarial prompt injection using `detect_prompt_injection.md` skill. Unsafe inputs are rejected and routed to observability for metrics.
2. **classifier_node** — Classifies objective as `simple` or `multi_hop` using `classify_and_route.md` skill
3. **planner_node** — Checks QA memory first (cosine similarity >= 0.92); on cache hit, short-circuits to memory writeback. Otherwise generates initial plan. For `multi_hop`, emits only the first step; replanner handles the rest adaptively.
4. **executor_node** — For each pending step: rewrites query (`query_rewrite.md`), retrieves from ChromaDB, synthesizes answer (`synthesize_answer.md`), grounds with `[Source N]` citations (`ground_and_cite.md`), computes confidence via cosine similarity
5. **evaluator_node** — Marks steps completed (confidence >= 0.7) or failed (< 0.7). Accumulates step summaries into `accumulated_context`. Sets explicit failure message if all steps fail.
6. **replanner_node** — (multi_hop only) Receives objective + accumulated evidence, decides: `next_step` (add new research step), `retry` (rephrase failed step), or `complete` (aggregate final answer).
7. **verify_answer_node** — Cross-checks final answer against evidence using `verify_answer.md` skill. On first failure, adds a corrective step using the LLM's `suggested_query` (a proper legal question). Second failure terminates cleanly without orphaned steps.
8. **memory_writeback_node** — Persists successful QA pairs (avg confidence >= 0.7) to ChromaDB `qa_memory` collection (cosine distance) for future cache hits.
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

TypedDict with `global_objective`, `planning_table` (list of `PlanStep`), `contingency_plan`, `query_type` ("simple"/"multi_hop"), `final_cited_answer`, `accumulated_context` (step summaries for replanner), `iteration_count` (loop guard, max 6), `injection_check` (safety result), `verification_result` (answer verification with `suggested_query`), `verification_retries` (1 corrective retry max), `memory_hit` (QA cache result, threshold 0.92), and `run_metrics` (observability data including parse failures and has_answer).

### Skill System (skills/)

8 markdown prompt files cached at first load via `@lru_cache` in `load_skill_instructions()`:
- `classify_and_route.md` — classify query complexity
- `plan_synthesis.md` — generate research plan
- `query_rewrite.md` — optimize retrieval queries
- `synthesize_answer.md` — synthesize grounded answers
- `ground_and_cite.md` — verify grounding and add citations
- `adaptive_replan.md` — decide next research step based on accumulated evidence
- `detect_prompt_injection.md` — screen input for adversarial prompts (fail-open)
- `verify_answer.md` — cross-check answer against evidence for unsupported claims, contradictions, and missing info

### External Tool Placeholders (external_tools.py)

`@tool`-decorated stubs for teammate's Playwright-based web lookup API:
- `web_search(query)` — web search placeholder
- `web_scrape(url)` — page scraping placeholder
- `external_api_call(endpoint, payload)` — generic API wrapper placeholder
- Configured via `EXTERNAL_TOOLS_BASE_URL`, `EXTERNAL_TOOLS_API_KEY` env vars
- `get_external_tools()` returns all tools for `llm.bind_tools()` binding

### LLM Config (llm_config.py)

- Single `get_llm()` function returning a cached `ChatOpenAI` singleton (`@lru_cache`)
- Configured via env vars: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- Default: Groq free tier with `llama-3.3-70b-versatile`
- Supports DeepSeek (`deepseek-chat`) and vLLM with automatic prefix-cache hit logging

### RAG / Retrieval (rag_utils.py)

- Embeddings: HuggingFace `all-MiniLM-L6-v2`
- Vector store: ChromaDB persisted to `./chroma_db/`
- `load_passages_to_chroma()` loads first 1000 passages from `barexam_qa/passages/` CSVs
- `retrieve_documents(query, k=5)` returns top-k `Document` objects
- `compute_confidence(query, docs)` returns mean cosine similarity between query and doc embeddings
- Singletons: `get_vectorstore()` and `get_memory_store()` return cached instances; `get_embeddings()` is `@lru_cache`
- QA Memory (separate `qa_memory` collection, cosine distance, same ChromaDB persist dir):
  - `check_memory(query, threshold=0.92)` returns `{"found": bool, "answer": str, "confidence": float, "question": str}`
  - `write_to_memory(question, answer, confidence)` stores QA pair with timestamp metadata

### Evaluation (eval.py)

Measures retrieval quality on `barexam_qa/qa/barexam_qa_validation.csv` (first 200 queries). Metrics: Recall@K and MRR. Expects columns `question` and `gold_idx`.

## Data Directories (gitignored)

- `barexam_qa/passages/` — legal passage CSVs
- `barexam_qa/qa/` — QA dataset CSVs
- `chroma_db/` — persisted ChromaDB vector store

## Key Dependencies

`langgraph`, `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-openai`, `chromadb`, `pandas`, `pydantic`, `tqdm`, `numpy`, `python-dotenv`
