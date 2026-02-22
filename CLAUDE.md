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

Five-node state machine with adaptive replanning:

1. **classifier_node** — Classifies objective as `simple` or `multi_hop` using `classify_and_route.md` skill
2. **planner_node** — Generates initial plan. For `multi_hop`, emits only the first step; replanner handles the rest adaptively.
3. **executor_node** — For each pending step: rewrites query (`query_rewrite.md`), retrieves from ChromaDB, synthesizes answer (`synthesize_answer.md`), grounds with citations (`ground_and_cite.md`), computes confidence via cosine similarity
4. **evaluator_node** — Marks steps completed (confidence >= 0.7) or failed (< 0.7). Accumulates step summaries into `accumulated_context`. Increments `iteration_count`.
5. **replanner_node** — (multi_hop only) Receives objective + accumulated evidence, decides: `next_step` (add new research step), `retry` (rephrase failed step), or `complete` (aggregate final answer).

Routing:
- After evaluator: 3-way — `executor` (pending steps) | `replanner` (multi_hop, all done) | `END` (simple done, or iteration limit >6)
- After replanner: 2-way — `executor` (new step added) | `END` (complete)

Graph: `classifier → planner → executor → evaluator → {executor | replanner | END}`; `replanner → {executor | END}`

### Shared State (`AgentState`)

TypedDict with `global_objective`, `planning_table` (list of `PlanStep`), `contingency_plan`, `query_type` ("simple"/"multi_hop"), `final_cited_answer`, `accumulated_context` (step summaries for replanner), and `iteration_count` (loop guard, max 6).

### Skill System (skills/)

6 markdown prompt files cached at first load via `@lru_cache` in `load_skill_instructions()`:
- `classify_and_route.md` — classify query complexity
- `plan_synthesis.md` — generate research plan
- `query_rewrite.md` — optimize retrieval queries
- `synthesize_answer.md` — synthesize grounded answers
- `ground_and_cite.md` — verify grounding and add citations
- `adaptive_replan.md` — decide next research step based on accumulated evidence

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

### Evaluation (eval.py)

Measures retrieval quality on `barexam_qa/qa/barexam_qa_validation.csv` (first 200 queries). Metrics: Recall@K and MRR. Expects columns `question` and `gold_idx`.

## Data Directories (gitignored)

- `barexam_qa/passages/` — legal passage CSVs
- `barexam_qa/qa/` — QA dataset CSVs
- `chroma_db/` — persisted ChromaDB vector store

## Key Dependencies

`langgraph`, `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-openai`, `chromadb`, `pandas`, `pydantic`, `tqdm`, `numpy`, `python-dotenv`
