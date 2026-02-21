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

Four-node state machine with conditional looping:

1. **classifier_node** — Classifies objective as `simple` or `multi_hop` using `classify_and_route.md` skill
2. **planner_node** — Breaks the `global_objective` into `PlanStep` entries using `plan_synthesis.md` skill
3. **executor_node** — For each pending step: rewrites query (`query_rewrite.md`), retrieves from ChromaDB, synthesizes answer (`synthesize_answer.md`), grounds with citations (`ground_and_cite.md`), computes confidence via cosine similarity
4. **evaluator_node** — If confidence >= 0.7 marks step "completed"; if < 0.7 marks "failed" and injects a sub-step. When all done, aggregates `final_cited_answer`.

Routing: after evaluation, loops back to executor if pending steps remain; ends when all complete or >10 total steps (hard limit).

Graph: `classifier → planner → executor → evaluator → {executor | END}`

### Shared State (`AgentState`)

TypedDict with `global_objective`, `planning_table` (list of `PlanStep`), `contingency_plan`, `query_type` ("simple"/"multi_hop"), and `final_cited_answer`.

### Skill System (skills/)

5 markdown prompt files loaded at runtime by `load_skill_instructions()`:
- `classify_and_route.md` — classify query complexity
- `plan_synthesis.md` — generate research plan
- `query_rewrite.md` — optimize retrieval queries
- `synthesize_answer.md` — synthesize grounded answers
- `ground_and_cite.md` — verify grounding and add citations

### LLM Config (llm_config.py)

- Single `get_llm()` function returning a `ChatOpenAI` instance
- Configured via env vars: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- Default: Groq free tier with `llama-3.3-70b-versatile`

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
