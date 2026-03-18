# CLAUDE.md

Source-of-truth context for working in this codebase. Verify claims against `main.py` before relying on them.

## Project Summary

Legal RAG agent over the `reglab/barexam_qa` corpus. Built on LangGraph with a plan-and-execute workflow: decompose a legal question into sub-questions, retrieve passages from ChromaDB, synthesize cited sub-answers, and aggregate into a final IRAC-style response.

## Runtime Architecture

Source of truth: `main.py`

### Graph (4 nodes)

```
START → planner_node → executor_node → replanner_node ─┬─→ executor_node  (next/retry)
                                                        └─→ synthesizer_node → END  (complete)
```

### Nodes

**planner_node** — Decomposes the question into 2-5 ordered `PlanningStep`s. Each step has a `sub_question`, `authority_target`, `retrieval_hints`, and `action_type` (`rag_search` | `web_search` | `direct_answer`). Loads `skills/planner.md`. Falls back to a single-step plan on JSON parse failure.

**executor_node** — Executes the first pending step:
- `rag_search`: LLM query rewrite (`skills/query_rewriter.md`) → multi-query ChromaDB retrieval with cross-step dedup → cited synthesis (`skills/synthesize_and_cite.md`)
- `web_search`: DuckDuckGo search → cited synthesis
- `direct_answer`: LLM answers from doctrine (no retrieval) → synthesis

After execution, an LLM judge evaluates retrieval sufficiency (`skills/judge.md` for retrieval steps, `skills/verifier.md` for direct_answer). The judge verdict drives replanner escalation. Confidence (cross-encoder logit, sigmoid-normalized) is logged but not used for control flow.

**replanner_node** — Two-phase decision:
1. *Deterministic escalation* (when judge says insufficient):
   - `rag_search` (attempt 0) → rewrite query, stay `rag_search`
   - `rag_search` (attempt 1+) → escalate to `web_search`
   - `web_search` → escalate to `direct_answer`
2. *LLM decision* (when judge says sufficient, or after direct_answer): `skills/replanner.md` decides `next` / `complete` / `retry`. Retry is honored once per step; further retries auto-advance.

Hard cap: `max_steps` (default 5) completed steps.

**synthesizer_node** — Aggregates all completed step results and the full evidence store into a final IRAC answer with `[Evidence N]` citations. Loads `skills/synthesizer.md`.

### Shared State (`LegalAgentState`)

- `agent_metadata` — provider, model, timestamps
- `inputs` — `{"question": "..."}`
- `run_config` — `{"max_steps": 5}`
- `planning_table` — list of `PlanningStep`
- `evidence_store` — accumulated retrieved passages (all steps)
- `final_answer` — synthesizer output
- `run_metrics` — LLM call counts, token usage
- `audit_log` — per-node trace entries with timestamps

`PlanningStep` fields: `step_id`, `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `rewrite_attempt`, `status`, `result`, `confidence`, `evidence_ids`, `retry_of`, `judge_verdict`.

## Skills

7 prompt files in `skills/`, all loaded by `main.py`:

| Skill file | Loaded as | Purpose |
|---|---|---|
| `planner.md` | `planner` | Decompose question into research steps with action types |
| `query_rewriter.md` | `query_rewriter` | Rewrite sub-question into primary + 2 alternative queries (JSON) |
| `synthesize_and_cite.md` | `synthesize_and_cite` | Per-step cited synthesis with `[Source N]` format |
| `judge.md` | `judge` | Evaluate retrieval sufficiency for rag_search/web_search steps |
| `verifier.md` | `verifier` | Evaluate direct_answer grounding in established doctrine |
| `replanner.md` | `replanner` | Decide next/complete/retry based on accumulated evidence |
| `synthesizer.md` | `synthesizer` | Final IRAC synthesis with `[Evidence N]` citations |

## Retrieval Stack

Source of truth: `rag_utils.py`

- **Vector store**: ChromaDB persisted to `./chroma_db/`, collection `legal_passages`
- **Embedding model**: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 8192 tokens). Configurable via `EMBEDDING_MODEL` env var.
- **Cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Hybrid retrieval** (default): BM25 (keyword) + bi-encoder (semantic) candidates pooled, deduplicated by `idx`, cross-encoder reranks to top k. BM25 catches exact legal terms embeddings miss; bi-encoder catches semantic matches BM25 misses.
- **BM25 index**: Built lazily on first retrieval from ChromaDB corpus (~8s for 20K passages), cached in memory. Uses `rank-bm25` (BM25Okapi).
- **Multi-query retrieval**: pools BM25 + dense candidates across all query variants, deduplicates, cross-encoder reranks against primary query
- **Cross-step dedup**: `exclude_ids` parameter filters out passages already retrieved in prior steps
- **Confidence**: `compute_confidence()` returns max cross-encoder score (raw logit). Converted to [0,1] via sigmoid in main.py. For logging only.

Caveat: `load_passages_to_chroma()` skips reload when existing count >= desired. Switching to a smaller corpus requires clearing `chroma_db/`.

## LLM Configuration

Source of truth: `llm_config.py`

- Provider selection via `LLM_PROVIDER` env var. Falls back to `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`.
- `get_llm()` returns a cached `ChatOpenAI` instance (LRU cache keyed on temperature only — does not invalidate on provider change mid-session).
- `_llm_call()` in main.py adds retry handling (3 attempts for transient 429/connection/timeout errors) and merges system+user messages for Gemma models.
- Run `uv run python llm_config.py` to list all providers with rate limits.

Providers: DeepSeek, Google AI Studio (Gemma, Gemini Flash), Groq (Llama, Maverick, Scout, Qwen, etc.), OpenRouter, Cerebras, Ollama.

## Commands

```bash
# Install
uv sync

# Configure
cp .env.example .env   # then add API keys

# Download dataset
uv run python utils/download_data.py
uv run python utils/download_data.py --check

# Build vector store
uv run python utils/load_corpus.py curated        # ~1.5K passages, ~3 min
uv run python utils/load_corpus.py 20000          # 20K passages, ~30 min
uv run python utils/load_corpus.py status         # Check current size

# Run agent
uv run python main.py simple       # "What are the elements of a negligence claim?"
uv run python main.py multi_hop    # Fourth Amendment suppression scenario
uv run python main.py medium       # Preliminary injunction standard

# Eval
uv run python eval/eval_qa.py 50                    # QA eval on 50 questions
uv run python eval/eval_qa.py 100 --continue         # Resume interrupted eval
uv run python eval/eval_baseline.py 50               # Direct-LLM baseline
uv run python eval/eval_reranker.py                   # Bi-encoder vs cross-encoder A/B

# List providers
uv run python llm_config.py
```

## Eval Scripts

| Script | Notes |
|---|---|
| `eval/eval_qa.py` | Main eval: N random QA pairs, MC accuracy, cost tracking |
| `eval/eval_baseline.py` | Direct-LLM baseline (no RAG, no LangGraph) |
| `eval/eval_reranker.py` | Retrieval-only A/B: bi-encoder vs cross-encoder |

## Data (gitignored)

- `datasets/barexam_qa/` — Passage CSVs and QA splits (from HuggingFace `reglab/barexam_qa`)
- `chroma_db/` — Persisted ChromaDB vector store
- `logs/` — Eval output logs
- `case_studies/` — JSON traces (if generated)

## Editing Guidance

- `main.py` is the source of truth for the pipeline. Verify architecture claims here before updating docs or skills.
- If you change step schema or routing, audit both `main.py` and the skill prompt contracts in `skills/`.
- `_get_metrics()`, `_reset_llm_call_counter()`, `_get_deepseek_balance()` are defined in main.py and exported to eval scripts.
