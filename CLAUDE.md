# CLAUDE.md

Source-of-truth context for working in this codebase. Verify claims against `main.py` before relying on them.

## Project Summary

Legal RAG agent over the `reglab/barexam_qa` and `reglab/housing_qa` corpora. Built on LangGraph with a plan-and-execute workflow: decompose a legal question into sub-questions, retrieve passages from ChromaDB, synthesize cited sub-answers, and aggregate into a final IRAC-style response.

## Runtime Architecture

Source of truth: `main.py`

### Graph (5 nodes)

```
START → router_node → planner_node → executor_node → replanner_node ─┬─→ executor_node  (next/retry)
                                                                      └─→ synthesizer_node → END  (complete)
```

### Nodes

**router_node** — Chooses which ChromaDB collection(s) to search. Current registry:
- `legal_passages`
- `housing_statutes`

Falls back to `legal_passages` on parse failure or uncertainty.

**planner_node** — Decomposes the question into 2-5 ordered `PlanningStep`s. Each step has a `sub_question`, `authority_target`, `retrieval_hints`, and `action_type` (`rag_search` | `web_search` | `direct_answer`). Loads `skills/planner.md`. Falls back to a single-step plan on JSON parse failure.

**executor_node** — Executes the first pending step:
- `rag_search`: LLM query rewrite (`skills/query_rewriter.md`) → multi-query ChromaDB retrieval with cross-step dedup → cited synthesis (`skills/synthesize_and_cite.md`)
- `web_search`: DuckDuckGo search → scrape top 2 URLs via `web_scraper.py` (trafilatura) → cited synthesis from snippets + full page text
- `direct_answer`: LLM answers from doctrine (no retrieval) → synthesis

After execution, an LLM judge evaluates retrieval sufficiency (`skills/judge.md` for retrieval steps, `skills/verifier.md` for direct_answer). The judge verdict drives replanner escalation. Confidence (cross-encoder logit, sigmoid-normalized) is logged but not used for control flow.

**replanner_node** — Two-phase decision:
1. *Deterministic escalation* (when judge says insufficient):
   - `rag_search` (attempt 0) → rewrite query, stay `rag_search`
   - `rag_search` (attempt 1+) → escalate to `web_search`
   - `web_search` → escalate to `direct_answer`
2. *LLM decision* (when judge says sufficient, or after direct_answer): `skills/replanner.md` decides `next` / `complete` / `retry`. Retry is honored once per step; further retries auto-advance.

Step budget: `max_steps` defaults to 7 original planned steps. Retry / escalation steps (`retry_of != None`) do not count toward the budget.

**synthesizer_node** — Aggregates all completed step results and the full evidence store into a final IRAC answer with `[Evidence N]` citations. Loads `skills/synthesizer.md`.

### Shared State (`LegalAgentState`)

- `agent_metadata` — provider, model, timestamps
- `inputs` — `{"question": "..."}`
- `run_config` — `{"max_steps": 7}`
- `collections` — chosen search collection(s), populated by `router_node`
- `planning_table` — list of `PlanningStep`
- `evidence_store` — accumulated retrieved passages (all steps)
- `final_answer` — synthesizer output
- `audit_log` — per-node trace entries with timestamps

`PlanningStep` fields: `step_id`, `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `rewrite_attempt`, `status`, `result`, `confidence`, `evidence_ids`, `retry_of`, `judge_verdict`.

### Logging

Two modes controlled by `--verbose` CLI flag or `VERBOSE=1` env var:
- **Compact** (default): step breakdown with evidence source counts, LLM call/token totals
- **Verbose**: full passage text with cross-encoder scores, query rewrite alternatives, web search URLs and scraped content previews, sub-answer previews, per-LLM-call token counts

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

- **Vector store**: ChromaDB persisted to `./chroma_db/`
  - `legal_passages`: 686,324 barexam passages
  - `housing_statutes`: 1,837,403 housing statutes
- **Embedding model**: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 8192 tokens). Configurable via `EMBEDDING_MODEL` env var.
- **Cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Hybrid retrieval** (default): BM25 (keyword) + bi-encoder (semantic) candidates pooled, deduplicated by `idx`, cross-encoder reranks to top k
- **BM25 index**: Built lazily on first retrieval from ChromaDB corpus, cached in memory. Uses `rank-bm25` (BM25Okapi).
- **BM25 corpus cap**: BM25 is skipped for collections larger than 1,000,000 docs, so `housing_statutes` currently uses dense-only fallback under the present code.
- **Multi-query retrieval**: pools BM25 + dense candidates across all query variants, deduplicates, cross-encoder reranks against primary query
- **Cross-step dedup**: `exclude_ids` parameter filters out passages already retrieved in prior steps
- **Confidence**: `compute_confidence()` returns max cross-encoder score (raw logit). Converted to [0,1] via sigmoid in main.py. For logging only.

## LLM Configuration

Source of truth: `llm_config.py`

- Provider selection via `LLM_PROVIDER` env var. Falls back to `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`.
- `get_llm()` returns a cached `ChatOpenAI` instance (LRU cache keyed on temperature + provider).
- `_llm_call()` in main.py adds retry handling (3 attempts for transient 429/connection/timeout errors) and merges system+user messages for Gemma models.
- Run `uv run python llm_config.py` to list all providers with rate limits.

Providers: DeepSeek, Google AI Studio (Gemma, Gemini Flash), Groq (Llama, Maverick, Scout, Qwen, etc.), OpenRouter, Cerebras, Ollama.

## Commands

```bash
# Install
uv sync

# Configure
cp .env.example .env   # then add API keys

# Download datasets
uv run python utils/download_data.py               # BarExam QA
uv run python utils/download_housingqa.py           # HousingQA

# Build vector stores (GPU-optimized)
uv run python utils/fast_embed.py barexam           # Full barexam (~2.2 hr on RTX 3070)
uv run python utils/fast_embed.py housing           # Full housing (~6 hr on RTX 3070)
uv run python utils/fast_embed.py housing --resume  # Resume interrupted embedding
uv run python utils/fast_embed.py status            # Check collection sizes

# Or use LangChain loader for smaller subsets
uv run python utils/load_corpus.py curated          # Gold passages + padding (~3 min)
uv run python utils/load_corpus.py 20000            # First 20K passages (~30 min)
uv run python utils/load_corpus.py status           # Check sizes

# Run agent
uv run python main.py simple                        # Simple doctrinal question
uv run python main.py multi_hop                      # Multi-step reasoning
uv run python main.py medium                         # Medium complexity
uv run python main.py simple --verbose               # Verbose output

# Current presentation-facing evals
uv run python eval/eval_baseline.py 100             # Direct-LLM baseline
uv run python eval/eval_bm25_baseline.py 100        # Simple retrieve-and-answer baseline
uv run python eval/eval_golden.py 100               # Golden-passage upper bound

# List providers
uv run python llm_config.py
```

## Current Reported Results

- Direct LLM baseline: `85/100` in `logs/eval_baseline_deepseek_20260322_13.txt`
- Simple retrieve-and-answer baseline: `70/100` in `logs/eval_bm25_baseline_deepseek_20260322_15.txt`
- Golden-passage upper bound: `77/100` in `logs/eval_golden_deepseek_20260322_13.txt`

## Eval Scripts

Presentation-facing current evals:

| Script | Notes |
|---|---|
| `eval/eval_baseline.py` | Direct-LLM baseline (no RAG, no LangGraph) |
| `eval/eval_bm25_baseline.py` | Simple retrieve-and-answer baseline using the current retriever |
| `eval/eval_golden.py` | Golden-passage upper bound |

Other scripts still present under `eval/` are exploratory or older and should not be treated as the current reported results unless rerun.

## Datasets

| Dataset | Collection | Docs | QA format | Source |
|---|---|---|---|---|
| BarExam QA | `legal_passages` | 686,324 | MC (A-D) | `reglab/barexam_qa` |
| HousingQA | `housing_statutes` | 1,837,403 | Yes/No | `reglab/housing_qa` |

HousingQA has per-state jurisdiction scoping — each question targets a specific state's statutes. The `state` field is stored in ChromaDB metadata and is available for filtered retrieval, but the current runtime does not yet apply automatic state filtering.

## Data (gitignored)

- `datasets/barexam_qa/` — Passage CSVs and QA splits
- `datasets/housing_qa/` — Statute CSVs and QA pairs
- `chroma_db/` — Persisted ChromaDB vector store
- `logs/` — Eval output logs
- `case_studies/` — JSON traces (if generated)

## Editing Guidance

- `main.py` is the source of truth for the pipeline. Verify architecture claims here before updating docs or skills.
- If you change step schema or routing, audit both `main.py` and the skill prompt contracts in `skills/`.
- `_get_metrics()`, `_reset_llm_call_counter()`, `_get_deepseek_balance()` are defined in main.py and exported to eval scripts.
- `web_scraper.py` is a standalone module (testable via CLI) imported by main.py for web_search steps.
- `utils/fast_embed.py` bypasses LangChain for bulk embedding — uses sentence-transformers directly with fp16 + chunked processing to avoid OOM. Supports `--resume` for interrupted runs.
