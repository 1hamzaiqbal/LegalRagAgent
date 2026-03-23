# CLAUDE.md

Source-of-truth context for working in this codebase. The runtime lives in the `legal_rag/` package; `main.py` is a thin CLI wrapper.

## Project Summary

Legal RAG agent over the `reglab/barexam_qa` and `reglab/housing_qa` corpora. Built on LangGraph with a parallel plan-and-execute architecture: decompose a legal question into independent sub-questions, execute them with per-step escalation, synthesize a cited IRAC answer, and loop back if evidence is incomplete.

## Runtime Architecture

Source of truth: `legal_rag/` package

| Module | Responsibility |
|---|---|
| `legal_rag/runtime.py` | Graph building, `run_experiment()` entry point |
| `legal_rag/nodes.py` | Graph node implementations (router, planner, synthesizer, replanner) |
| `legal_rag/step_executor.py` | Round-safe parallel execution + per-step escalation |
| `legal_rag/retrieval.py` | RAG/web/direct action execution + query rewriting |
| `legal_rag/profiles.py` | 6 experiment profile definitions |
| `legal_rag/prompts.py` | Inline prompts (router, MC adjudicator, completeness check) |
| `legal_rag/models.py` | Core types: `PlanningStep`, `LegalAgentState`, `ExperimentProfile` |
| `legal_rag/core.py` | LLM call wrapper, JSON parsing, skill loading, metrics |
| `legal_rag/baselines.py` | `llm_only` and `rag_baseline` runners |
| `legal_rag/artifacts.py` | Run artifact JSON persistence |
| `main.py` | Thin CLI wrapper — delegates to `legal_rag/runtime.py` |

### Graph

```
START → llm_snap_node → router_node → planner_node → execute_round_node → synthesizer_node
                                           ↑                                      |
                                           └─── replanner_node ←──────────────────┘ (if incomplete)
                                                                                  └→ END (if complete)
```

### Nodes

**llm_snap_node** (`nodes.py`) — Quick LLM-only answer (1 call, no retrieval) before the pipeline runs. Stores `llm_snap_answer` and `llm_snap_letter` in state. Used by the synthesizer's arbitration step to catch cases where the pipeline researches the wrong theory.

**router_node** (`nodes.py`) — Lightweight LLM call to choose which ChromaDB collection(s) to search. Current registry: `legal_passages`, `housing_statutes`. Falls back to `legal_passages`.

**planner_node** (`nodes.py:181`) — Decomposes the question into `PlanningStep`s. Outputs:
- `complexity`: `"simple"` / `"moderate"` / `"complex"` (LLM decides)
- Steps: each with `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `max_retries`
- Hard cap: 5 steps max. `max_retries` capped at 3.

Loads `skills/planner.md`. Falls back to a single-step plan on parse failure.

**execute_round_node** (`step_executor.py:323`) — Executes ALL pending steps in parallel via `ThreadPoolExecutor`, each with its own internal escalation chain via `_execute_step_with_escalation()`:

Per-step execution:
- `rag_search`: LLM query rewrite → multi-query retrieval → cross-encoder rerank → cited synthesis → judge
- `web_search`: DuckDuckGo → scrape top 2 URLs (trafilatura) → cited synthesis → judge
- `direct_answer`: LLM answers from doctrine (no retrieval) → verifier

Per-step escalation (if judge says insufficient, up to `max_retries`):
- `rag_search` → rewrite query → `direct_answer` (web skipped for doctrinal queries)
- `web_search` → `direct_answer`

Round-safe: takes snapshots of `planning_table` and `evidence_store` before parallel execution. Results merged via `_merge_round_results()` with evidence deduplication.

**synthesizer_node** (`nodes.py`) — Aggregates all completed steps into an IRAC answer (`skills/synthesizer.md`). For MC questions: (1) runs MC adjudicator to force an explicit letter choice, (2) if the pipeline's letter differs from the LLM snap, runs an **arbitration** step that compares both analyses and picks the better-supported answer. Then runs a completeness check. If gaps identified, returns `missing_topics` and routes back to replanner. Max 3 rounds.

**replanner_node** (`nodes.py:384`) — Creates new `PlanningStep`s from the synthesizer's `missing_topics` and feeds them back to the executor.

### Shared State (`LegalAgentState`)

- `agent_metadata` — provider, model, timestamps
- `inputs` — `{"question": "..."}`
- `run_config` — `{"max_steps": 7, "max_parallel_rounds": 3}`
- `collections` — chosen search collection(s), populated by `router_node`
- `planning_table` — list of `PlanningStep`
- `evidence_store` — accumulated retrieved passages (all steps)
- `final_answer` — synthesizer output
- `audit_log` — per-node trace entries with timestamps
- `completeness_verdict` — synthesizer's completeness check result
- `parallel_round` — current planner→executor→synthesizer iteration

`PlanningStep` fields: `step_id`, `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `max_retries`, `rewrite_attempt`, `status`, `result`, `confidence`, `evidence_ids`, `retry_of`, `judge_verdict`.

### Logging

Two modes controlled by `--verbose` CLI flag or `VERBOSE=1` env var:
- **Compact** (default): step breakdown with evidence source counts, LLM call/token totals
- **Verbose**: full passage text with cross-encoder scores, query rewrite alternatives, web search URLs and scraped content previews, sub-answer previews, per-LLM-call token counts

## Skills

7 prompt files in `skills/`, loaded by `legal_rag/core.py`:

| Skill file | Loaded as | Purpose |
|---|---|---|
| `planner.md` | `planner` | Decompose question into research steps with complexity + max_retries |
| `query_rewriter.md` | `query_rewriter` | Rewrite sub-question into primary + 2 alternative queries (JSON) |
| `aspect_query_rewriter.md` | `aspect_query_rewriter` | Rewrite into rule/exception/application queries (aspect mode only) |
| `synthesize_and_cite.md` | `synthesize_and_cite` | Per-step cited synthesis with `[Source N]` format |
| `judge.md` | `judge` | Evaluate retrieval sufficiency (full/partial/insufficient) |
| `verifier.md` | `verifier` | Evaluate direct_answer grounding in established doctrine |
| `synthesizer.md` | `synthesizer` | Final IRAC synthesis with `[Evidence N]` citations |

## Retrieval Stack

Source of truth: `rag_utils.py`

- **Vector store**: ChromaDB persisted to `./chroma_db/`
  - `legal_passages`: 686,324 barexam passages
  - `housing_statutes`: 1,837,403 housing statutes
- **Embedding model**: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 8192 tokens)
- **Cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Hybrid retrieval**: BM25 + bi-encoder candidates pooled, deduplicated by `idx`, cross-encoder reranks to top k
- **BM25 corpus cap**: BM25 skipped for collections >1M docs (housing_statutes uses dense-only)
- **Multi-query retrieval**: pools candidates across all query variants, deduplicates, reranks against primary query
- **Cross-step dedup**: `exclude_ids` filters out passages already retrieved in prior steps

## LLM Configuration

Source of truth: `llm_config.py`

- Provider selection via `LLM_PROVIDER` env var (default: `deepseek`)
- `get_llm()` returns a cached `ChatOpenAI` instance (LRU cache keyed on temperature + provider)
- `_llm_call()` adds retry handling (3 attempts for transient errors)
- Run `uv run python llm_config.py` to list all providers

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

# Run agent
uv run python main.py simple                        # Simple doctrinal question
uv run python main.py simple --verbose              # Verbose output
uv run python main.py --profile rewrite_rag simple  # Use a specific profile
uv run python main.py --question "literal question" # Custom question
uv run python main.py --list-profiles               # Show available profiles

# Evals
uv run python eval/eval_baseline.py 100             # Direct-LLM baseline
uv run python eval/eval_bm25_baseline.py 100        # Simple retrieve-and-answer baseline
uv run python eval/eval_golden.py 100               # Golden-passage upper bound
uv run python eval/eval_rag_rewrite.py 100          # RAG + query rewrite
uv run python eval/eval_qa.py 100                   # Full pipeline eval (default: full_parallel)
uv run python eval/eval_qa.py 100 --profile full_parallel_aspect  # Specific profile

# List providers
uv run python llm_config.py
```

## Results (N=100, BarExam QA, DeepSeek)

| Method | Profile | Accuracy | Gold Recall@5 |
|---|---|---|---|
| LLM-only (no RAG) | `llm_only` | 85% | n/a |
| RAG + query rewrite | `rewrite_rag` | 80% | 8% |
| Golden passage (upper bound) | — | 77% | n/a |
| Full parallel pipeline | `full_parallel` | 76% | 15% |
| Simple RAG (raw question) | `simple_rag` | 70% | 0% |

Baselines run 2026-03-22. Full parallel run 2026-03-23. See `logs/` for detailed per-question results.

## Eval Scripts

| Script | Notes |
|---|---|
| `eval/eval_baseline.py` | Direct-LLM baseline (no RAG) |
| `eval/eval_bm25_baseline.py` | Simple retrieve-and-answer baseline |
| `eval/eval_golden.py` | Golden-passage upper bound |
| `eval/eval_rag_rewrite.py` | RAG with query rewriting |
| `eval/eval_qa.py` | Full pipeline eval |
| `eval/eval_query_strategies.py` | Compare query rewriting strategies |
| `eval/eval_retrieval_recall.py` | Gold passage retrieval recall |

## Datasets

| Dataset | Collection | Docs | QA format | Source |
|---|---|---|---|---|
| BarExam QA | `legal_passages` | 686,324 | MC (A-D) | `reglab/barexam_qa` |
| HousingQA | `housing_statutes` | 1,837,403 | Yes/No | `reglab/housing_qa` |

## Data (gitignored)

- `datasets/barexam_qa/` — Passage CSVs and QA splits
- `datasets/housing_qa/` — Statute CSVs and QA pairs
- `chroma_db/` — Persisted ChromaDB vector store
- `logs/` — Eval output logs

## Editing Guidance

- The `legal_rag/` package is the source of truth for the pipeline. `main.py` is a thin CLI wrapper.
- If you change step schema or routing, audit `legal_rag/nodes.py`, `legal_rag/step_executor.py`, and the skill prompt contracts in `skills/`.
- `_get_metrics()`, `_reset_llm_call_counter()`, `_get_deepseek_balance()` are defined in `legal_rag/core.py` and re-exported via `main.py` for eval scripts.
- `web_scraper.py` is a standalone module (testable via CLI) imported by `legal_rag/retrieval.py` for web_search steps.
- `utils/fast_embed.py` bypasses LangChain for bulk embedding — uses sentence-transformers with fp16 + chunked processing. Supports `--resume`.
- Sequential pipeline code archived in branch `archive/sequential-pipeline`.
