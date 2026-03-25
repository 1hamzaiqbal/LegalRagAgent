# CLAUDE.md

Source-of-truth context for working in this codebase. Verify claims against `main.py` before relying on them.

## Environment Note

`uv` is at `~/.local/bin/uv` and is NOT on PATH in this shell. Always use `~/.local/bin/uv` (or set PATH) when running commands.

## Project Summary

Legal RAG agent over the `reglab/barexam_qa` and `reglab/housing_qa` corpora. Built on LangGraph with a parallel plan-and-execute architecture: decompose a legal question into independent sub-questions, execute them with per-step escalation, synthesize a cited IRAC answer, and loop back if evidence is incomplete.

## Runtime Architecture

Source of truth: `main.py`

### Graph

```
START → router_node → planner_node → parallel_executor_node → parallel_synthesizer_node
                          ↑                                          |
                          └── parallel_replanner_node ←──────────────┘ (if incomplete)
                                                                     └→ END (if complete)
```

### Nodes

**router_node** — Lightweight LLM call to choose which ChromaDB collection(s) to search. Current registry: `legal_passages`, `housing_statutes`. Falls back to `legal_passages`.

**planner_node** — Decomposes the question into `PlanningStep`s. Outputs:
- `complexity`: `"simple"` / `"moderate"` / `"complex"` (LLM decides)
- Steps: each with `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `max_retries`
- Hard cap: 5 steps max. `max_retries` capped at 3.

Loads `skills/planner.md`. Falls back to a single-step plan on parse failure.

**parallel_executor_node** — Executes ALL pending steps, each with its own internal escalation chain via `_execute_step_with_escalation()`:

Per-step execution:
- `rag_search`: LLM query rewrite → multi-query retrieval → cited synthesis → judge
- `web_search`: DuckDuckGo → scrape top 2 URLs (trafilatura) → cited synthesis → judge
- `direct_answer`: LLM answers from doctrine (no retrieval) → judge

Per-step escalation (if judge says insufficient, up to `max_retries`):
- `rag_search` → rewrite query → `direct_answer` (web skipped for doctrinal queries)
- `web_search` → `direct_answer`

**parallel_synthesizer_node** — Aggregates all completed steps into an IRAC answer (`skills/synthesizer.md`), then runs a completeness check. If gaps identified, returns `missing_topics` and routes back to replanner. Max 3 rounds.

**parallel_replanner_node** — Creates new `PlanningStep`s from the synthesizer's `missing_topics` and feeds them back to the executor.

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

4 prompt files in `skills/`, loaded by `main.py`:

| Skill file | Loaded as | Purpose |
|---|---|---|
| `planner.md` | `planner` | Decompose question into research steps |
| `query_rewriter.md` | `query_rewriter` | Rewrite sub-question into primary + 2 alternative queries (JSON) |
| `synthesize_and_cite.md` | `synthesize_and_cite` | Per-step cited synthesis with `[Source N]` format |
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
uv run python main.py multi_hop                      # Multi-step reasoning
uv run python main.py medium                         # Medium complexity
uv run python main.py simple --verbose               # Verbose output

# Evals (all via eval_harness.py)
uv run python eval/eval_harness.py --mode llm_only --provider groq-llama70b --questions 100
uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 100
uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 100 --dataset housing
uv run python eval/eval_harness.py --mode golden_passage --provider groq-llama70b --questions 100

# List providers
uv run python llm_config.py
```

## Current Best Results

### BarExam QA (Llama 70B, seed=42)

| Method | N=100 | N=200 | Notes |
|---|---|---|---|
| llm_only | 64% | 64%* | baseline |
| rag_snap_hyde | **82%** | **76.5%** | best approach; N=200 is reliable |
| rag_devil_hyde | 76% | — | counterevidence hurts |
| rag_top2_hyde | 79% | — | less focused = worse |
| golden_passage | 81% | — | oracle upper bound |

*llm_only is stable across N=100/200 for this model.

### HousingQA (Llama 70B, seed=42)

| Method | N=100 | N=200 | Notes |
|---|---|---|---|
| llm_only | 44% | 47% | massive Yes-bias |
| rag_snap_hyde | **61%** | **56%** | largest validated RAG lift (+9) |

See EXPERIMENTS.md for full history, cross-model results, and per-subject breakdowns.

## Eval Scripts

| Script | Notes |
|---|---|
| `eval/eval_harness.py` | Unified multi-model harness (14 modes, 2 datasets) |
| `eval/eval_config.py` | Config, question loading, answer extraction |
| `eval/eval_analyze.py` | Post-hoc analysis of JSONL logs |
| `eval/eval_qa.py` | Legacy full pipeline eval |

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

- `main.py` is the source of truth for the pipeline. Verify architecture claims here before updating docs or skills.
- If you change step schema or routing, audit both `main.py` and the skill prompt contracts in `skills/`.
- `_get_metrics()`, `_reset_llm_call_counter()`, `_get_deepseek_balance()` are defined in main.py and exported to eval scripts.
- `web_scraper.py` is a standalone module (testable via CLI) imported by main.py for web_search steps.
- `utils/fast_embed.py` bypasses LangChain for bulk embedding — uses sentence-transformers with fp16 + chunked processing. Supports `--resume`.
- Sequential pipeline code archived in branch `archive/sequential-pipeline`.
