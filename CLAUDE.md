# CLAUDE.md

Source-of-truth context for working in this codebase. Verify claims against `main.py` before relying on them.

## Environment Note

`uv` may not be on PATH in every shell. Prefer `uv` when available, otherwise fall back to `~/.local/bin/uv`.

## Project Summary

Legal RAG research repo with two distinct surfaces:
- `main.py` = the full LangGraph agentic pipeline / demo system
- `eval/` = the current research harness, where adaptive retrieval variants are compared under a fixed evaluation setup

Current research direction: the original heavy pipeline underperformed, but the long-term goal is still a strong full agentic system. For now, the project rebuilds toward that goal **atomically**: simpler adaptive retrieval strategies are the default baseline, and extra structure only stays when it proves itself in `eval/eval_harness.py`.

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

## Current Best Results / Direction Snapshot

- **BarExam (Llama 70B, N=200):** `ce_threshold` = **80.0%**, `snap_hyde` = **76.5%**
- **BarExam (Gemma 4 E4B, N=200):** `snap_hyde` = **65.5%**, `vectorless_direct` ~**63-65%** (in progress)
- **HousingQA:** `rag_snap_hyde` on Llama 70B = **56.0%**
- **CaseHOLD:** `llm_only` / `confidence_gated` on Llama 70B = **72.5%**
- **Best small model:** Gemma 4 E4B — 55.5% llm_only, 58.6% snap_hyde (N=1195)
- **Embedding comparison:** 7 embedders tested; cross-encoder reranking dominates (all converge to 65.0% with aligned reranking)

Working interpretation:
- snap reasoning is the biggest contributor (+5pp), more than retrieval itself
- HyDE passage generation adds +3.5pp retrieval quality + +3pp reranking quality
- gap architecture was broken by CE filter bug; rerunning with fix
- vectorless RAG (LLM generates knowledge, no vector store) is competitive with snap_hyde
- heavier architectural combinations have mostly underperformed simpler adaptive methods

Use `RESEARCH.md` for the current queue/handoff and `EXPERIMENTS.md` for the full tables + keep/discard history.

## Eval Scripts

| Script | Notes |
|---|---|
| `eval/eval_harness.py` | Unified multi-model harness (37 modes, 5 datasets) |
| `eval/eval_config.py` | Config, question loading, answer extraction, EVAL_MODES dict |
| `eval/eval_analyze.py` | Post-hoc analysis of JSONL logs |
| `eval/eval_qa.py` | Legacy full pipeline eval |

## Running Evals

### Environment requirements

- **HuggingFace offline mode**: HF Hub may be unreachable from this network. Always set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when running evals. The embedding model (`gte-large-en-v1.5`) is cached locally in `~/.cache/huggingface/hub/`.
- **uv**: Prefer `uv`; if it is missing from PATH in the current shell, use `~/.local/bin/uv`.
- **API keys**: All in `.env`. Groq, DeepSeek, Google, OpenRouter, OpenAI, Cerebras.

### Launch pattern (IMPORTANT for agents)

```bash
# Single run (recommended — monitor before scaling up)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode decompose_rag --provider groq-scout --questions 200 --dataset barexam

# Background run — do NOT pipe through grep/tail (eats errors and buffers output)
# Instead, run directly and redirect to a file:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup uv run python eval/eval_harness.py \
  --mode rag_snap_hyde --provider groq-llama70b --questions 200 --dataset barexam \
  > /tmp/eval_run.log 2>&1 &
```

**Common pitfalls:**
- Piping through `grep` or `tail` buffers stdout and hides errors — run failed silently
- Launching 6+ concurrent Groq calls hits rate limits (Llama: 1K RPD / 100K TPD, Scout: 1K RPD / 500K TPD)
- Parallel runs that share the same Groq model will contend on rate limits — run one per provider at a time
- Detail log and experiments.jsonl are written at END of run (not incrementally) — killing mid-run loses all results

### Groq rate limits

| Provider | Model | RPD | TPD |
|---|---|---|---|
| groq-llama70b | llama-3.3-70b-versatile | 1,000 | 100,000 |
| groq-scout | llama-4-scout-17b-16e-instruct | 1,000 | 500,000 |

Decompose_rag uses ~8 LLM calls per question (split + 3×snap + 3×hyde + synthesize). N=200 = ~1,600 calls. Fits within 1K RPD only because retry logic spaces them out — but running 2 evals on the same model will exceed limits.

### Monitoring a running eval

```bash
# Count completed questions in a background run's output
grep -c "PASS\|FAIL" /tmp/eval_run.log

# Current accuracy
echo "$(grep -c PASS /tmp/eval_run.log) / $(grep -c 'PASS\|FAIL' /tmp/eval_run.log)"

# Check for errors
grep -i "error\|traceback\|rate.limit" /tmp/eval_run.log | tail -5

# Watch for running eval processes
pgrep -a python | grep eval
```

### Output files

- **Detail log**: `logs/eval_{mode}_{provider}_{YYYYMMDD_HHMM}_detail.jsonl` — one JSON record per question
- **Summary**: appended to `logs/experiments.jsonl` — one JSON record per run
- Both written ONLY when run completes successfully

### Analyzing results

Use `eval/eval_analyze.py` for post-hoc analysis of JSONL detail logs.

```bash
# List all results for a mode
python3 -c "import json; [print(f\"{d['timestamp']} {d['mode']:25s} {d['provider']:20s} acc={d['accuracy']}  N={d['n_questions']}\") for d in (json.loads(l) for l in open('logs/experiments.jsonl')) if 'decompose' in d.get('mode','')]"
```

## Datasets

| Dataset | Collection | Docs | QA format | Source |
|---|---|---|---|---|
| BarExam QA | `legal_passages` | 686,324 | MC (A-D) | `reglab/barexam_qa` |
| HousingQA | `housing_statutes` | 1,837,403 | Yes/No | `reglab/housing_qa` |
| Legal-RAG-QA | `legal_rag_passages` | 190 | Open-ended | `isaacus/legal-rag-qa` |
| Australian Legal QA | `australian_legal` | 2,124 | Open-ended | `isaacus/open-australian-legal-qa` |
| CaseHOLD | `casehold_holdings` | 50,291 | MC (A-E) | `coastalcph/lex_glue` (case_hold) |

## Data (gitignored)

- `datasets/barexam_qa/` — Passage CSVs and QA splits
- `datasets/housing_qa/` — Statute CSVs and QA pairs
- `chroma_db/` — Persisted ChromaDB vector store
- `logs/` — Eval output logs

## Editing Guidance

- `main.py` is the source of truth for the pipeline (all runtime logic is currently here). Verify architecture claims against it before updating docs.
- If you change step schema or routing, audit both `main.py` and the skill prompt contracts in `skills/`.
- `web_scraper.py` is a standalone module (testable via CLI) imported by main.py for web_search steps.
- `utils/fast_embed.py` bypasses LangChain for bulk embedding — sentence-transformers with fp16 + chunked processing. Supports `--resume`.
- Verify the current working branch with `git branch --show-current` before relying on branch-specific notes; the repo is no longer guaranteed to be on `lightweight-rebuild`.
- Sequential pipeline code archived in branch `archive/sequential-pipeline`.
- See `RESEARCH.md` for current research state, experiment queue, and session handoff.
