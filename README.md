# LegalRagAgent

Legal RAG research project studying **when retrieval helps legal QA and when it hurts**.

The repo contains two layers:
- `main.py` — the full LangGraph agentic pipeline / demo system
- `eval/` — the current research loop, where simpler adaptive methods are benchmarked against heavier agentic variants

**Project direction:** the long-term goal is still a strong full agentic pipeline, but the current research program is rebuilding toward it atomically from smaller, controlled retrieval strategies and only keeping improvements that survive fixed-eval scrutiny.

**Current headline results:**
- BarExam best: `ce_threshold` on Llama 70B = **80.0%**
- HousingQA best: `rag_snap_hyde` on Llama 70B = **56.0%**
- CaseHOLD best: `llm_only` / `confidence_gated` = **72.5%**
- Current small-model audit: **5/7** full-set Phase 1 baselines complete; best completed baseline is **OR Qwen3-32B = 61.42%** (`734/1195`)

See `RESEARCH.md` for the current state + queue, and `EXPERIMENTS.md` for the full keep/discard history.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/shrango/adaptive-plan-and-solve-agent.git
cd adaptive-plan-and-solve-agent
uv sync
```

Requires Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).

### 2. Configure an LLM provider

```bash
cp .env.example .env
# Edit .env and add at least one provider API key
```

Default provider: `deepseek`. Run `uv run python llm_config.py` to list configured providers.

### 3. Download datasets

```bash
uv run python utils/download_data.py          # BarExam QA
uv run python utils/download_housingqa.py      # HousingQA
uv run python utils/download_new_datasets.py   # CaseHOLD, Legal-RAG-QA, Australian Legal QA
```

### 4. Build vector stores

```bash
uv run python utils/fast_embed.py barexam      # ~2.2 hr on RTX 3070
uv run python utils/fast_embed.py housing      # ~6 hr on RTX 3070
uv run python utils/fast_embed.py housing --resume
uv run python utils/fast_embed.py status
```

## Running

```bash
# Demo questions (full agentic pipeline)
uv run python main.py simple
uv run python main.py multi_hop
uv run python main.py medium
uv run python main.py simple --verbose

# Evals (all via eval_harness.py — 17 modes, 5 datasets)
uv run python eval/eval_harness.py --mode llm_only --provider groq-llama70b --questions 200
uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 200 --dataset housing
uv run python eval/eval_harness.py --mode confidence_gated --provider groq-llama70b --questions 200

# List providers
uv run python llm_config.py
```

## Key Results (N=200, seed=42 unless noted)

| Mode | BarExam (Llama 70B) | HousingQA (Llama 70B) | CaseHOLD (Llama 70B) |
|---|---|---|---|
| llm_only | 64% | 47% | 72.5% |
| rag_snap_hyde | 76.5% | **56%** | 71% |
| confidence_gated | **79%** | 50.5% | 72.5% |

RAG helps most when the model has a genuine knowledge gap (HousingQA). On better-known domains, retrieval is often neutral or harmful unless carefully gated.

## Pipeline Architecture

Source of truth: `main.py`

```
START → router_node → planner_node → parallel_executor_node → parallel_synthesizer_node
                          ↑                                          |
                          └── parallel_replanner_node ←──────────────┘ (if incomplete)
                                                                     └→ END (if complete)
```

- **router_node** — Chooses ChromaDB collection(s): `legal_passages`, `housing_statutes`
- **planner_node** — Decomposes question into 1-5 PlanningSteps
- **parallel_executor_node** — Executes steps with per-step escalation (rag_search → web_search → direct_answer)
- **parallel_synthesizer_node** — IRAC synthesis + completeness check (max 3 rounds)

## Retrieval Stack

Source of truth: `rag_utils.py`

- ChromaDB persisted in `./chroma_db/`
- Embedding: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 8192 tokens)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hybrid: BM25 + dense, cross-encoder reranks (BM25 skipped for collections >1M docs)

## Project Structure

```
main.py                    # Full pipeline: graph, nodes, executor, synthesizer (1000 LOC)
rag_utils.py               # ChromaDB, BM25, cross-encoder, multi-query retrieval
llm_config.py              # 30+ LLM provider configs, LRU-cached
web_scraper.py             # DuckDuckGo + trafilatura for web_search steps
skills/                    # 4 prompt files: planner, query_rewriter, synthesize_and_cite, synthesizer
eval/
  eval_harness.py          # Unified eval: 17 modes, 5 datasets, JSONL logging
  eval_config.py           # Config, question loaders, answer extractors
  eval_analyze.py          # Post-hoc JSONL analysis
  curate_questions.py      # One-time question curation utility
utils/
  fast_embed.py            # GPU bulk embedding with resume support
  download_data.py         # BarExam dataset fetcher
  download_housingqa.py    # HousingQA dataset fetcher
  download_new_datasets.py # CaseHOLD, Legal-RAG-QA, Australian Legal QA fetcher
RESEARCH.md                # Research state, experiment queue, session handoff
EXPERIMENTS.md             # Full experiment log (hypothesis → result → verdict)
CLAUDE.md                  # Operational source of truth
ideas/                     # Archived idea docs (active queue in RESEARCH.md)
logs/                      # Eval output (gitignored)
datasets/                  # Downloaded data (gitignored)
chroma_db/                 # Vector store (gitignored)
```

## Datasets

| Dataset | Collection | Docs | QA format | Source |
|---|---|---|---|---|
| BarExam QA | `legal_passages` | 686,324 | MC (A-D) | `reglab/barexam_qa` |
| HousingQA | `housing_statutes` | 1,837,403 | Yes/No | `reglab/housing_qa` |
| CaseHOLD | `casehold_holdings` | 50,291 | MC (A-E) | `coastalcph/lex_glue` |
| Legal-RAG-QA | `legal_rag_passages` | 190 | Open-ended | `isaacus/legal-rag-qa` |
| Australian Legal QA | `australian_legal` | 2,124 | Open-ended | `isaacus/open-australian-legal-qa` |
