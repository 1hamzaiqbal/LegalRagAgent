# LegalRagAgent

Legal RAG research project studying **when retrieval helps legal QA and when it hurts**.

The repo contains two layers:
- `main.py` — the full LangGraph agentic pipeline / demo system
- `eval/` — the current research loop, where simpler adaptive methods are benchmarked against heavier agentic variants

**Project direction:** the long-term goal is still a strong full agentic pipeline, but the current research program is rebuilding toward it atomically from smaller, controlled retrieval strategies and only keeping improvements that survive fixed-eval scrutiny.

**Current headline results:**
- BarExam full-N best: Gemma 4 26B-A4B `rag_snap_hyde` = **81.17%** (N=1195, post-prompt, audited); historical Llama 70B `ce_threshold` = **80.0%** (N=200)
- HousingQA best: `rag_snap_hyde` on Llama 70B = **56.0%** (N=200)
- CaseHOLD best: `llm_only` / `confidence_gated` = **72.5%** (N=200)
- Best small-model full tier: **Gemma 4 E4B** — `rag_snap_hyde` **62.18%**, `subagent_rag` **60.92%**, `snap_hyde_report` **60.75%**, `rag_hyde` **60.59%** at N=1195 post-prompt
- MuSiQue multi-hop: `multi_hyde_diverse` is the first cross-family lift at N=100 (Llama 70B **33.0%** vs 21.0%, +12pp p=0.023; Gemma 3 27B **30.0%** vs 22.0%, +8pp p=0.134). `iter_hyde` hurts Gemma 3 27B at N=30 (**6.7%**, -20pp vs rag_simple).
- Working interpretation: `rag_snap_hyde` is the current legal-MC winner; `multi_hyde_diverse` is the current multi-hop exception; showing snap to the final agent still hurts.
- **Multi-turn reasoning** (historical `vectorless_*` family): `vectorless_direct` **64.5%**, `vectorless_hybrid` **65.0%** — LLM parametric knowledge, not corpus search
- **Real structured search** (in progress): case summary index + NLP entity graph for actual corpus navigation without embeddings
- `logs/experiments.jsonl` contains **288** records as of 2026-04-27 early
- 53 eval modes tested across retrieval, reasoning, gap, and subagent architectures

See `RESEARCH.md` for the current state + queue, and `EXPERIMENTS.md` for the full keep/discard history.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/1hamzaiqbal/LegalRagAgent.git
cd LegalRagAgent
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

# Evals (all via eval_harness.py — 53 modes, 5 datasets)
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
| ce_threshold | **80%** | — | — |

RAG helps most when the model has a genuine knowledge gap (HousingQA). On better-known domains, retrieval is often neutral or harmful unless carefully gated.

Current Gemma 4 E4B audited full snapshot (N=1195, BarExam): `rag_snap_hyde` **62.18%**, `subagent_rag` **60.92%**, `snap_hyde_report` **60.75%**, `rag_hyde` **60.59%**, `rag_simple` **58.49%**.

### HPC Cluster Results (N=1195 full BarExam, local vLLM inference)

| Model | llm_only | golden_passage | rag_simple | rag_hyde | rag_snap_hyde |
|---|---|---|---|---|---|
| Gemma 4 E4B | — | — | 58.49% | 60.59% | **62.18%** |
| Gemma 4 26B-A4B | 79.75% | 78.66% | 78.08% | 78.91% | **81.17%** |

E4B `llm_only` and `golden_passage` were not completed in the Phase 12 wave; the job wallclocked after `rag_simple` and `rag_hyde`. Older 57.9%/58.6% HyDE-family rows are pre-prompt-fix historical references.

### Embedding Model Comparison (Gemma 4 E4B, N=200, BarExam)

| Embedding Model | Params | rag_simple | rag_snap_hyde |
|---|---|---|---|
| gte-large-en-v1.5 (baseline) | 434M | 57.0% | **65.5%** |
| legal-bert-base-uncased | 110M | **62.0%** | 60.0% |
| stella-en-400M-v5 | 400M | 61.0% | 60.0% |
| bge-m3 | 568M | 61.0% | 60.0% |

All alternative embedders beat baseline on `rag_simple` (+4-5pp), but `rag_snap_hyde` flattens differences to ~60%. This suggests HyDE-generated passages are already well-matched by the baseline embedder, while raw questions benefit from different embedding geometry.

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

- ChromaDB persisted in `./chroma_db/` (configurable via `CHROMA_DB_DIR` env var)
- Default embedding: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 8192 tokens)
- Embedding A/B testing: multiple models in `utils/fast_embed.py`, override via `EVAL_EMBEDDING_MODEL` env var
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Dense retrieval (k=15) → cross-encoder rerank (top 5). BM25 available but disabled by default

## Project Structure

```
main.py                    # Full pipeline: graph, nodes, executor, synthesizer (1000 LOC)
rag_utils.py               # ChromaDB, BM25, cross-encoder, multi-query retrieval
llm_config.py              # 30+ LLM provider configs, LRU-cached
web_scraper.py             # DuckDuckGo + trafilatura for web_search steps
skills/                    # 4 prompt files: planner, query_rewriter, synthesize_and_cite, synthesizer
eval/
  eval_harness.py          # Unified eval: 53 modes, 5 datasets, JSONL logging
  eval_config.py           # Config, question loaders, answer extractors
  eval_analyze.py          # Post-hoc JSONL analysis
  curate_questions.py      # One-time question curation utility
utils/
  fast_embed.py            # GPU bulk embedding with resume + A/B testing support
  build_case_summaries.py  # Case-level summary builder for structured-search experiments
  build_entity_graph.py    # Entity-graph / inverted-index builder for structured search
  download_data.py         # BarExam dataset fetcher
  download_housingqa.py    # HousingQA dataset fetcher
  download_new_datasets.py # CaseHOLD, Legal-RAG-QA, Australian Legal QA fetcher
scripts/hpc/               # SLURM job scripts for WashU HPC cluster
scripts/run_embedding_probe.sh  # Local embedding diagnostics
docs/                      # HPC throughput data, setup logs, experiment summaries
RESEARCH.md                # Research state, experiment queue, session handoff
EXPERIMENTS.md             # Full experiment log (hypothesis → result → verdict)
CLAUDE.md                  # Operational source of truth
ideas/                     # Archived idea docs (active queue in RESEARCH.md)
agentic_ideas/             # Gitignored scratch notes / archive directory
logs/                      # Eval output (gitignored)
datasets/                  # Downloaded data (gitignored)
chroma_db/                 # Vector store (gitignored)
```

Local cache-only directories such as `legal_rag/`, `playtests/`, and `tests/` may exist in a checkout, but they currently only contain `__pycache__` artifacts rather than tracked source files.

## Datasets

| Dataset | Collection | Docs | QA format | Source |
|---|---|---|---|---|
| BarExam QA | `legal_passages` | 686,324 | MC (A-D) | `reglab/barexam_qa` |
| HousingQA | `housing_statutes` | 1,837,403 | Yes/No | `reglab/housing_qa` |
| CaseHOLD | `casehold_holdings` | 50,291 | MC (A-E) | `coastalcph/lex_glue` |
| Legal-RAG-QA | `legal_rag_passages` | 190 | Open-ended | `isaacus/legal-rag-qa` |
| Australian Legal QA | `australian_legal` | 2,124 | Open-ended | `isaacus/open-australian-legal-qa` |
