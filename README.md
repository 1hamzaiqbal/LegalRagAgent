# LegalRagAgent

Agentic Legal RAG system built on LangGraph. Uses a **plan-and-execute** loop with corpus routing, retrieval judging, and adaptive replanning to answer legal research questions over bar-exam materials and housing statutes.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/shrango/adaptive-plan-and-solve-agent.git
cd adaptive-plan-and-solve-agent
uv sync
```

Requires Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).

### 2. Configure LLM provider

```bash
cp .env.example .env
# Edit .env — add your API key for at least one provider
```

Default provider: **DeepSeek** (pay-per-use, no TPM cap). Other options: Google AI Studio, Groq, OpenRouter, Cerebras, Ollama. Run `uv run python llm_config.py` to see all providers.

### 3. Download datasets

```bash
uv run python utils/download_data.py           # BarExam QA (~686K passages)
uv run python utils/download_housingqa.py       # HousingQA (~1.84M statutes)
```

### 4. Build vector stores

```bash
# Fast embedding (bypasses LangChain, uses GPU fp16 batching)
uv run python utils/fast_embed.py barexam       # Full barexam corpus (~2.2 hr)
uv run python utils/fast_embed.py housing       # Full housing statutes (~6 hr)
uv run python utils/fast_embed.py status        # Check collection sizes

# Or use the LangChain loader for smaller subsets
uv run python utils/load_corpus.py curated      # Gold passages + padding (~3 min)
uv run python utils/load_corpus.py 20000        # First 20K passages (~30 min)
```

To rebuild from scratch, delete `chroma_db/` first. Set `HF_HUB_OFFLINE=1` after first download to skip network checks.

### 5. Run the agent

```bash
uv run python main.py simple                    # "What are the elements of a negligence claim?"
uv run python main.py multi_hop                  # Fourth Amendment suppression scenario
uv run python main.py medium                     # Preliminary injunction standard
uv run python main.py simple --verbose           # Full passage text, token counts, sub-answers
```

## Architecture

Five-node LangGraph state machine with collection routing, judge-driven control flow, and adaptive replanning.

```
START → router → planner → executor → replanner ─┬─→ executor  (next/retry)
                                                  └─→ synthesizer → END
```

- **Router**: Chooses which ChromaDB collection(s) to search, typically `legal_passages` or `housing_statutes`
- **Planner**: Decomposes the question into 2-5 research steps, each with an action type (`rag_search`, `web_search`, or `direct_answer`)
- **Executor**: Per step — rewrites `rag_search` queries into primary + 2 alternatives, retrieves from the routed collection(s), synthesizes a cited sub-answer, and runs a judge/verifier
- **Replanner**: Applies deterministic escalation on failure (`rag_search` rewrite → `web_search` → `direct_answer`) and uses an LLM to decide `next` / `retry` / `complete` when evidence is sufficient. Default budget: 7 original planned steps; retry/escalation steps do not count against the budget
- **Synthesizer**: Aggregates all completed research into a final IRAC-style answer with `[Evidence N]` citations

### Skills (7 prompt files in `skills/`)

| Skill | Purpose |
|-------|---------|
| `planner.md` | Decompose question into research steps with action types |
| `query_rewriter.md` | Rewrite sub-question into primary + 2 alternative queries |
| `synthesize_and_cite.md` | Per-step cited synthesis with `[Source N]` format |
| `judge.md` | Evaluate retrieval sufficiency for rag/web steps |
| `verifier.md` | Evaluate direct_answer grounding in doctrine |
| `replanner.md` | Decide next/complete/retry from accumulated evidence |
| `synthesizer.md` | Final IRAC synthesis with `[Evidence N]` citations |

### Retrieval

Default `rag_search` retrieval is hybrid BM25 + dense retrieval with cross-encoder reranking:
1. Query rewriter produces 1 primary query and 2 alternatives
2. Bi-encoder (`gte-large-en-v1.5`) retrieves candidates for each query variant
3. BM25 (`rank-bm25`) retrieves keyword candidates when collection size allows
4. Pool + deduplicate by passage ID
5. Cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks to top 5

Current corpus-specific nuance:

- `legal_passages` uses hybrid BM25 + dense retrieval
- `housing_statutes` currently falls back to dense-only retrieval because BM25 is skipped for collections larger than 1,000,000 documents
- `housing_statutes` documents store `state` and `citation` metadata, but the current runtime does not yet apply automatic state filtering or citation-aware retrieval

### Datasets

| Dataset | Corpus size | QA pairs | Domain |
|---------|-------------|----------|--------|
| [BarExam QA](https://huggingface.co/datasets/reglab/barexam_qa) | 686K passages | MC (A-D) | Bar exam (all subjects) |
| [HousingQA](https://huggingface.co/datasets/reglab/housing_qa) | 1.84M statutes | 6,853 Yes/No | Housing law (51 jurisdictions) |

## Evaluation

Current presentation-facing evaluations:

```bash
uv run python eval/eval_baseline.py 100
uv run python eval/eval_bm25_baseline.py 100
uv run python eval/eval_golden.py 100
```

Most recent current-system results:

- Direct LLM baseline: `85/100` in `logs/eval_baseline_deepseek_20260322_13.txt`
- Simple retrieve-and-answer baseline: `70/100` in `logs/eval_bm25_baseline_deepseek_20260322_15.txt`
- Golden-passage upper bound: `77/100` in `logs/eval_golden_deepseek_20260322_13.txt`

Other scripts in `eval/` are exploratory or older and are not the current reported results for presentation.

## Project Structure

```
main.py                # LangGraph pipeline (5 nodes, routing, state)
rag_utils.py           # Retrieval: dense + BM25 when available + cross-encoder reranking
llm_config.py          # Provider registry, LLM singleton
web_scraper.py         # Web page text extraction (trafilatura) for web_search steps
skills/                # 7 prompt files
eval/                  # Evaluation scripts
  eval_baseline.py     # Direct-LLM baseline
  eval_bm25_baseline.py # Simple retrieve-and-answer baseline (uses current hybrid retriever)
  eval_golden.py       # Golden-passage upper bound
  ...                  # Additional exploratory / older eval scripts
utils/                 # Setup and utility scripts
  download_data.py     # Download BarExam QA from HuggingFace
  download_housingqa.py # Download HousingQA from HuggingFace
  fast_embed.py        # GPU-optimized bulk embedding (fp16, chunked)
  load_corpus.py       # LangChain-based corpus loader (smaller subsets)
  render_graph.py      # Graph visualization
  cudacheck.py         # GPU availability check
```
