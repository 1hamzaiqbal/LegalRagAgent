# LegalRagAgent

Agentic Legal RAG system built on LangGraph. Uses a **plan-and-execute** loop to answer legal research questions by retrieving passages from ChromaDB vector stores of bar exam materials and housing statutes.

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

Four-node LangGraph state machine with adaptive replanning and judge-driven control flow.

```
START → planner → executor → replanner ─┬─→ executor  (next/retry)
                                         └─→ synthesizer → END
```

- **Planner**: Decomposes question into 2-5 research steps, each with an action type (`rag_search`, `web_search`, or `direct_answer`)
- **Executor**: Per step — rewrites query into primary + 2 alternatives, hybrid retrieves from ChromaDB (BM25 + bi-encoder → cross-encoder rerank), synthesizes cited sub-answer. LLM judge evaluates retrieval sufficiency. For web_search steps, DuckDuckGo finds URLs then trafilatura scrapes full page content.
- **Replanner**: Deterministic escalation on failure (rag → rewrite → web → direct_answer). LLM decides next/complete/retry when judge is satisfied. Hard cap: 5 completed steps.
- **Synthesizer**: Aggregates all research into a final IRAC-style answer with `[Evidence N]` citations.

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

Hybrid BM25 + dense retrieval with cross-encoder reranking:
1. BM25 (keyword, `rank-bm25`) retrieves top-20 candidates
2. Bi-encoder (`gte-large-en-v1.5`) retrieves top-20 candidates
3. Pool + deduplicate by passage ID
4. Cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks to top 5

### Datasets

| Dataset | Corpus size | QA pairs | Domain |
|---------|-------------|----------|--------|
| [BarExam QA](https://huggingface.co/datasets/reglab/barexam_qa) | 686K passages | MC (A-D) | Bar exam (all subjects) |
| [HousingQA](https://huggingface.co/datasets/reglab/housing_qa) | 1.84M statutes | 6,853 Yes/No | Housing law (51 jurisdictions) |

## Evaluation

```bash
uv run python eval/eval_qa.py 50                    # QA eval on 50 questions
uv run python eval/eval_qa.py 100 --continue         # Resume interrupted eval
uv run python eval/eval_baseline.py 50               # Direct-LLM baseline
uv run python eval/eval_reranker.py                   # Retrieval A/B test
uv run python eval/eval_retrieval.py                  # BM25 vs dense vs hybrid comparison
```

## Project Structure

```
main.py                # LangGraph pipeline (4 nodes, routing, state)
rag_utils.py           # Hybrid retrieval: BM25 + dense + cross-encoder reranking
llm_config.py          # Provider registry, LLM singleton
web_scraper.py         # Web page text extraction (trafilatura) for web_search steps
skills/                # 7 prompt files
eval/                  # Evaluation scripts
  eval_qa.py           # Full QA evaluation on N queries
  eval_baseline.py     # No-RAG baseline for comparison
  eval_reranker.py     # Retrieval A/B test
  eval_retrieval.py    # BM25 vs dense vs hybrid comparison
  eval_utils.py        # Shared eval utilities
  web_search_suite.py  # Web search query definitions
utils/                 # Setup and utility scripts
  download_data.py     # Download BarExam QA from HuggingFace
  download_housingqa.py # Download HousingQA from HuggingFace
  fast_embed.py        # GPU-optimized bulk embedding (fp16, chunked)
  load_corpus.py       # LangChain-based corpus loader (smaller subsets)
  render_graph.py      # Graph visualization
  cudacheck.py         # GPU availability check
```
