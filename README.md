# LegalRagAgent

Legal RAG system for the `reglab/barexam_qa` and `reglab/housing_qa` corpora. The current branch is moving toward a profile-driven, round-safe plan-and-execute pipeline with true parallel step execution, structured run artifacts, and curated playtests for iteration before large evaluations.

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
uv run python utils/download_data.py
uv run python utils/download_housingqa.py
```

### 4. Build vector stores

```bash
uv run python utils/fast_embed.py barexam
uv run python utils/fast_embed.py housing
uv run python utils/fast_embed.py housing --resume
uv run python utils/fast_embed.py status
```

Optional smaller loader:

```bash
uv run python utils/load_corpus.py curated
uv run python utils/load_corpus.py 20000
```

## Running

List profiles:

```bash
uv run python main.py --list-profiles
```

Run demo questions:

```bash
uv run python main.py simple --profile full_parallel
uv run python main.py multi_hop --profile full_parallel
uv run python main.py medium --profile full_seq
uv run python main.py --question "What are the elements of adverse possession?" --profile full_parallel_aspect
uv run python main.py simple --profile full_parallel --verbose
```

## Runtime Shape

`main.py` is intentionally thin. The live runtime is split across:

- `legal_rag/runtime.py` for graph construction and top-level execution
- `legal_rag/nodes.py` for router, planner, synthesizer, and replanner graph nodes
- `legal_rag/step_executor.py` for round execution and per-step escalation
- `legal_rag/retrieval.py` for query construction, retrieval, and step-level answer generation
- `legal_rag/execution.py` as a compatibility facade over those modules
- `legal_rag/profiles.py` for named experiment profiles
- `legal_rag/artifacts.py` for structured run logging

Current full-profile graph:

```text
START -> router_node -> planner_node -> execute_round_node
                                           |
                                           v
                                   synthesizer_node
                                           |
                               incomplete? v
                                     replanner_node
                                           |
                                           v
                                     planner_node
```

There are two important execution modes:

- `full_seq`: executes one pending step per round, then loops until no pending steps remain before synthesis.
- `full_parallel`: executes all pending steps from the same round snapshot, merges their results at a barrier, then synthesizes.

### Node responsibilities

- `router_node`: chooses the relevant collection set, usually `legal_passages` or `housing_statutes`
- `planner_node`: turns the research question into 1-5 `PlanningStep`s, and on follow-up rounds appends only new steps using the replanning brief
- `execute_round_node`: runs pending steps either sequentially or in parallel depending on profile
- `synthesizer_node`: combines completed step results into the final answer and checks completeness
- `replanner_node`: converts accumulated findings into a plain-language replanning brief and routes back to `planner_node`

### Per-step behavior

Each step owns its own escalation path inside the executor:

- `rag_search`: retrieve, synthesize, judge, optionally rewrite, then fall back if allowed
- `web_search`: search, scrape, synthesize, judge, then fall back if allowed
- `direct_answer`: answer from model knowledge and verify grounding

For `rag_strategy="aspect"`, one step generates three retrieval angles:

- `rule`
- `exception`
- `application`

Those aspect queries retrieve independently, are pooled and reranked, and then support one sub-answer for that step.

### Round semantics

The intended model on this branch is round-safe execution:

- all sibling steps in the same round see the same immutable snapshot of prior state
- sibling steps do not see each other's newly retrieved evidence mid-round
- evidence is canonically deduplicated only at merge time
- per-step evidence references and traces are preserved for synthesis and debugging
- incomplete rounds feed a summarized replanning brief back into `planner_node` instead of dumping raw evidence/traces into the planner prompt

## Profiles

The current named profiles are:

| Profile | Purpose |
|---|---|
| `llm_only` | Direct answer baseline |
| `simple_rag` | Single-step retrieve-and-answer baseline |
| `rewrite_rag` | Single-step RAG with synonym-style query rewrite |
| `full_seq` | Full pipeline with sequential step execution |
| `full_parallel` | Full pipeline with round-safe parallel step execution |
| `full_parallel_aspect` | Full pipeline with round-safe parallel step execution and aspect-specialized retrieval |

## Prompt Files

Current prompt files in `skills/`:

| Skill file | Purpose |
|---|---|
| `planner.md` | Generate research steps |
| `query_rewriter.md` | Standard synonym-style rewrite queries |
| `aspect_query_rewriter.md` | Rule / exception / application queries for aspect retrieval |
| `synthesize_and_cite.md` | Step-level cited synthesis |
| `judge.md` | Sufficiency judgment for retrieved evidence |
| `verifier.md` | Grounding check for direct answers |
| `synthesizer.md` | Final IRAC-style synthesis |

## Retrieval

Source of truth: `rag_utils.py`

- ChromaDB persisted in `./chroma_db/`
- `legal_passages`: 686,324 BarExam passages
- `housing_statutes`: 1,837,403 housing statutes
- Embedding model: `Alibaba-NLP/gte-large-en-v1.5`
- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hybrid retrieval: dense plus BM25 where the collection is small enough
- BM25 is skipped above 1M docs, so `housing_statutes` is effectively dense-only

## Evaluation and Playtests

Profile-driven evals:

```bash
uv run python eval/eval_baseline.py 100 --suite bar
uv run python eval/eval_bm25_baseline.py 100
uv run python eval/eval_rag_rewrite.py 100
uv run python eval/eval_qa.py 100 --profile full_parallel
uv run python eval/eval_qa.py 100 --profile full_parallel_aspect
```

Curated playtests:

```bash
uv run python eval/run_playtests.py
uv run python eval/run_playtests.py --profile full_parallel_aspect
```

Local tests:

```bash
uv run pytest -q -s
```

## Historical Results

These are earlier reported numbers from pre-redesign runs and should be rerun after the current branch stabilizes:

| Method | Accuracy | Gold Recall@5 |
|---|---|---|
| LLM-only | 85% | n/a |
| RAG + query rewrite | 80% | 8% |
| Golden passage upper bound | 77% | n/a |
| Simple RAG | 70% | 0% |

## Project Structure

```text
main.py                  # Thin CLI and compatibility wrapper
legal_rag/
  runtime.py             # Graph construction and top-level execution
  execution.py           # Compatibility facade over the split runtime modules
  nodes.py               # Router, planner, synthesizer, replanner graph nodes
  step_executor.py       # Round execution and per-step escalation
  retrieval.py           # Query building, retrieval, and per-step answer generation
  prompts.py             # Inline router/completeness prompts and prompt-version helpers
  state_utils.py         # Shared state serialization/logging helpers
  profiles.py            # Experiment profiles
  core.py                # LLM helpers, skill loading, parsing, metrics
  artifacts.py           # Structured run artifacts
  baselines.py           # Baseline profile runners
rag_utils.py             # Dense/BM25 retrieval and reranking
llm_config.py            # Provider configuration
web_scraper.py           # DuckDuckGo scrape helper for web-search steps
skills/                  # Prompt contracts used by the runtime
eval/                    # Profile-driven eval scripts and retrieval experiments
playtests/               # Curated manual regression cases
tests/                   # Runtime and prompt-contract tests
utils/                   # Data download, embedding, and utility scripts
```
