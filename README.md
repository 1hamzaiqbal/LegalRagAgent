# LegalRagAgent

Agentic Legal RAG system built on LangGraph. Uses a **classify-plan-execute-evaluate** loop to answer legal research questions by retrieving passages from a ChromaDB vector store of bar exam materials.

## Quickstart

### 1. Install dependencies

```bash
uv sync
```

### 2. Set up your LLM

Copy the example env file and add your API key:

```bash
cp .env.example .env
```

The default configuration uses **Cerebras free tier** (14K requests/day, 1M tokens/day). Get an API key at [cloud.cerebras.ai](https://cloud.cerebras.ai) and paste it in `.env`.

Other supported backends: Groq, Google AI Studio, Ollama (local), OpenAI, DeepSeek, vLLM — see `.env.example` for config.

### 3. Load passages (first run only)

If the ChromaDB store hasn't been populated yet:

```bash
uv run python load_corpus.py
```

This loads the full 220K bar exam passage corpus into `./chroma_db/`.

### 4. Run the agent

```bash
uv run python main.py simple       # "What are the elements of a negligence claim?"
uv run python main.py multi_hop    # Constitutional rights + 4th/5th Amendment search scenario
uv run python main.py medium       # Preliminary injunction standard and factors
```

## Architecture

Nine-node LangGraph state machine with adaptive replanning, injection detection, answer verification, and QA memory:

```
detect_injection → classifier → planner → executor ⇄ evaluator → replanner → verify_answer → memory_writeback → observability → END
```

- **Injection Check**: Screens for adversarial prompts (skippable via `SKIP_INJECTION_CHECK=1`)
- **Classifier**: Routes queries as `simple` (1 step) or `multi_hop` (adaptive steps)
- **Planner**: Checks QA memory cache first, then LLM generates a structured research plan
- **Executor**: Per step — rewrites query into primary + 2 alternatives, multi-query retrieves from ChromaDB, synthesizes answer with inline citations in one pass
- **Evaluator**: Checks confidence (cosine similarity), accumulates context for replanner
- **Replanner**: (multi_hop only) Adaptively adds research steps based on accumulated evidence
- **Verify**: Cross-checks final answer against evidence; triggers corrective step on first failure
- **Memory**: Caches verified answers for future retrieval (cosine similarity >= 0.92)

### Skills (7 prompt files)

| Skill | Purpose |
|-------|---------|
| `classify_and_route.md` | Classify query complexity |
| `plan_synthesis.md` | Generate research plan |
| `query_rewrite.md` | Rewrite into primary + 2 alternative queries (JSON, multi-query retrieval) |
| `synthesize_and_cite.md` | Synthesize answer with inline `[Source N]` citations in one pass |
| `adaptive_replan.md` | Decide next research step from accumulated evidence |
| `detect_prompt_injection.md` | Screen for adversarial prompts |
| `verify_answer.md` | Cross-check answer against evidence |

See [ARCHITECTURE.md](ARCHITECTURE.md) for full diagrams and state schema.

## Evaluation

```bash
uv run python eval_comprehensive.py    # Two-phase: retrieval + full pipeline
uv run python eval_reranker.py         # A/B: bi-encoder vs cross-encoder reranking
```

## Required Skills

The LangGraph agent dynamically loads instructions (skills) from markdown files in the `skills/` directory. All 7 skill files are included in this repo.
