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

Set `LLM_PROVIDER` to switch between backends. Default: **Google AI Studio** (Gemma 3 27B, 14.4K RPD free tier). Run `uv run python llm_config.py` to see all available providers.

### 3. Load passages (first run only)

```bash
uv run python load_corpus.py 20000          # 20K passages (~30 min with gte-large)
uv run python load_corpus.py curated        # Gold passages + 500 padding (~1.5K, ~3 min)
uv run python load_corpus.py status         # Check current collection size
```

### 4. Run the agent

```bash
uv run python main.py simple       # "What are the elements of a negligence claim?"
uv run python main.py multi_hop    # Constitutional rights + 4th/5th Amendment search scenario
uv run python main.py medium       # Preliminary injunction standard and factors
```

## Architecture

Nine-node LangGraph state machine with adaptive replanning, injection detection, MC answer selection, and QA memory.

![LangGraph Pipeline](graph.png)

- **Injection Check**: Screens for adversarial prompts (skippable via `SKIP_INJECTION_CHECK=1`)
- **Classifier**: Routes queries as `simple` (1 step, 4 LLM calls) or `multi_hop` (adaptive steps, 10-11 calls)
- **Planner**: Checks QA memory cache first (cosine >= 0.92), then generates initial research step. Strips MC answer choices before planning.
- **Executor**: Per step — rewrites query into primary + 2 alternatives, multi-query retrieves from ChromaDB (with cross-step dedup), synthesizes answer with inline `[Query X][Source N]` citations
- **Evaluator**: Checks confidence against threshold (`EVAL_CONFIDENCE_THRESHOLD`, default 0.70). Accumulates step summaries for replanner.
- **Replanner**: (multi_hop only) Adaptively adds research steps based on accumulated evidence. Hard cap: 3 completed steps.
- **MC Select**: For MC questions, applies accumulated research to select answer letter. Non-MC passes through.
- **Memory**: Caches answers (confidence >= 0.70) for future retrieval. Strips MC selection before caching.
- **Observability**: Prints run metrics (LLM calls, confidence, steps, timing).

### Skills (7 prompt files)

| Skill | Purpose |
|-------|---------|
| `classify_and_route.md` | Classify query complexity (simple vs multi_hop) |
| `plan_synthesis.md` | Generate initial research step |
| `query_rewrite.md` | Rewrite into primary + 2 alternative queries (JSON) |
| `synthesize_and_cite.md` | Synthesize answer with inline `[Source N]` citations |
| `adaptive_replan.md` | Decide next research step from accumulated evidence |
| `detect_prompt_injection.md` | Screen for adversarial prompts |
| `verify_answer.md` | Cross-check answer against evidence (retained for future use) |

See [ARCHITECTURE.md](ARCHITECTURE.md) for full node-by-node reference, state schema, and annotated case studies.

## Evaluation

```bash
uv run python eval_comprehensive.py              # Two-phase: retrieval + full pipeline
uv run python eval_comprehensive.py retrieval     # Phase 1 only (no LLM)
uv run python eval_comprehensive.py pipeline 10   # Phase 2, first N queries
uv run python eval_trace.py 3                     # Traced diagnostics, first N queries
uv run python eval_trace.py 3 --save              # Save case studies to case_studies/
```

### Latest Results (Gemma 3 27B, 20K passages)

| Query | MC | Steps | Conf | LLM calls |
|---|---|---|---|---|
| torts | Y | 3c/0f | 0.773 | 11 |
| contracts | Y | 3c/0f | 0.773 | 11 |
| crimlaw | N | 3c/0f | 0.721 | 11 |
| evidence | Y | 3c/0f | 0.790 | 11 |
| constlaw | N | 0c/3f | — | 11 |
| realprop | Y | 3c/0f | 0.775 | 11 |

MC accuracy: 4/6. 100% passage diversity across all steps.
