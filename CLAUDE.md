# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Legal RAG system built on LangGraph. It uses a classify-plan-execute-evaluate loop to answer legal research questions by retrieving passages from a ChromaDB vector store of bar exam materials. LLM calls are made via `langchain-openai`'s `ChatOpenAI`, compatible with any OpenAI-compatible API (Google AI Studio, Groq, OpenRouter, Cerebras, Ollama, etc.).

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run the agent with a demo query
uv run python main.py simple       # Single-concept question (negligence elements)
uv run python main.py multi_hop    # Multi-concept constitutional rights scenario
uv run python main.py medium       # Preliminary injunction standard

# Run comprehensive evaluation (retrieval + full pipeline)
uv run python eval_comprehensive.py              # Both phases
uv run python eval_comprehensive.py retrieval     # Phase 1 only (no LLM)
uv run python eval_comprehensive.py pipeline      # Phase 2 only
uv run python eval_comprehensive.py pipeline 10   # Phase 2, first N queries

# Run traced experiment (detailed per-query diagnostics)
uv run python eval_trace.py                       # All 8 trace queries
uv run python eval_trace.py 3                     # First N queries
uv run python eval_trace.py --query "What is..."  # Custom query
uv run python eval_trace.py 3 --save              # Save case studies to case_studies/

# Run retrieval A/B test (bi-encoder vs cross-encoder reranking)
uv run python eval_reranker.py

# List all available LLM providers
uv run python llm_config.py

# Load/check corpus
uv run python load_corpus.py 20000         # Load first 20K passages
uv run python load_corpus.py status        # Check current collection size
uv run python load_corpus.py curated       # Gold passages + 500 padding (~1.5K)
uv run python load_corpus.py curated 2000  # Gold passages + 2000 padding (~3K)
```

## Environment Setup

### LLM Provider (`.env`)

Set `LLM_PROVIDER` to switch between backends. API keys are stored per-provider:

```bash
LLM_PROVIDER=gemma              # Google AI Studio: gemma-3-27b-it (14.4K RPD, free)
# LLM_PROVIDER=groq-llama70b    # Groq: llama-3.3-70b (1K RPD)
# LLM_PROVIDER=or-llama70b      # OpenRouter: llama-3.3-70b (free tier)
# LLM_PROVIDER=cerebras         # Cerebras: llama-3.3-70b (14K RPD)
# LLM_PROVIDER=ollama           # Local: llama3 via Ollama

GOOGLE_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
CEREBRAS_API_KEY=...
```

Run `uv run python llm_config.py` to see all 19 providers with rate limits.

### Embedding Model

Default: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 434M params, 8192 token context). Configurable via `EMBEDDING_MODEL` env var. Previous model was `all-MiniLM-L6-v2` (384d, 22M params).

### Evaluator Threshold

`EVAL_CONFIDENCE_THRESHOLD` sets the cosine similarity threshold for marking a research step as completed vs failed. Default `0.70` (calibrated for gte-large-en-v1.5, observed range 0.709-0.804). Previous default was 0.6 which was a no-op (0 failures in 24 traced steps). Old MiniLM default was `0.4`.

### Retrieval Mode

`SOURCE_DIVERSE_RETRIEVAL=1` enables source-aware pool splitting (MBE/wex vs caselaw). Default is off — the cross-encoder picks the best k passages regardless of source.

## Architecture

### LangGraph Workflow (main.py)

Nine-node state machine with adaptive replanning, injection detection, answer verification, QA memory, and observability:

1. **detect_injection_node** — Screens input for adversarial prompt injection using `detect_prompt_injection.md` skill. Skippable via `SKIP_INJECTION_CHECK=1` env var (saves 1 LLM call for eval/testing). Unsafe inputs are rejected and routed to observability for metrics.
2. **classifier_node** — Classifies objective as `simple` or `multi_hop` using `classify_and_route.md` skill (includes MC-specific guidance: single-concept MC → simple, multi-concept MC → multi_hop, defaults to multi_hop when in doubt)
3. **planner_node** — Checks QA memory first (cosine similarity >= 0.92); on cache hit, short-circuits to memory writeback. Otherwise generates initial plan. For `multi_hop`, emits only the first step; replanner handles the rest adaptively. **MC isolation**: strips answer choices from objective before planning — research stays unbiased.
4. **executor_node** — For each pending step: rewrites query into primary + 2 alternatives (`query_rewrite.md`, JSON output), multi-query retrieves from ChromaDB (`retrieve_documents_multi_query`, excludes doc_ids from prior steps for cross-step dedup), synthesizes answer with inline `[Source N]` citations in a single pass (`synthesize_and_cite.md`), computes confidence via cosine similarity
5. **evaluator_node** — Marks steps completed or failed against a configurable confidence threshold (`EVAL_CONFIDENCE_THRESHOLD` env var, default 0.70). Accumulates step summaries into `accumulated_context`. Sets explicit failure message if all steps fail.
6. **replanner_node** — (multi_hop only) Receives objective (MC choices stripped) + accumulated evidence, decides: `next_step` (add new research step), `retry` (rephrase failed step), or `complete` (aggregate final answer). Hard cap: 3 completed steps max. On persistent LLM failure, gracefully falls back to `complete` to preserve accumulated evidence.
7. **verify_answer_node** — MC answer selection node. Verifier LLM call removed (was always passing — see R1 in pipeline_flags.md). Auto-passes verification. **MC selection**: if the objective contains answer choices, a dedicated `skill_select_mc_answer()` call runs ONCE — applies all accumulated research to pick a letter with `**Answer: (X)**` format. This keeps per-step research unbiased.
8. **memory_writeback_node** — Persists successful QA pairs (avg confidence >= 0.70) to ChromaDB `qa_memory` collection (cosine distance) for future cache hits. Strips MC selection from cached answer so MC re-runs fresh each time (prevents stale letter answers). Skips write if verification failed.
9. **observability_node** — Aggregates and prints run metrics: LLM calls, char usage, parse failures, steps completed/failed, memory hit status, has_answer, injection status.

Routing:
- After injection check: 2-way — `classifier` (safe) | `observability` (unsafe, for metrics)
- After planner: 2-way — `executor` (no memory hit) | `memory_writeback` (memory hit)
- After evaluator: 3-way — `executor` (pending steps) | `replanner` (multi_hop, all done, <3 completed) | `verify_answer` (simple done, iteration limit >4, hard step cap >=3, or stagnation)
- After replanner: 2-way — `executor` (new step added) | `verify_answer` (complete)
- After verify: fixed — `memory_writeback` (always, no corrective retry)
- After memory writeback: fixed — `observability → END`

Graph: `detect_injection → {classifier | observability}`; `classifier → planner → {executor | memory_writeback}`; `executor → evaluator → {executor | replanner | verify_answer}`; `replanner → {executor | verify_answer}`; `verify_answer → memory_writeback → observability → END`

### Shared State (`AgentState`)

TypedDict with `global_objective`, `planning_table` (list of `PlanStep`), `query_type` ("simple"/"multi_hop"), `final_cited_answer`, `accumulated_context` (step summaries for replanner), `iteration_count` (loop guard, max 4), `injection_check` (safety result), `verification_result` (auto-pass dict), `memory_hit` (QA cache result, threshold 0.92), and `run_metrics` (observability data including parse failures and has_answer).

`PlanStep` model: `step_id` (float), `status` (pending/completed/failed), `phase` (str), `question` (str), `execution` (dict — stores `cited_answer`, `optimized_query`, `sources` (passage texts), `retrieved_doc_ids` (idx strings), `confidence_score`).

### Skill System (skills/)

7 markdown prompt files (~1700 words total, trimmed ~40% from original ~2800) cached at first load via `@lru_cache` in `load_skill_instructions()`. Design principle: **principles and output format only; hard rules only when necessary.**

- `classify_and_route.md` (~200w) — classify query complexity (simple vs multi_hop). MC-specific guidance: single-concept MC → simple, multi-concept MC → multi_hop, defaults to multi_hop when in doubt.
- `plan_synthesis.md` (~230w) — generate research plan (1 step for simple, 1 initial step for multi_hop). Two examples (simple + multi_hop) demonstrating correct 1-step output.
- `query_rewrite.md` (~320w) — rewrite query into primary + 2 alternatives with different legal terminology (JSON output). 5 merged rules (down from 7).
- `synthesize_and_cite.md` (~300w) — synthesize with inline `[Source N]` citations and `## Sources` map. Anti-fabrication block (critical): only state facts from evidence passages.
- `adaptive_replan.md` (~250w) — decide next research step based on accumulated evidence. Hard cap: 3 completed steps. Stagnation: 3+ consecutive failures → stop. Explicit rule: do not reference answer choices or previous step IDs.
- `detect_prompt_injection.md` (~160w) — screen input for adversarial prompts. Leads with guardrail: legal topics involving crime are SAFE. Fail-open, skippable via `SKIP_INJECTION_CHECK=1`.
- `verify_answer.md` (~240w) — cross-check answer against evidence. Retained but **not currently called** (verifier was always passing — removed in R1). Available for future use with independent-evidence architecture.

### LLM Config (llm_config.py)

- **Provider registry**: 19 providers across Google AI Studio, Groq, OpenRouter, Cerebras, Ollama — each with model name, API key env var, and rate limits (RPD/TPD)
- `get_llm()` — cached `ChatOpenAI` singleton resolved from `LLM_PROVIDER` env var (falls back to raw `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`)
- `get_provider_info()` — returns current provider name, model, and rate limits (used by eval logging)
- `list_providers()` — prints all providers with rate limits
- Transient error retry with exponential backoff (3 attempts, covers rate-limit, connection, and timeout errors; respects API-suggested retry delay)
- Gemma models: `_llm_call()` in main.py auto-converts SystemMessage to HumanMessage (Gemma doesn't support system messages via OpenAI-compatible API)

### RAG / Retrieval (rag_utils.py)

- **Embeddings**: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 434M params, 8192 token context). Configurable via `EMBEDDING_MODEL` env var. Uses `trust_remote_code=True`. Set `HF_HUB_OFFLINE=1` for cached offline inference.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (cross-encoder, cached singleton)
- **Vector store**: ChromaDB persisted to `./chroma_db/`. Currently 20K passages for fast iteration (full corpus is 686K; 220K was the previous working set).
- **Two retrieval modes** (controlled by `SOURCE_DIVERSE_RETRIEVAL` env var):
  - **Unified** (default): Over-retrieve 4x candidates from full corpus, cross-encoder reranks to top k
  - **Source-diverse**: Split into MBE/wex and caselaw pools, rerank within each, interleave (3 study + 2 caselaw)
- **Multi-query retrieval** in `retrieve_documents_multi_query(queries, k=5, exclude_ids=None)`:
  - Pools bi-encoder candidates from multiple query variants (primary + alternatives)
  - Deduplicates by `idx`, cross-encoder reranks the full pool against primary query
  - `exclude_ids` param filters out docs already retrieved in prior steps (cross-step dedup)
  - Bridges terminological gaps (e.g., "cancellation clause" also retrieves "illusory promise")
- `compute_confidence(query, docs)` returns mean cosine similarity between query and doc embeddings
- `load_passages_to_chroma(csv_path, max_passages=0)` loads passages with batch progress reporting
- Singletons: `get_vectorstore()`, `get_memory_store()`, `get_embeddings()`, `get_cross_encoder()` all cached
- QA Memory (separate `qa_memory` collection, cosine distance, same ChromaDB persist dir):
  - `check_memory(query, threshold=0.92)` — returns cached answer if near-exact match found
  - `write_to_memory(question, answer, confidence)` — stores QA pair with timestamp

### Evaluation

- `eval_comprehensive.py` — Two-phase eval:
  - Phase 1: retrieval-only Recall@5/MRR on all in-store QA pairs (dynamically sized — queries vectorstore for corpus size, not hardcoded). With 20K passages, ~953 QA pairs have gold passages in store.
  - Phase 2: full pipeline on 26 diverse queries (18 bar exam MC + 4 multi-hop + 2 out-of-corpus + 2 edge cases)
  - MC correctness checking via `_check_mc_correctness()` (3 strategies: text match, letter extraction, word overlap)
  - Clears QA memory cache before each run for clean eval
  - Logs provider info at start
  - Bar exam queries now include MC answer choices in the objective
- `eval_trace.py` — Detailed per-query diagnostics:
  - Clears QA memory cache at start for clean eval
  - Raw retrieval, query rewrite, multi-query retrieval with gold passage marking
  - Full pipeline execution trace (classify → plan → execute → verify)
  - MC correctness check and trace summary table (Steps, Verified, Confidence columns)
  - `--save` flag dumps case study JSON to `case_studies/`
- `eval_reranker.py` — A/B comparison of bi-encoder-only vs cross-encoder reranking
- `load_corpus.py` — Load passage corpus: `uv run python load_corpus.py [count|status|curated]`

## Eval Metrics Reference

- **Recall@5**: Does the gold passage appear in the top 5 retrieved? (Phase 1, no LLM)
- **Gold passage**: The specific passage in the dataset that was used to write the answer for each QA pair (`gold_idx` field). A strict metric — requires the exact passage, not a similar one.
- **MC Accuracy**: Does the pipeline's answer match the correct multiple-choice letter? Uses 3 strategies: text substring match, letter extraction regex (`**Answer: (X)**`), word overlap scoring.
- **A+B rate**: Percentage of queries graded A or B (pipeline produces a useful, cited answer)
- **Confidence**: Mean cosine similarity between query embedding and retrieved doc embeddings

### Known Eval Findings (as of 2026-02-23)

With `gte-large-en-v1.5` on 20K passages, Gemma 3 27B (via Google AI Studio):

**8-query trace results** (6 MC bar exam + 1 multi-hop + 1 out-of-corpus):

| Query | MC | Steps | Conf | Time | LLM calls |
|---|---|---|---|---|---|
| torts | Y | 3c/0f | 0.773 | 74s | 11 |
| contracts | Y | 3c/0f | 0.773 | 57s | 11 |
| crimlaw | N | 3c/0f | 0.721 | 69s | 11 |
| evidence | Y | 3c/0f | 0.790 | 78s | 11 |
| constlaw | N | 0c/3f | — | 91s | 11 |
| realprop | Y | 3c/0f | 0.775 | 74s | 11 |
| multihop | n/a | 3c/0f | 0.795 | 54s | 10 |
| oof | n/a | 1c/0f | 0.717 | 23s | 4 |

- **MC accuracy: 4/6** (torts Y, contracts Y, crimlaw N, evidence Y, constlaw N, realprop Y). Constlaw scores 0.67-0.70 (borderline threshold), crimlaw is an LLM reasoning error.
- **LLM calls**: multi_hop MC = 11 (classify + plan + 3x[rewrite + synthesize] + 3x replan + mc_select). multi_hop non-MC = 10 (no mc_select). simple = 4 (classify + plan + rewrite + synthesize).
- **Passage diversity: 100%** on every query. Cross-step `exclude_ids` working.
- **Citation format**: `[Query X][Source Y]` with `### Step N: {phase}` headers in aggregated answer.
- **Memory writeback**: MC selection stripped from cache, write threshold 0.70.
- **Verifier removed**: Was always passing (8/8). Saves 1 LLM call/query. Skill file retained for future use.

**Operational notes:**
- QA memory cache cleared automatically at eval start (eval_trace.py and eval_comprehensive.py)
- Connection errors during replanner retry via `_llm_call` and fall back to `complete` on persistent failure
- `SKIP_INJECTION_CHECK=1` saves 1 LLM call per query (recommended for eval runs)

**Bottleneck analysis:**
1. **MC selector** (biggest remaining lever): 2/6 wrong answers — crimlaw is an LLM reasoning error, constlaw is a borderline-threshold retrieval issue. Research quality is generally good.
2. **Classifier granularity**: single-concept MC → simple would save ~7 LLM calls per query, but risks less thorough research. Current multi_hop-for-everything is safe but expensive.

## Data Directories (gitignored)

- `datasets/barexam_qa/` — passage CSVs and QA dataset CSVs (from `reglab/barexam_qa` on HuggingFace)
- `chroma_db/` — persisted ChromaDB vector store
- `case_studies/` — JSON trace files from `eval_trace.py --save` (per-query pipeline diagnostics)

## Key Dependencies

`langgraph`, `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-openai`, `chromadb`, `pandas`, `pydantic`, `tqdm`, `numpy`, `python-dotenv`, `sentence-transformers`

## Known Issues & Tech Debt

See `pipeline_flags.md` for the full audit with severity ratings, evidence, and fix proposals.

**Unverified claims in this doc:**
- **Full corpus size "686K"**: From original dataset description. Actual `barexam_qa_train.csv` row count not recently verified.
- **Source-diverse retrieval mode**: `SOURCE_DIVERSE_RETRIEVAL=1` not tested with gte-large-en-v1.5. May need recalibration.
- **`eval_reranker.py`**: A/B comparison not run recently. Results may differ with current embeddings.
- **Provider registry "19 providers"**: Count may have drifted. Run `uv run python llm_config.py` to verify.

## WSL Setup

```bash
# Clone + install
git clone <repo-url> && cd LegalRagAgent
uv sync

# Copy .env and fill in API keys
cp .env.example .env

# Rebuild ChromaDB (not committed to git)
# ~30min for 20K passages, ~3.5hr for 220K with gte-large-en-v1.5 on RTX 3070
HF_HUB_OFFLINE=1 uv run python load_corpus.py 20000

# GPU note: sentence-transformers will auto-detect CUDA for embeddings/reranker.
# No code changes needed — just ensure torch has CUDA support:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# WSL DNS fix (if HuggingFace downloads fail):
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
# Or use HF_HUB_OFFLINE=1 if model is already cached
```
