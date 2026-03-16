# CLAUDE.md

This file captures reusable repository context for future work in this codebase. It should track the code, not the historical docs.

## Read This First

- `README.md` is useful for setup and high-level intent, but parts of it are stale.
- The active pipeline is in `main.py`.
- `main_old.py` still exists and is used by `eval/eval_trace.py`; do not assume all eval scripts exercise the same graph.
- `docs/ARCHITECTURE.md` and `docs/pipeline_flags.md` describe an older pipeline generation in several places. Verify against code before relying on them.

## Project Summary

Legal RAG agent over the `reglab/barexam_qa` corpus.

Current design:
- LangGraph plan-and-execute workflow
- ChromaDB vector store for passage retrieval
- HuggingFace embeddings + cross-encoder reranking
- LLM-driven planning, query rewrite, judging, replanning, and final synthesis

The codebase has shifted from an older classify/plan/evaluate pipeline to a simpler planner/executor/replanner/synthesizer loop.

## Current Runtime Architecture

`main.py` is the source of truth.

Top-level graph:
- `planner_node`
- `executor_node`
- `replanner_node`
- `synthesizer_node`

Graph shape:
- `START -> planner_node -> executor_node -> replanner_node`
- `replanner_node -> executor_node` for `next` or `retry`
- `replanner_node -> synthesizer_node` for `complete`
- `synthesizer_node -> END`

Important behavior:
- Planner decomposes the question into ordered `PlanningStep`s.
- Executor runs one pending step at a time.
- Step `action_type` can be `rag_search`, `web_search`, or `direct_answer`.
- RAG steps use LLM query rewrite, multi-query retrieval, cross-step evidence dedup, then cited synthesis.
- Every executed step is judged for sufficiency by an LLM (`judge.md` for retrieval-backed steps, `verifier.md` for `direct_answer`).
- Replanner uses deterministic escalation for insufficient steps:
  - `rag_search` -> rewritten `rag_search`
  - `rag_search` -> `web_search`
  - `web_search` -> `direct_answer`
- When a step is sufficient, replanner uses `skills/replanner.md` to decide `next`, `retry`, or `complete`.
- Final answer is synthesized from completed steps in IRAC style by `skills/synthesizer.md`.

## Shared State

The active state object is `LegalAgentState` in `main.py`, not the older state schema referenced by some docs.

Key fields:
- `agent_metadata`
- `inputs`
- `run_config`
- `planning_table`
- `evidence_store`
- `final_answer`
- `run_metrics`
- `audit_log`

`PlanningStep` tracks:
- `step_id`
- `sub_question`
- `authority_target`
- `retrieval_hints`
- `action_type`
- `rewrite_attempt`
- `status`
- `result`
- `confidence`
- `evidence_ids`
- `retry_of`
- `judge_verdict`

## Skills

The active runtime loads prompt files dynamically from `skills/*.md`.

Current skills used by `main.py`:
- `planner.md`
- `executor.md`
- `judge.md`
- `replanner.md`
- `synthesize_and_cite.md`
- `synthesizer.md`
- `verifier.md`

Present but not used by the active `main.py` flow:
- `adaptive_replan.md`
- `classify_and_plan.md`
- `detect_prompt_injection.md`
- `query_rewrite.md`
- `verify_answer.md`

Do not delete unused skill files without checking whether older scripts or archived flows still depend on them.

## Retrieval Stack

Source of truth: `rag_utils.py`

- Chroma persist dir: `./chroma_db`
- Main collection: `legal_passages`
- Default embedding model: `Alibaba-NLP/gte-large-en-v1.5`
- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Optional source-aware retrieval toggle: `SOURCE_DIVERSE_RETRIEVAL=1`

Retrieval details:
- `retrieve_documents()` does bi-encoder retrieval then cross-encoder reranking.
- `retrieve_documents_multi_query()` pools candidates across query variants, dedups by `idx`, and reranks against the primary query.
- Cross-step dedup is implemented by passing prior evidence IDs as `exclude_ids`.
- `compute_confidence()` returns the max stored cross-encoder score from retrieved docs.

Important caveat:
- `load_passages_to_chroma()` skips reload when existing count is already `>=` desired count. Switching from a larger store to a smaller one requires manually clearing `chroma_db/`.

## LLM Configuration

Source of truth: `llm_config.py`

- Provider selection is controlled by `LLM_PROVIDER`.
- If `LLM_PROVIDER` is unset or unknown, the code falls back to `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`.
- `get_llm()` returns a cached `ChatOpenAI` client.
- `main._llm_call()` adds retry handling for transient failures and merges system prompts into a single user message for Gemma models.

Current built-in providers include:
- DeepSeek
- Google AI Studio
- Groq
- OpenRouter
- Cerebras
- Ollama

Run `uv run python llm_config.py` to verify the exact current registry.

## Setup And Common Commands

Install:

```bash
uv sync
```

Configure env:

```bash
cp .env.example .env
```

Download dataset:

```bash
uv run python download_data.py
uv run python download_data.py --check
```

Build corpus:

```bash
uv run python load_corpus.py curated
uv run python load_corpus.py curated 2000
uv run python load_corpus.py 20000
uv run python load_corpus.py status
```

Run agent:

```bash
uv run python main.py simple
uv run python main.py multi_hop
uv run python main.py medium
```

Eval scripts:

```bash
uv run python eval/eval_qa.py 50
uv run python eval/eval_qa.py 50 --parallel 5
uv run python eval/eval_qa.py 100 --continue

uv run python eval/eval_baseline.py 50
uv run python eval/eval_baseline.py 100 --continue

uv run python eval/eval_comprehensive.py
uv run python eval/eval_comprehensive.py retrieval
uv run python eval/eval_comprehensive.py pipeline
uv run python eval/eval_comprehensive.py pipeline 10

uv run python eval/eval_reranker.py
uv run python eval/eval_trace.py
uv run python eval/eval_trace.py 3
uv run python eval/eval_trace.py --query "..."
uv run python eval/eval_trace.py 3 --save
```

## Evaluation Notes

- `eval/eval_qa.py` uses the current `main.py` graph and sets `SKIP_INJECTION_CHECK=1`, but the current `main.py` no longer has an injection node. That env var is effectively a leftover.
- `eval/eval_baseline.py` is a direct-LLM baseline and bypasses LangGraph.
- `eval/eval_comprehensive.py` evaluates retrieval plus the current pipeline.
- `eval/eval_trace.py` imports from `main_old.py`, not `main.py`. Treat its traces as legacy-flow diagnostics unless it gets migrated.

## Data And Generated Artifacts

Typically gitignored / generated locally:
- `datasets/barexam_qa/`
- `chroma_db/`
- `logs/`
- `case_studies/`

Useful files:
- `graph.png`
- `docs/ARCHITECTURE.md`
- `docs/pipeline_flags.md`

## Practical Guidance For Future Edits

- Verify architecture claims against `main.py` before updating docs or prompts.
- If you change step schema or routing logic, audit both `main.py` and the prompt contracts in `skills/`.
- If you change evaluation assumptions, check `eval/eval_qa.py`, `eval/eval_baseline.py`, and `eval/eval_comprehensive.py` separately.
- Be careful with older docs and scripts that still reference:
  - injection detection
  - classify/simple vs multi-hop routing
  - verifier / MC-selection nodes
  - `main_old.py` state fields
- When debugging retrieval quality, inspect both `rag_utils.py` and whether the Chroma store was built from `curated`, `20k`, or another corpus size.

## Known Documentation Drift

At the time of this update:
- `README.md` still describes the older classify-plan-execute-evaluate architecture.
- `docs/ARCHITECTURE.md` still documents the older 7-node graph.
- `eval/eval_trace.py` still targets `main_old.py`.

If those files are updated later, keep this file aligned with the current executable path rather than copying historical descriptions forward.
