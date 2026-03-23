# AGENTS.md

Source-of-truth context for working in this codebase. Verify claims against `legal_rag/runtime.py`, `legal_rag/nodes.py`, `legal_rag/step_executor.py`, and `legal_rag/retrieval.py` before relying on them. `main.py` is a thin CLI and compatibility wrapper on this branch.

## Project Summary

Legal RAG agent over the `reglab/barexam_qa` and `reglab/housing_qa` corpora. The branch uses a profile-driven round executor that supports sequential and true-parallel step execution, structured run artifacts, and curated playtests before large evaluations.

## Runtime Architecture

Primary source of truth:

- `legal_rag/runtime.py`
- `legal_rag/nodes.py`
- `legal_rag/step_executor.py`
- `legal_rag/retrieval.py`
- `legal_rag/profiles.py`

### Graph

```text
START -> router_node -> planner_node -> execute_round_node
                                           |
                                           v
                                   synthesizer_node
                                           |
                                 terminal? v
                                     replanner_node
                                           |
                                           v
                                     planner_node
```

`route_after_execution_round()` controls whether execution loops again immediately or hands off to the synthesizer. `route_after_synthesizer()` decides whether to end or create another research round.

### Profiles

The runtime is profile-driven. Current named profiles:

- `llm_only`
- `simple_rag`
- `rewrite_rag`
- `full_seq`
- `full_parallel`
- `full_parallel_aspect`

`full_seq` and `full_parallel` share the same runtime components. The difference is `step_execution_mode`, not a separate pipeline implementation.

### Nodes

**router_node**
- Chooses which ChromaDB collection(s) to search.
- Current registry: `legal_passages`, `housing_statutes`.
- Falls back to `legal_passages`.

**planner_node**
- Decomposes the research question into `PlanningStep`s.
- Outputs up to 5 steps initially and up to 3 follow-up steps per replanning round.
- Each step carries `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, and `max_retries`.
- Falls back to a single `rag_search` step on parse failure.
- On follow-up rounds, reads `replanning_brief` and appends only new steps instead of replacing completed history.

**execute_round_node**
- Executes pending steps according to profile.
- `full_seq`: one pending step at a time.
- `full_parallel`: all pending sibling steps from the same round snapshot.
- Uses `_execute_step_with_escalation()` for per-step behavior.
- Merges sibling results with canonical evidence dedup in `_merge_round_results()`.

**synthesizer_node**
- Aggregates completed step results into the final answer using `skills/synthesizer.md`.
- Runs the completeness check unless disabled by profile.
- Records both whether the question was actually answered and why the run stopped.
- Current terminal reasons are `answered`, `max_rounds`, `stalled`, `loop_disabled`, and `parse_failure`.

**replanner_node**
- Distills completed work, weak spots, and `missing_topics` into a plain-language `replanning_brief`.
- Routes back to `planner_node` for a fresh follow-up planning pass.
- Does not handle per-step retry/escalation directly.

### Round Semantics

This branch is intentionally round-safe:

- sibling steps in the same round see the same immutable snapshot of prior state
- sibling steps do not see each other's new evidence mid-round
- evidence dedup happens at merge time, not during sibling execution
- per-step evidence IDs are preserved even when the canonical evidence store is deduplicated
- replanning is driven by a compact follow-up brief, not by dumping raw evidence passages and traces into the planner prompt
- fallback `direct_answer` results after failed retrieval are downgraded to `support_only` so they cannot silently masquerade as decisive authority

### Per-Step Execution

Implemented in `_execute_step_with_escalation()` inside `legal_rag/step_executor.py`.

`rag_search`
- Standard mode: query rewrite -> retrieve -> synthesize -> judge
- Aspect mode: rule/exception/application queries -> pooled retrieval -> synthesize -> judge
- On insufficiency: rewrite once, then fall back according to profile

`web_search`
- DuckDuckGo text search -> scrape top URLs -> synthesize -> judge
- On insufficiency: fall back according to profile

`direct_answer`
- Answer from model knowledge -> verifier
- If reached only after retrieval failure, the result is retained as support but downgraded to `support_only`

Per-step fallback is executor-local. Replanning happens only after synthesis/completeness identifies new missing topics.

## Shared State (`LegalAgentState`)

- `agent_metadata` — provider, model, timestamps, prompt versions
- `inputs` — includes `question` and `research_question`
- `run_config` — includes `max_steps` and `max_parallel_rounds`
- `profile` — serialized `ExperimentProfile`
- `collections` — routed collection set
- `planning_table` — list of `PlanningStep`
- `evidence_store` — canonical accumulated evidence
- `final_answer` — synthesizer output
- `audit_log` — structured node-level trace
- `completeness_verdict` — synthesizer completeness and terminal result
- `parallel_round` — current synthesis/replan round counter
- `replanning_brief` — plain-language follow-up brief for the next planner pass
- `step_traces` — structured per-step attempt traces
- `run_artifact` — structured execution artifact payload

`PlanningStep` currently includes:

- `step_id`
- `sub_question`
- `authority_target`
- `retrieval_hints`
- `action_type`
- `max_retries`
- `rewrite_attempt`
- `status`
- `result`
- `confidence`
- `evidence_ids`
- `retry_of`
- `judge_verdict`
- `result_origin`
- `support_level`

## Skills

Current prompt files in `skills/`:

| Skill file | Purpose |
|---|---|
| `planner.md` | Generate research steps |
| `query_rewriter.md` | Standard multi-query rewrite for `rag_search` |
| `aspect_query_rewriter.md` | Rule / exception / application queries for aspect retrieval |
| `synthesize_and_cite.md` | Step-level cited synthesis |
| `judge.md` | Sufficiency judgment for retrieved evidence |
| `verifier.md` | Grounding check for direct answers |
| `synthesizer.md` | Final synthesis and IRAC-style answer |

If you change step execution, retrieval strategy, or prompt contracts, audit both the runtime code and these prompt files together.

## Retrieval Stack

Source of truth: `rag_utils.py`

- ChromaDB persisted to `./chroma_db/`
- `legal_passages`: 686,324 barexam passages
- `housing_statutes`: 1,837,403 housing statutes
- Embedding model: `Alibaba-NLP/gte-large-en-v1.5`
- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hybrid retrieval: dense plus BM25 where collection size allows
- BM25 skipped above 1M docs, so `housing_statutes` is effectively dense-only

## LLM Configuration

Source of truth: `llm_config.py`

- Provider selection via `LLM_PROVIDER`
- `get_llm()` returns a cached client
- `_llm_call()` in `legal_rag/core.py` handles retries and token metrics
- Run `uv run python llm_config.py` to list providers

## Logging and Artifacts

- `--verbose` or `VERBOSE=1` enables token-count logging for LLM calls
- `legal_rag/artifacts.py` writes structured run artifacts to `logs/run_artifacts/`
- Artifacts capture profile name, graph mode, prompt versions, planning table, evidence store, completeness verdict, stop reason, audit log, and step traces
- Eval detail rows include `artifact_path`, `parallel_rounds`, `collections`, `completeness_verdict`, `terminal_reason`, timings, and LLM metrics

## Commands

```bash
# Install
uv sync

# Configure
cp .env.example .env

# Download datasets
uv run python utils/download_data.py
uv run python utils/download_housingqa.py

# Build vector stores
uv run python utils/fast_embed.py barexam
uv run python utils/fast_embed.py housing
uv run python utils/fast_embed.py housing --resume
uv run python utils/fast_embed.py status

# Run runtime profiles
uv run python main.py --list-profiles
uv run python main.py simple --profile full_parallel
uv run python main.py medium --profile full_seq
uv run python main.py --question "What are the elements of adverse possession?" --profile full_parallel_aspect

# Evals
uv run python eval/eval_baseline.py 100 --suite bar
uv run python eval/eval_bm25_baseline.py 100
uv run python eval/eval_rag_rewrite.py 100
uv run python eval/eval_qa.py 100 --profile full_parallel
uv run python eval/run_playtests.py --profile full_parallel_aspect

# Tests
uv run pytest -q -s
```

## Historical Results

Reported results in the repo are historical pre-redesign numbers. Treat them as reference points, not guaranteed current behavior for this branch.

## Editing Guidance

- For runtime behavior, start in `legal_rag/runtime.py`, then read `legal_rag/nodes.py`, `legal_rag/step_executor.py`, and `legal_rag/retrieval.py`.
- Keep `main.py` readable and small; it should stay a thin wrapper around the real runtime.
- If you change profiles, update both `legal_rag/profiles.py` and any eval/playtest entrypoints that expose them.
- If you change step schema or state keys, update runtime code, tests, and prompt contracts together.
- `main.py` still re-exports compatibility helpers used by some legacy scripts, but new code should prefer importing from `legal_rag`.
- `web_scraper.py` remains a standalone module used by `web_search` steps.
- `utils/fast_embed.py` bypasses LangChain for bulk embedding and supports `--resume`.
