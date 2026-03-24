# Actionable Ideas

Ideas gathered from experiments, branch explorations, and analysis. Not yet implemented — evaluate for interactions before building.

## Architecture Ideas

### LLM Snap + Adversarial Arbitration
- Before any retrieval, get the LLM's "gut instinct" answer (1 extra call)
- After pipeline completes, if pipeline answer disagrees with snap, run an arbitration step
- Targets the "RAG hurts performance" failure mode (85% LLM-only vs 76% agentic)
- Cost: +1 LLM call always, +1 on ~17% disagreements
- Source: `gemma-eval-fixes` branch, commit `935f45f`

### MC Choice-Aware Research
- Make the planner and synthesizer aware of MC answer choices
- Instead of researching blind, target research toward distinguishing between options
- Specific to bar exam MC format — may not generalize to open-ended questions
- Source: `gemma-eval-fixes` branch, commit `065f005`

### True Parallel Execution (ThreadPoolExecutor)
- Current "parallel" executor runs steps sequentially in a loop
- Switch to `concurrent.futures.ThreadPoolExecutor` for actual parallelism
- Would cut wall-clock time by ~Nx for N independent steps
- Risk: GPU contention on embedding model, need to test throughput
- Code is already stubbed out (commented) in `_execute_step_with_escalation`

### Synthesizer-Driven Replanning (Outer Loop)
- Synthesizer checks completeness and can send back for more research
- Already implemented but undertested — need cases where it actually fires
- Max 3 rounds hardcoded

## Retrieval Ideas

### Aspect-Based Query Strategy
- Tested: generates rule/exception/application queries per sub-question
- Avg cross-encoder score 6.04 vs current approach's 2.97 (2x improvement)
- Retrieves more MBE/study material vs caselaw
- Maps well to parallel agent design — each agent targets a different dimension
- Source: `eval/eval_query_strategies.py`, logs in `logs/eval_query_strategies_10.txt`

### State Filtering for Housing Collection
- Housing corpus has 1.84M docs across 51 jurisdictions
- Without state filter, retrieval pulls from wrong states
- Router could extract state from question and pass as ChromaDB metadata filter
- Deferred — focusing on barqa for now

### Agentic Retrieval (No Vector DB)
- From Supermemory ASMR paper: replace vector search with LLM agents that actively read/search stored findings
- Parallel observer agents for ingestion, parallel search agents for retrieval
- Interesting for temporal/contradictory data — overkill for static legal corpus
- Source: user-shared article on ASMR technique

## Eval Infrastructure Ideas

### Multi-Model Eval Framework
- Kick off same eval across multiple models (DeepSeek, Gemma 4B, Llama, etc.)
- Compare accuracy, latency, cost per model
- Need lightweight profile system to swap models easily

### Varied Question Sets
- Current eval uses 100 random barqa questions — all MC format
- Need: open-ended legal questions, multi-hop reasoning, jurisdiction-specific
- HousingQA is Yes/No format — different eval adapter needed

### Incremental Eval Checkpointing
- Save results after each query so interrupted runs can resume
- Already implemented in `gemma-eval-fixes` branch (commit `20a273c`)

## Query Rewriting Strategies Tested

| Strategy | Avg Score | Gold Recall | Notes |
|---|---|---|---|
| raw (no rewrite) | -2.3 | 0% | Baseline — terrible |
| current (synonyms) | 3.0 | 0% | Our approach pre-aspect |
| aspect (rule/exc/app) | 6.0 | 0% | BEST overall scores |
| decompose (concepts) | 3.9 | 0% | Inconsistent |
| abstract (doctrine) | 2.5 | 10% | Only one to find gold |

## Performance Baselines (N=100, DeepSeek)

| Method | Accuracy | Gold Recall@5 | Latency/q |
|---|---|---|---|
| LLM-only | 85% | n/a | 17s |
| Golden passage | 77% | n/a | 4s |
| RAG + rewrite | 80% | 8% | 34s |
| Simple RAG | 70% | 0% | 21s |
| Agentic RAG | 76% | 15% | 82s |
