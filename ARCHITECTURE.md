# Architecture

## Current Implementation (7 Skills + 9-Node Graph)

```mermaid
flowchart TD
    START([Start]) --> DPI[DetectPromptInjection]
    DPI -->|safe| CLASSIFY[ClassifyAndRoute]
    DPI -->|adversarial| OBS
    CLASSIFY --> PLAN[PlanSynthesis]
    PLAN -->|memory hit| MWB[MemoryWriteBack]
    PLAN -->|no hit| EXEC[ExecutePlan]

    subgraph EXEC_SUB [Executor — per step]
        QR[QueryRewrite + MultiQuery] --> RET[RetrieveEvidence]
        RET --> SC[SynthesizeAndCite]
    end

    EXEC --> EXEC_SUB
    EXEC_SUB --> EVAL{Evaluator}
    EVAL -->|pending steps| EXEC
    EVAL -->|multi_hop, all done, no prior correction| REPLAN[AdaptiveReplan]
    EVAL -->|simple done / limit / correction done| VA[VerifyAnswer]
    REPLAN -->|next_step / retry| EXEC
    REPLAN -->|complete| VA
    VA -->|not verified, first failure| EXEC
    VA -->|verified / retries exhausted| MWB
    MWB --> OBS[Observability]
    OBS --> DONE

    style DPI fill:#4CAF50,color:#fff
    style CLASSIFY fill:#4CAF50,color:#fff
    style PLAN fill:#4CAF50,color:#fff
    style QR fill:#4CAF50,color:#fff
    style SC fill:#4CAF50,color:#fff
    style REPLAN fill:#4CAF50,color:#fff
    style VA fill:#4CAF50,color:#fff
    style RET fill:#2196F3,color:#fff
    style EVAL fill:#FF9800,color:#fff
    style MWB fill:#9C27B0,color:#fff
    style OBS fill:#607D8B,color:#fff
```

**Legend**: Green = LLM skill, Blue = retrieval (ChromaDB), Orange = evaluation logic, Purple = memory, Grey = observability

**Graph topology**: 9 nodes — `detect_injection → {classifier | observability}` (safe/unsafe); `classifier → planner → {executor | memory_writeback}` (no hit/memory hit); `executor → evaluator → {executor | replanner | verify_answer}`; `replanner → {executor | verify_answer}`; `verify_answer → {executor | memory_writeback}` (1 corrective retry max); `memory_writeback → observability → END`

**Caching**: The LLM client is a singleton (`@lru_cache`), and skill prompt files are cached after first read. When using providers that support prefix caching (DeepSeek, vLLM with `--enable-prefix-caching`, OpenAI), cache-hit metrics are logged automatically per call.

**QA Memory**: Successful query-answer pairs (avg confidence >= 0.7) are persisted to a separate ChromaDB collection (`qa_memory`, cosine distance). On subsequent runs, the planner checks for cached answers before generating a plan, short-circuiting execution on high-similarity matches (>= 0.92). The higher threshold ensures only near-exact question matches are served from cache.

**Injection Check**: Skippable via `SKIP_INJECTION_CHECK=1` env var (saves 1 LLM call for eval/testing). Default ON for production safety.

**Verification**: The verify_answer skill produces a `suggested_query` — a proper legal research question — when it finds issues. If verification fails on the first attempt, a corrective step using that query runs through the standard executor pipeline, then routes directly back to verification (skipping the replanner to avoid tangential research). A second failure terminates without adding orphaned steps. Citations use unified `[Source N]` labels from the synthesize_and_cite skill.

---

## Skill Descriptions

### LLM Skills (7 prompt files in `skills/`)

| # | Skill | File | Description |
|---|-------|------|-------------|
| 1 | ClassifyAndRoute | `classify_and_route.md` | Classifies query as `simple` or `multi_hop` to determine plan complexity |
| 2 | PlanSynthesis | `plan_synthesis.md` | Decomposes objective into a structured JSON plan of retrieval steps |
| 3 | QueryRewrite | `query_rewrite.md` | Rewrites question into primary + 2 alternative queries (JSON) for multi-query retrieval |
| 4 | SynthesizeAndCite | `synthesize_and_cite.md` | Synthesizes grounded answers with inline `[Source N]` citations and source map in one pass |
| 5 | AdaptiveReplan | `adaptive_replan.md` | Decides next research step based on accumulated evidence (multi_hop only) |
| 6 | DetectPromptInjection | `detect_prompt_injection.md` | Screens user input for adversarial prompts (skippable via `SKIP_INJECTION_CHECK=1`) |
| 7 | VerifyAnswer | `verify_answer.md` | Cross-checks final answer against retrieved evidence for consistency |

### Non-LLM Nodes

| Node | Description |
|------|-------------|
| ExecutePlan | Orchestrates per-step execution: rewrite, multi-query retrieve, synthesize+cite |
| RetrieveEvidence | Two-stage retrieval: bi-encoder over-retrieve + cross-encoder rerank (ChromaDB) |
| Evaluator | Checks confidence (cosine similarity), accumulates context for replanner |
| MemoryWriteBack | Persists successful query-answer pairs for future retrieval |
| Observability | Tracks LLM calls, char usage, parse failures, step metrics, answer status |

## State Schema

```
AgentState:
  global_objective: str                # User's legal research question
  planning_table: List[PlanStep]       # Steps with status, execution results, confidence
  query_type: str                      # "simple" or "multi_hop" (set by classifier)
  final_cited_answer: str              # Aggregated output with citations
  accumulated_context: List[Dict]      # Step summaries for replanner (question, answer, confidence, status)
  iteration_count: int                 # Cycle counter for loop guard (max 6)
  injection_check: Dict[str, Any]      # {"is_safe": bool, "reasoning": str}
  verification_result: Dict[str, Any]  # {"is_verified": bool, "issues": [...], "reasoning": str}
  verification_retries: int            # Counter for verification retry attempts (max 1 corrective step)
  memory_hit: Dict[str, Any]           # {"found": bool, "answer": str, "confidence": float}
  run_metrics: Dict[str, Any]          # Aggregated metrics from observability node
```
