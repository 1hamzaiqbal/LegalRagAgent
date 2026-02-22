# Architecture

## Current Implementation (6 Skills + Adaptive Replanning)

```mermaid
flowchart TD
    START([Start]) --> CLASSIFY[ClassifyAndRoute]
    CLASSIFY --> PLAN[PlanSynthesis]
    PLAN --> EXEC[ExecutePlan]

    subgraph EXEC_SUB [Executor — per step]
        QR[QueryRewrite] --> RET[RetrieveEvidence]
        RET --> SYN[SynthesizeAnswer]
        SYN --> GC[GroundAndCite]
    end

    EXEC --> EXEC_SUB
    EXEC_SUB --> EVAL{Evaluator}
    EVAL -->|pending steps| EXEC
    EVAL -->|multi_hop, all done| REPLAN[AdaptiveReplan]
    EVAL -->|simple done / limit| DONE([End])
    REPLAN -->|next_step / retry| EXEC
    REPLAN -->|complete| DONE

    style CLASSIFY fill:#4CAF50,color:#fff
    style PLAN fill:#4CAF50,color:#fff
    style QR fill:#4CAF50,color:#fff
    style SYN fill:#4CAF50,color:#fff
    style GC fill:#4CAF50,color:#fff
    style REPLAN fill:#4CAF50,color:#fff
    style RET fill:#2196F3,color:#fff
    style EVAL fill:#FF9800,color:#fff
```

**Legend**: Green = LLM skill (implemented), Blue = retrieval (ChromaDB), Orange = evaluation logic

**Key change**: The replanner node replaces hardcoded sub-step injection. For multi_hop queries, the planner emits only the first step. After each step completes, the replanner decides what to research next based on accumulated evidence ("model creates 1 query, gets answers, then creates next query").

**Caching**: The LLM client is a singleton (`@lru_cache`), and skill prompt files are cached after first read. When using providers that support prefix caching (DeepSeek, vLLM with `--enable-prefix-caching`, OpenAI), cache-hit metrics are logged automatically per call.

---

## Full 12-Skill Vision

```mermaid
flowchart TD
    START([Start]) --> DPI[DetectPromptInjection]
    DPI --> CAR[ClassifyAndRoute]
    CAR --> PS[PlanSynthesis]
    PS --> EP[ExecutePlan]

    subgraph EXEC [Execute Loop]
        QRD[QueryRewriteAndDecompose] --> RE[RetrieveEvidence]
        RE --> SSA[SynthesizeSubtaskAnswer]
        SSA --> GAC[GroundAndCite]
    end

    EP --> EXEC
    EXEC --> VA[VerifyAnswer]
    VA -->|needs retry| EP
    VA -->|verified| MWB[MemoryWriteBack]
    MWB --> OCC[ObservabilityAndCostControl]
    OCC --> ORCH[OrchestratePlanExecuteRAG]
    ORCH --> DONE([End])

    style DPI fill:#9E9E9E,color:#fff
    style CAR fill:#4CAF50,color:#fff
    style PS fill:#4CAF50,color:#fff
    style QRD fill:#4CAF50,color:#fff
    style RE fill:#2196F3,color:#fff
    style SSA fill:#4CAF50,color:#fff
    style GAC fill:#4CAF50,color:#fff
    style VA fill:#9E9E9E,color:#fff
    style MWB fill:#9E9E9E,color:#fff
    style OCC fill:#9E9E9E,color:#fff
    style ORCH fill:#9E9E9E,color:#fff
    style EP fill:#4CAF50,color:#fff
```

**Legend**: Green = built, Blue = retrieval, Grey = planned

---

## Skill Descriptions

| # | Skill | Status | Description |
|---|-------|--------|-------------|
| 1 | ClassifyAndRoute | Built | Classifies query as `simple` or `multi_hop` to determine plan complexity |
| 2 | PlanSynthesis | Built | Decomposes objective into a structured JSON plan of retrieval steps |
| 3 | ExecutePlan | Built | Orchestrates per-step execution: rewrite, retrieve, synthesize, cite |
| 4 | QueryRewriteAndDecompose | Built | Optimizes questions into dense legal retrieval queries (MBE/MEE vocab) |
| 5 | RetrieveEvidence | Built | Top-k vector retrieval from ChromaDB (HuggingFace all-MiniLM-L6-v2) |
| 6 | SynthesizeSubtaskAnswer | Built | Synthesizes grounded answers with Rule/Elements/Exceptions structure |
| 7 | GroundAndCite | Built | Audits answers for grounding, adds `[Source N]` citations, flags gaps |
| 8 | AdaptiveReplan | Built | Decides next research step based on accumulated evidence (multi_hop only) |
| 9 | RetrieveLegalPassages | Built | `@tool`-decorated wrapper for retrieval, bindable via `llm.bind_tools()` |
| 10 | VerifyAnswer | Planned | Cross-checks final answer against retrieved evidence for consistency |
| 11 | DetectPromptInjection | Planned | Screens user input for adversarial prompts before processing |
| 12 | MemoryWriteBack | Planned | Persists successful query-answer pairs for future retrieval |
| 13 | ObservabilityAndCostControl | Planned | Tracks token usage, latency, and cost per query |

## External Tool Placeholders (external_tools.py)

| Tool | Description |
|------|-------------|
| `web_search` | Placeholder for web search via teammate's Playwright API |
| `web_scrape` | Placeholder for page scraping via Playwright |
| `external_api_call` | Generic placeholder for teammate's API wrapper |

Configured via `EXTERNAL_TOOLS_BASE_URL` and `EXTERNAL_TOOLS_API_KEY` env vars. All decorated with `@tool` for `llm.bind_tools()` integration.

## State Schema

```
AgentState:
  global_objective: str                # User's legal research question
  planning_table: List[PlanStep]       # Steps with status, execution results, confidence
  contingency_plan: str                # Fallback strategy if retrieval fails
  query_type: str                      # "simple" or "multi_hop" (set by classifier)
  final_cited_answer: str              # Aggregated output with citations
  accumulated_context: List[Dict]      # Step summaries for replanner (question, answer, confidence, status)
  iteration_count: int                 # Cycle counter for loop guard (max 6)
```
