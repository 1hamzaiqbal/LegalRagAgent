# Architecture

## Current Implementation (5 Skills)

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
    EVAL -->|confidence >= 0.7| CHECK{Pending steps?}
    EVAL -->|confidence < 0.7| INJECT[Inject sub-step]
    INJECT --> CHECK
    CHECK -->|Yes| EXEC
    CHECK -->|No| AGG[Aggregate Final Answer]
    AGG --> DONE([End])

    style CLASSIFY fill:#4CAF50,color:#fff
    style PLAN fill:#4CAF50,color:#fff
    style QR fill:#4CAF50,color:#fff
    style SYN fill:#4CAF50,color:#fff
    style GC fill:#4CAF50,color:#fff
    style RET fill:#2196F3,color:#fff
    style EVAL fill:#FF9800,color:#fff
```

**Legend**: Green = LLM skill (implemented), Blue = retrieval (ChromaDB), Orange = evaluation logic

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
| 8 | VerifyAnswer | Planned | Cross-checks final answer against retrieved evidence for consistency |
| 9 | DetectPromptInjection | Planned | Screens user input for adversarial prompts before processing |
| 10 | MemoryWriteBack | Planned | Persists successful query-answer pairs for future retrieval |
| 11 | ObservabilityAndCostControl | Planned | Tracks token usage, latency, and cost per query |
| 12 | OrchestratePlanExecuteRAG | Planned | Top-level orchestrator managing the full plan-execute-evaluate loop |

## State Schema

```
AgentState:
  global_objective: str          # User's legal research question
  planning_table: List[PlanStep] # Steps with status, execution results, confidence
  contingency_plan: str          # Fallback strategy if retrieval fails
  query_type: str                # "simple" or "multi_hop" (set by classifier)
  final_cited_answer: str        # Aggregated output with citations
```
