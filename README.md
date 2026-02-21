# LegalRagAgent
Prototype implementation for Agentic Legal Rag System

## Required Skills

The LangGraph agent dynamically loads agent instructions (skills) from markdown files located in the `skills/` directory.

In order for the agent to function fully, you need to create the following three prompt files:

1. **`skills/plan_synthesis.md`**
   - **Used in:** `planner_node`
   - **Purpose:** Guides the LLM on how to break down the user's overall `global_objective` into a logical step-by-step research plan (generating the `PlanStep` items for the `planning_table`).

2. **`skills/query_rewrite.md`**
   - **Used in:** `executor_node`
   - **Purpose:** Instructs the LLM on how to take the specific `question` of the current step and reformulate/optimize it into a highly effective search query for the ChromaDB retriever.

3. **`skills/synthesize_answer.md`**
   - **Used in:** `executor_node`
   - **Purpose:** Directs the LLM on how to read the retrieved legal passages alongside the step's question in order to synthesize a comprehensive, grounded answer for that specific step.
