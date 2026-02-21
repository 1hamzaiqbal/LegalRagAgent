import json
import os
import re
import sys
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from llm_config import get_llm
from rag_utils import retrieve_documents, compute_confidence

# ---------------------------------------------------------------------------
# 1. Core Data Models
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    step_id: float
    status: Literal["pending", "completed", "failed"] = "pending"
    phase: str
    question: str
    execution: Dict[str, Any] = Field(default_factory=dict)
    expectation: Dict[str, Any] = Field(default_factory=dict)
    deviation_analysis: Optional[str] = None


class AgentState(TypedDict):
    global_objective: str
    planning_table: List[PlanStep]
    contingency_plan: str
    query_type: str            # "simple" or "multi_hop"
    final_cited_answer: str    # aggregated output


# ---------------------------------------------------------------------------
# 2. Skill Loaders
# ---------------------------------------------------------------------------

def load_skill_instructions(skill_name: str) -> str:
    """Loads markdown instructions from the skills/ directory."""
    skill_path = os.path.join("skills", f"{skill_name}.md")
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[WARNING: Instruction file '{skill_path}' not found!]"


def _parse_json(text: str) -> Any:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _llm_call(system_prompt: str, user_prompt: str) -> str:
    """Make an LLM call and return the text response."""
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# 3. Skill Wrappers
# ---------------------------------------------------------------------------

def skill_classify_and_route(objective: str) -> Dict[str, str]:
    """Classify objective as simple or multi_hop."""
    instructions = load_skill_instructions("classify_and_route")
    user_msg = f"Objective: {objective}"
    raw = _llm_call(instructions, user_msg)
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and "query_type" in parsed:
        return parsed
    return {"query_type": "multi_hop", "reasoning": "Fallback — could not parse classifier output"}


def skill_plan_synthesis(objective: str, query_type: str) -> List[Dict]:
    """Generate a plan as a list of step dicts."""
    instructions = load_skill_instructions("plan_synthesis")
    user_msg = f"Objective: {objective}\nQuery type: {query_type}"
    raw = _llm_call(instructions, user_msg)
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, list):
        return parsed
    # Fallback: single step
    return [{
        "step_id": 1.0,
        "phase": "Direct Research",
        "question": objective,
        "expectation": "Direct answer to the question",
    }]


def skill_query_rewrite(question: str) -> str:
    """Rewrite a question into an optimized retrieval query."""
    instructions = load_skill_instructions("query_rewrite")
    return _llm_call(instructions, f"Question: {question}").strip()


def skill_synthesize_answer(question: str, evidence: List[str]) -> str:
    """Synthesize an answer from retrieved evidence."""
    instructions = load_skill_instructions("synthesize_answer")
    evidence_text = "\n\n".join(
        f"[Passage {i+1}]: {text}" for i, text in enumerate(evidence)
    )
    user_msg = f"Question: {question}\n\nEvidence:\n{evidence_text}"
    return _llm_call(instructions, user_msg).strip()


def skill_ground_and_cite(answer: str, evidence: List[str]) -> str:
    """Ground an answer and add source citations."""
    instructions = load_skill_instructions("ground_and_cite")
    evidence_text = "\n\n".join(
        f"[Passage {i+1}]: {text}" for i, text in enumerate(evidence)
    )
    user_msg = f"Answer:\n{answer}\n\nEvidence:\n{evidence_text}"
    return _llm_call(instructions, user_msg).strip()


# ---------------------------------------------------------------------------
# 4. Graph Nodes
# ---------------------------------------------------------------------------

def classifier_node(state: AgentState) -> AgentState:
    """Classify the objective to determine routing."""
    print("\n--- CLASSIFIER NODE ---")
    objective = state["global_objective"]
    print(f"Classifying: {objective}")

    result = skill_classify_and_route(objective)
    state["query_type"] = result["query_type"]
    print(f"Classification: {result['query_type']} — {result.get('reasoning', '')}")
    return state


def planner_node(state: AgentState) -> AgentState:
    """Generate a research plan using the LLM."""
    print("\n--- PLANNER NODE ---")
    if not state.get("planning_table"):
        objective = state["global_objective"]
        query_type = state.get("query_type", "multi_hop")
        print(f"Generating plan for: {objective} (type: {query_type})")

        raw_steps = skill_plan_synthesis(objective, query_type)
        steps = []
        for s in raw_steps:
            steps.append(PlanStep(
                step_id=float(s.get("step_id", len(steps) + 1)),
                phase=s.get("phase", "Research"),
                question=s.get("question", objective),
                expectation={"outcome": s.get("expectation", "")},
            ))

        state["planning_table"] = steps
        print(f"Generated {len(steps)} plan steps")
    else:
        print("Plan already exists.")

    _print_table(state["planning_table"])
    return state


def executor_node(state: AgentState) -> AgentState:
    """Execute the next pending plan step with real LLM calls."""
    print("\n--- EXECUTOR NODE ---")
    table = state["planning_table"]

    for step in table:
        if step.status == "pending":
            print(f"Executing step {step.step_id}: {step.question}")

            # 1. Query rewrite
            optimized_query = skill_query_rewrite(step.question)
            print(f"  Optimized query: {optimized_query[:80]}...")

            # 2. Retrieve
            docs = retrieve_documents(optimized_query, k=5)
            evidence = [doc.page_content for doc in docs]
            print(f"  Retrieved {len(evidence)} passages")

            # 3. Synthesize
            answer = skill_synthesize_answer(step.question, evidence)
            print(f"  Synthesized answer ({len(answer)} chars)")

            # 4. Ground and cite
            cited_answer = skill_ground_and_cite(answer, evidence)
            print(f"  Grounded with citations ({len(cited_answer)} chars)")

            # 5. Compute confidence
            confidence = compute_confidence(optimized_query, docs)
            print(f"  Confidence score: {confidence:.3f}")

            step.execution = {
                "answer": answer,
                "cited_answer": cited_answer,
                "optimized_query": optimized_query,
                "sources": evidence,
                "confidence_score": confidence,
            }
            break  # One step per node run

    _print_table(state["planning_table"])
    return state


def evaluator_node(state: AgentState) -> AgentState:
    """Evaluate executed steps and decide whether to continue or inject sub-steps."""
    print("\n--- EVALUATOR NODE ---")
    table = state["planning_table"]

    for step in table:
        if step.status == "pending" and "confidence_score" in step.execution:
            score = step.execution["confidence_score"]
            if score >= 0.7:
                print(f"Step {step.step_id} PASSED (score: {score:.3f}). Marking completed.")
                step.status = "completed"
                step.expectation["is_aligned"] = True
            else:
                print(f"Step {step.step_id} FAILED (score: {score:.3f}). Injecting sub-step.")
                step.status = "failed"
                step.expectation["is_aligned"] = False
                step.deviation_analysis = "Insufficient evidence — retrieval confidence below threshold."

                new_step_id = round(step.step_id + 0.1, 1)
                new_step = PlanStep(
                    step_id=new_step_id,
                    phase="Clarification Retrieval",
                    question=f"Clarify gaps for: {step.question}",
                    expectation={"outcome": "Fill missing information from failed step."},
                )
                insert_idx = table.index(step) + 1
                table.insert(insert_idx, new_step)
                print(f"Injected step {new_step_id}")
            break

    # Aggregate final answer when all steps are done
    all_done = all(s.status in ("completed", "failed") for s in table)
    if all_done:
        completed_answers = [
            s.execution.get("cited_answer", s.execution.get("answer", ""))
            for s in table
            if s.status == "completed" and s.execution
        ]
        if completed_answers:
            state["final_cited_answer"] = "\n\n---\n\n".join(completed_answers)
            print("Aggregated final cited answer from completed steps.")

    _print_table(state["planning_table"])
    return state


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------

def _print_table(table: List[PlanStep]):
    print("\nCurrent Planning Table:")
    for s in table:
        executed = "Yes" if "confidence_score" in s.execution else "No"
        score = f" (score: {s.execution['confidence_score']:.3f})" if "confidence_score" in s.execution else ""
        print(f"  [{s.status.upper()}] Step {s.step_id}: {s.question} | Executed: {executed}{score}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# 6. Routing
# ---------------------------------------------------------------------------

def route_after_evaluator(state: AgentState) -> Literal["executor_node", "__end__"]:
    table = state.get("planning_table", [])

    if len(table) > 10:
        print("Hard failure limit hit (>10 steps). Routing to END.")
        return "__end__"

    has_pending = any(step.status == "pending" for step in table)
    if has_pending:
        print("Routing back to EXECUTOR...")
        return "executor_node"

    print("All steps done. Routing to END.")
    return "__end__"


# ---------------------------------------------------------------------------
# 7. Graph Topology
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("classifier_node", classifier_node)
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("executor_node", executor_node)
    workflow.add_node("evaluator_node", evaluator_node)

    workflow.set_entry_point("classifier_node")
    workflow.add_edge("classifier_node", "planner_node")
    workflow.add_edge("planner_node", "executor_node")
    workflow.add_edge("executor_node", "evaluator_node")

    workflow.add_conditional_edges(
        "evaluator_node",
        route_after_evaluator,
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# 8. Demo Queries
# ---------------------------------------------------------------------------

DEMO_QUERIES = {
    "simple": {
        "objective": "What are the elements of a negligence claim?",
        "contingency": "If passages lack negligence specifics, broaden to general tort liability.",
    },
    "multi_hop": {
        "objective": (
            "A police officer pulls over a driver for a broken taillight, smells marijuana, "
            "and searches the car without a warrant, finding illegal drugs. The driver invokes "
            "the Fifth Amendment at trial. What are the driver's constitutional rights and what "
            "legal standards apply to the search and the testimony?"
        ),
        "contingency": "If constitutional search passages are sparse, retrieve 4th and 5th Amendment separately.",
    },
    "medium": {
        "objective": (
            "A plaintiff seeks a preliminary injunction to prevent a competitor from using a "
            "similar trademark. What legal standard must the court apply, and what factors "
            "are considered?"
        ),
        "contingency": "If injunction passages are limited, fall back to general equitable remedies.",
    },
}


if __name__ == "__main__":
    # Select demo query via CLI arg: python main.py [simple|multi_hop|medium]
    query_key = sys.argv[1] if len(sys.argv) > 1 else "simple"
    if query_key not in DEMO_QUERIES:
        print(f"Unknown query key '{query_key}'. Choose from: {', '.join(DEMO_QUERIES)}")
        sys.exit(1)

    demo = DEMO_QUERIES[query_key]
    print(f"\n{'='*60}")
    print(f"Demo mode: {query_key}")
    print(f"Objective: {demo['objective']}")
    print(f"{'='*60}")

    app = build_graph()

    initial_state = {
        "global_objective": demo["objective"],
        "planning_table": [],
        "contingency_plan": demo["contingency"],
        "query_type": "",
        "final_cited_answer": "",
    }

    print("\nStarting Legal RAG Agent...")
    final_state = None
    try:
        for output in app.stream(initial_state):
            # Capture the last state from the stream
            for node_name, node_state in output.items():
                final_state = node_state
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

    # Print final answer
    print(f"\n{'='*60}")
    print("FINAL CITED ANSWER")
    print(f"{'='*60}")
    if final_state and final_state.get("final_cited_answer"):
        print(final_state["final_cited_answer"])
    else:
        print("(No final answer produced — check step outputs above)")
