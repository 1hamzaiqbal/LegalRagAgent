import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from rag_utils import retrieve_documents # Add Chroma retriever

# 1. Core Data Models

class PlanStep(BaseModel):
    step_id: float
    status: Literal["pending", "completed", "failed"] = "pending"
    phase: str
    question: str
    execution: Dict[str, Any] = Field(default_factory=dict) # answer, sources, confidence_score
    expectation: Dict[str, Any] = Field(default_factory=dict) # outcome, is_aligned
    deviation_analysis: Optional[str] = None

import os

def load_skill_instructions(skill_name: str) -> str:
    """Loads markdown instructions from a skills directory based on skill name."""
    skill_path = os.path.join("skills", f"{skill_name}.md")
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[WARNING: Instruction file '{skill_path}' not found!]"

class AgentState(TypedDict):
    global_objective: str
    planning_table: List[PlanStep]
    contingency_plan: str

# 2. Skill Placeholders

def skill_plan_synthesis(objective: str) -> str:
    """Loads plan synthesis instructions."""
    return load_skill_instructions("plan_synthesis")

def skill_query_rewrite(question: str) -> str:
    """Loads query rewrite instructions."""
    return load_skill_instructions("query_rewrite")

def skill_retrieve_evidence(query: str) -> List[str]:
    """Retrieve evidence documents from ChromaDB."""
    docs = retrieve_documents(query, k=5)
    return [doc.page_content for doc in docs]

def skill_synthesize_answer(question: str, evidence: List[str]) -> str:
    """Loads answer synthesis instructions."""
    return load_skill_instructions("synthesize_answer")

def calculate_bge_confidence(query: str, docs: List[str]) -> float:
    """Stub to calculate confidence. Returns a low score for step 1 to trigger replanning."""
    # Let's artificially make step 1 fail (e.g. if query contains 'main components')
    if "main components" in query.lower():
        return 0.5 # Below 0.7 threshold
    return 0.9

# 3. Nodes

def planner_node(state: AgentState) -> AgentState:
    print("\n--- PLANNER NODE ---")
    if not state.get("planning_table"):
        print(f"Initializing plan for objective: {state['global_objective']}")
        
        # Load skill instructions (to be used by an LLM in a real implementation)
        instructions = skill_plan_synthesis(state["global_objective"])
        print(f"(Using instructions: {instructions[:50]}...)")
        
        # For now, manually create steps as the LLM isn't plugged in yet
        steps = [
            PlanStep(
                step_id=1.0,
                phase="Initial Research",
                question=f"What are the main components of: {state['global_objective']}?",
                expectation={"outcome": "Identify key entities and regulations."}
            ),
             PlanStep(
                step_id=2.0,
                phase="Deep Dive",
                question="Analyze the specific impacts on supply chains.",
                expectation={"outcome": "Detailed impact assessment."}
            )
        ]
        
        state["planning_table"] = steps
    else:
        print("Plan already exists.")
    
    _print_table(state["planning_table"])
    return state

def executor_node(state: AgentState) -> AgentState:
    print("\n--- EXECUTOR NODE ---")
    table = state["planning_table"]
    
    # Identify first pending step
    for step in table:
        if step.status == "pending":
            print(f"Executing step {step.step_id}: {step.question}")
            
            # Load instructions
            rewrite_instructions = skill_query_rewrite(step.question)
            synth_instructions = skill_synthesize_answer(step.question, [])
            
            # In a real implementation, you would pass these instructions to an LLM
            # Here we mock the LLM output but still show we loaded the instructions
            optimized_query = f"[Used {rewrite_instructions[:20]}...] Optimized query for: {step.question}"
            evidence = skill_retrieve_evidence(optimized_query)
            answer = f"[Used {synth_instructions[:20]}...] Based on {len(evidence)} documents, the answer is..."
            confidence = calculate_bge_confidence(optimized_query, evidence)
            
            step.execution = {
                "answer": answer,
                "sources": evidence,
                "confidence_score": confidence
            }
            print(f"Confidence score: {confidence}")
            break # Only execute one step per node run
            
    _print_table(state["planning_table"])
    return state

def evaluator_node(state: AgentState) -> AgentState:
    print("\n--- EVALUATOR NODE ---")
    table = state["planning_table"]
    
    for step in table:
        if step.status == "pending" and "confidence_score" in step.execution:
            score = step.execution["confidence_score"]
            if score >= 0.7:
                print(f"Step {step.step_id} executed successfully (Score: {score}). Marking completed.")
                step.status = "completed"
                step.expectation["is_aligned"] = True
            else:
                print(f"Step {step.step_id} failed confidence threshold (Score: {score}). Injecting new step.")
                step.status = "failed"
                step.expectation["is_aligned"] = False
                step.deviation_analysis = "Insufficient evidence retrieved for the query."
                
                # Dynamically inject new sub-step
                new_step_id = round(step.step_id + 0.1, 1)
                new_step = PlanStep(
                    step_id=new_step_id,
                    phase="Clarification Retrieval",
                    question=f"Clarify gaps for: {step.question}",
                    expectation={"outcome": "Fill missing information from failed step."}
                )
                
                # Insert right after the failed step
                insert_idx = table.index(step) + 1
                table.insert(insert_idx, new_step)
                print(f"Injected step {new_step_id}: {new_step.question}")
                
            break # Only evaluate the step that was just executed

    _print_table(state["planning_table"])
    return state

def _print_table(table: List[PlanStep]):
    print("\nCurrent Planning Table:")
    for s in table:
        print(f"  [{s.status.upper()}] Step {s.step_id}: {s.question} | Executed: {'Yes' if 'confidence_score' in s.execution else 'No'}")
    print("-" * 40)

# 4. Routing

def route_after_evaluator(state: AgentState) -> Literal["executor_node", "__end__"]:
    table = state.get("planning_table", [])
    
    # Hard failure limit: prevent infinite loops by capping total steps
    if len(table) > 10:
        print("Hard failure limit hit (too many steps). Routing to END.")
        return "__end__"
        
    # Check if there are any pending steps
    has_pending = any(step.status == "pending" for step in table)
    
    if has_pending:
        print("Routing back to EXECUTOR...")
        return "executor_node"
    
    print("All steps completed. Routing to END.")
    return "__end__"

# 5. Graph Topology

def build_graph() -> Any:
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("executor_node", executor_node)
    workflow.add_node("evaluator_node", evaluator_node)
    
    workflow.set_entry_point("planner_node")
    workflow.add_edge("planner_node", "executor_node")
    workflow.add_edge("executor_node", "evaluator_node")
    
    workflow.add_conditional_edges(
        "evaluator_node",
        route_after_evaluator
    )
    
    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = build_graph()
    
    initial_state = {
        "global_objective": "Analyze the impact of 2026 energy regulations on semiconductor supply chains.",
        "planning_table": [],
        "contingency_plan": "If unable to find specific 2026 regulations, abstract to general pending legislation."
    }
    
    print("\nStarting AntiGravity Execution...")
    try:
        # Run the graph
        for output in app.stream(initial_state):
            # The nodes themselves print their outputs.
            pass
    except Exception as e:
        print(f"Error during execution: {e}")
