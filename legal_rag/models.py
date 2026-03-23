"""Core models used by the Legal RAG runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

ACTION_RAG = "rag_search"
ACTION_WEB = "web_search"
ACTION_DIRECT = "direct_answer"

STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

SUPPORT_PRIMARY = "primary"
SUPPORT_SUPPORT_ONLY = "support_only"

TERMINAL_ANSWERED = "answered"
TERMINAL_MAX_ROUNDS = "max_rounds"
TERMINAL_STALLED = "stalled"
TERMINAL_LOOP_DISABLED = "loop_disabled"
TERMINAL_PARSE_FAILURE = "parse_failure"

STEP_MODE_SEQUENTIAL = "sequential"
STEP_MODE_PARALLEL = "parallel"

RAG_STRATEGY_STANDARD = "standard"
RAG_STRATEGY_ASPECT = "aspect"

PROFILE_KIND_BASELINE_LLM = "baseline_llm"
PROFILE_KIND_BASELINE_RAG = "baseline_rag"
PROFILE_KIND_FULL = "full"


class PlanningStep(BaseModel):
    """A single research step in the planning table."""

    step_id: int
    sub_question: str
    authority_target: str = ""
    retrieval_hints: List[str] = Field(default_factory=list)
    action_type: Literal["direct_answer", "rag_search", "web_search"] = ACTION_RAG
    max_retries: int = 2
    rewrite_attempt: int = 0
    status: Literal["pending", "completed", "failed"] = STATUS_PENDING
    result: str = ""
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)
    retry_of: Optional[int] = None
    judge_verdict: Optional[Dict[str, Any]] = None
    result_origin: str = ""
    support_level: Literal["primary", "support_only"] = SUPPORT_PRIMARY


class LegalAgentState(TypedDict):
    """Shared state propagated across graph nodes."""

    agent_metadata: Dict[str, Any]
    inputs: Dict[str, Any]
    run_config: Dict[str, Any]
    profile: Dict[str, Any]
    collections: List[str]
    planning_table: List[PlanningStep]
    evidence_store: List[Dict[str, Any]]
    final_answer: str
    audit_log: List[Dict[str, Any]]
    completeness_verdict: Dict[str, Any]
    parallel_round: int
    replanning_brief: str
    step_traces: List[Dict[str, Any]]
    run_artifact: Dict[str, Any]


@dataclass(frozen=True)
class ExperimentProfile:
    """Runtime configuration for a named experiment variant."""

    name: str
    kind: Literal["baseline_llm", "baseline_rag", "full"]
    description: str = ""
    use_router: bool = True
    use_planner: bool = True
    step_execution_mode: Literal["sequential", "parallel"] = STEP_MODE_PARALLEL
    rag_strategy: Literal["standard", "aspect"] = RAG_STRATEGY_STANDARD
    use_bm25: bool = True
    use_query_rewrite: bool = True
    use_judge: bool = True
    use_completeness_loop: bool = True
    allow_web_search: bool = True
    allow_direct_answer_fallback: bool = True
    write_run_artifact: bool = True

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "use_router": self.use_router,
            "use_planner": self.use_planner,
            "step_execution_mode": self.step_execution_mode,
            "rag_strategy": self.rag_strategy,
            "use_bm25": self.use_bm25,
            "use_query_rewrite": self.use_query_rewrite,
            "use_judge": self.use_judge,
            "use_completeness_loop": self.use_completeness_loop,
            "allow_web_search": self.allow_web_search,
            "allow_direct_answer_fallback": self.allow_direct_answer_fallback,
            "write_run_artifact": self.write_run_artifact,
        }


@dataclass
class ExecutionResult:
    """Normalized result returned by runtime and baseline runners."""

    profile: ExperimentProfile
    final_answer: str
    planning_table: List[PlanningStep] = field(default_factory=list)
    evidence_store: List[Dict[str, Any]] = field(default_factory=list)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    completeness_verdict: Dict[str, Any] = field(default_factory=dict)
    collections: List[str] = field(default_factory=list)
    parallel_round: int = 1
    agent_metadata: Dict[str, Any] = field(default_factory=dict)
    run_artifact: Dict[str, Any] = field(default_factory=dict)
    artifact_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


DEMO_QUERIES: Dict[str, str] = {
    "simple": "What are the elements of a negligence claim?",
    "multi_hop": (
        "A police officer stopped a vehicle without reasonable suspicion, found drugs "
        "during a warrantless search, and the defendant seeks suppression. "
        "What are the defendant's strongest Fourth Amendment arguments and how would "
        "a court analyze application of the exclusionary rule?"
    ),
    "medium": "What is the standard for granting a preliminary injunction?",
}
