"""Compatibility facade for runtime execution helpers.

The implementation is split by mental model:
- `legal_rag.nodes` for router/planner/synthesizer/replanner graph nodes
- `legal_rag.step_executor` for round execution and escalation
- `legal_rag.retrieval` for query construction and retrieval-backed step helpers
"""

from __future__ import annotations

from .core import _llm_call
from .nodes import (
    planner_node,
    replanner_node,
    route_after_synthesizer,
    router_node,
    synthesizer_node,
)
from .prompts import COLLECTIONS_REGISTRY, COMPLETENESS_CHECK_PROMPT, ROUTER_PROMPT, inline_prompt_versions
from .retrieval import _execute_direct_answer, _execute_rag_search, _execute_web_search, web_search
from .step_executor import (
    StepExecutionResult,
    _call_judge,
    _execute_step_with_escalation,
    execute_round_node,
    route_after_execution_round,
)


def _inline_prompt_versions(profile):
    """Compatibility wrapper retained for older imports."""
    return inline_prompt_versions(profile)


__all__ = [
    "COLLECTIONS_REGISTRY",
    "COMPLETENESS_CHECK_PROMPT",
    "ROUTER_PROMPT",
    "StepExecutionResult",
    "_execute_direct_answer",
    "_execute_rag_search",
    "_execute_web_search",
    "_call_judge",
    "_llm_call",
    "_execute_step_with_escalation",
    "_inline_prompt_versions",
    "execute_round_node",
    "planner_node",
    "replanner_node",
    "route_after_execution_round",
    "route_after_synthesizer",
    "router_node",
    "synthesizer_node",
    "web_search",
]
