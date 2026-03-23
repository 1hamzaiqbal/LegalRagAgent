"""Experiment profile presets used by runtime and evals."""

from __future__ import annotations

from typing import Dict, Iterable

from .models import (
    ExperimentProfile,
    PROFILE_KIND_BASELINE_LLM,
    PROFILE_KIND_BASELINE_RAG,
    PROFILE_KIND_FULL,
    RAG_STRATEGY_ASPECT,
    RAG_STRATEGY_STANDARD,
    STEP_MODE_PARALLEL,
    STEP_MODE_SEQUENTIAL,
)

PROFILES: Dict[str, ExperimentProfile] = {
    "llm_only": ExperimentProfile(
        name="llm_only",
        kind=PROFILE_KIND_BASELINE_LLM,
        description="Direct model answer with no retrieval pipeline.",
        use_router=False,
        use_planner=False,
        step_execution_mode=STEP_MODE_SEQUENTIAL,
        use_query_rewrite=False,
        use_judge=False,
        use_completeness_loop=False,
        allow_web_search=False,
        allow_direct_answer_fallback=False,
    ),
    "simple_rag": ExperimentProfile(
        name="simple_rag",
        kind=PROFILE_KIND_BASELINE_RAG,
        description="Single retrieval pass with no planner, judge, or completeness loop.",
        use_router=False,
        use_planner=False,
        step_execution_mode=STEP_MODE_SEQUENTIAL,
        rag_strategy=RAG_STRATEGY_STANDARD,
        use_query_rewrite=False,
        use_judge=False,
        use_completeness_loop=False,
        allow_web_search=False,
        allow_direct_answer_fallback=False,
    ),
    "rewrite_rag": ExperimentProfile(
        name="rewrite_rag",
        kind=PROFILE_KIND_BASELINE_RAG,
        description="Single retrieval pass with synonym-style query rewriting.",
        use_router=False,
        use_planner=False,
        step_execution_mode=STEP_MODE_SEQUENTIAL,
        rag_strategy=RAG_STRATEGY_STANDARD,
        use_query_rewrite=True,
        use_judge=False,
        use_completeness_loop=False,
        allow_web_search=False,
        allow_direct_answer_fallback=False,
    ),
    "full_seq": ExperimentProfile(
        name="full_seq",
        kind=PROFILE_KIND_FULL,
        description="Sequential full pipeline with planner, judge, and completeness loop.",
        step_execution_mode=STEP_MODE_SEQUENTIAL,
        rag_strategy=RAG_STRATEGY_STANDARD,
        use_bm25=False,
    ),
    "full_parallel": ExperimentProfile(
        name="full_parallel",
        kind=PROFILE_KIND_FULL,
        description="Round-safe parallel full pipeline with standard retrieval strategy.",
        step_execution_mode=STEP_MODE_PARALLEL,
        rag_strategy=RAG_STRATEGY_STANDARD,
        use_bm25=False,
    ),
    "full_parallel_aspect": ExperimentProfile(
        name="full_parallel_aspect",
        kind=PROFILE_KIND_FULL,
        description="Round-safe parallel full pipeline with aspect-specialized retrieval.",
        step_execution_mode=STEP_MODE_PARALLEL,
        rag_strategy=RAG_STRATEGY_ASPECT,
        use_bm25=False,
    ),
}

PROFILE_NAMES = tuple(PROFILES.keys())


def get_profile(profile: str | ExperimentProfile) -> ExperimentProfile:
    """Resolve a profile name or return the profile unchanged."""
    if isinstance(profile, ExperimentProfile):
        return profile
    if profile not in PROFILES:
        raise KeyError(f"Unknown experiment profile: {profile}")
    return PROFILES[profile]


def list_profiles() -> Iterable[ExperimentProfile]:
    """Return all known profile presets."""
    return PROFILES.values()
