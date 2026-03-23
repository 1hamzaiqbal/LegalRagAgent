from legal_rag.execution import _execute_step_with_escalation
from legal_rag.models import PlanningStep
from legal_rag.profiles import get_profile


def _state(profile_name: str):
    return {
        "agent_metadata": {},
        "inputs": {"question": "Question?", "research_question": "Question?"},
        "run_config": {"max_steps": 7, "max_parallel_rounds": 3},
        "profile": get_profile(profile_name).to_state_dict(),
        "collections": ["legal_passages"],
        "planning_table": [],
        "evidence_store": [],
        "final_answer": "",
        "audit_log": [],
        "completeness_verdict": {},
        "parallel_round": 1,
        "step_traces": [],
        "run_artifact": {},
    }


def test_rag_escalation_rewrites_then_falls_back_to_direct(monkeypatch):
    profile = get_profile("full_parallel")
    step = PlanningStep(step_id=1, sub_question="orig", action_type="rag_search", max_retries=2)
    judge_results = iter(
        [
            {"sufficient": False, "reason": "miss", "missing": None, "suggested_rewrite": "rewrite me"},
            {"sufficient": False, "reason": "still miss", "missing": None, "suggested_rewrite": None},
            {"sufficient": "full", "reason": "now okay", "missing": None, "suggested_rewrite": None},
        ]
    )

    monkeypatch.setattr("legal_rag.execution._execute_rag_search", lambda *args, **kwargs: ("rag answer", [], 0.0, {}))
    monkeypatch.setattr("legal_rag.execution._execute_direct_answer", lambda *args, **kwargs: ("direct answer", [], 0.0, {}))
    monkeypatch.setattr("legal_rag.execution._call_judge", lambda *args, **kwargs: next(judge_results))

    result = _execute_step_with_escalation(step, _state("full_parallel"), [], [], profile)

    attempts = result.trace["attempts"]
    assert [attempt["action_type"] for attempt in attempts] == ["rag_search", "rag_search", "direct_answer"]
    assert attempts[1]["sub_question"] == "rewrite me"
    assert result.step.action_type == "direct_answer"
    assert result.step.status == "completed"
