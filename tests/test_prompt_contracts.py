from legal_rag.core import _parse_json
from legal_rag.execution import _call_judge, planner_node, synthesizer_node
from legal_rag.models import PlanningStep
from legal_rag.profiles import get_profile


def _state():
    return {
        "agent_metadata": {},
        "inputs": {"question": "Question?", "research_question": "Question?"},
        "run_config": {"max_steps": 7, "max_parallel_rounds": 3},
        "profile": get_profile("full_parallel").to_state_dict(),
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


def test_parse_json_handles_common_llm_wrapping():
    parsed = _parse_json(
        """```json
        {"primary": "query", "alternatives": ["a", "b",],}
        ```"""
    )
    assert parsed["primary"] == "query"
    assert parsed["alternatives"] == ["a", "b"]


def test_planner_falls_back_to_single_step_on_parse_failure(monkeypatch):
    monkeypatch.setattr("legal_rag.execution._llm_call", lambda *args, **kwargs: "not-json")
    result = planner_node(_state())
    assert len(result["planning_table"]) == 1
    assert result["planning_table"][0].sub_question == "Question?"


def test_judge_parse_failure_defaults_to_full(monkeypatch):
    monkeypatch.setattr("legal_rag.execution._llm_call", lambda *args, **kwargs: "not-json")
    verdict = _call_judge(
        PlanningStep(step_id=1, sub_question="Q"),
        "answer",
        ["passage"],
        "Question?",
        get_profile("full_parallel"),
    )
    assert verdict["sufficient"] == "full"


def test_synthesizer_completeness_prompt_includes_step_verdicts(monkeypatch):
    prompts = []

    def fake_llm_call(system_prompt, user_prompt, label=""):
        prompts.append((label, user_prompt))
        if label == "synthesizer":
            return "Synthesized answer."
        return '{"complete": true, "reasoning": "enough", "missing_topics": []}'

    state = _state()
    state["planning_table"] = [
        PlanningStep(
            step_id=1,
            sub_question="What elements govern negligence?",
            action_type="rag_search",
            status="completed",
            result="Negligence requires duty, breach, causation, and damages.",
            judge_verdict={"sufficient": "partial", "reason": "needs contractor-specific standard", "missing": "contractor-specific duty"},
        )
    ]
    state["evidence_store"] = [{"idx": "doc_1", "text": "Evidence text", "source": "mbe"}]

    monkeypatch.setattr("legal_rag.execution._llm_call", fake_llm_call)
    synthesizer_node(state)

    completeness_prompt = next(prompt for label, prompt in prompts if label == "completeness")
    assert "STEP VERDICTS" in completeness_prompt
    assert "judge=partial" in completeness_prompt
    assert "contractor-specific duty" in completeness_prompt
