from legal_rag.core import _parse_json
from legal_rag.execution import _call_judge, planner_node, route_after_synthesizer, synthesizer_node
from legal_rag.models import PlanningStep
from legal_rag.profiles import get_profile
from legal_rag.state_utils import research_question_from_state


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
        "replanning_brief": "",
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
        if label == "mc/answer":
            return '{"answer": "A", "reasoning": "best supported"}'
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


def test_research_question_prefers_mc_objective():
    state = _state()
    state["inputs"]["question"] = "Question stem\n\nAnswer choices:\n  (A) One\n  (B) Two"
    state["inputs"]["research_question"] = "Question stem"

    assert research_question_from_state(state) == state["inputs"]["question"]


def test_synthesizer_normalizes_mc_answer_with_adjudicator(monkeypatch):
    def fake_exec_call(system_prompt, user_prompt, label=""):
        if label == "synthesizer":
            return "Analysis body.\n\n**Answer: (None of the choices are correct)**"
        return '{"complete": true, "reasoning": "enough", "missing_topics": []}'

    def fake_node_call(system_prompt, user_prompt, label=""):
        assert label == "mc/answer"
        return '{"answer": "B", "reasoning": "best supported"}'

    state = _state()
    state["inputs"]["question"] = "Question stem\n\nAnswer choices:\n  (A) One\n  (B) Two"
    state["planning_table"] = [
        PlanningStep(
            step_id=1,
            sub_question="What matters?",
            action_type="rag_search",
            status="completed",
            result="Result.",
            judge_verdict={"sufficient": "full", "reason": "complete"},
        )
    ]
    state["evidence_store"] = [{"idx": "doc_1", "text": "Evidence text", "source": "mbe"}]

    monkeypatch.setattr("legal_rag.execution._llm_call", fake_exec_call)
    monkeypatch.setattr("legal_rag.nodes._llm_call", fake_node_call)

    result = synthesizer_node(state)
    assert result["final_answer"].endswith("**Answer: (B)**")


def test_synthesizer_marks_max_rounds_terminal_without_claiming_complete(monkeypatch):
    def fake_exec_call(system_prompt, user_prompt, label=""):
        if label == "synthesizer":
            return "Analysis body."
        return '{"complete": false, "reasoning": "missing a decisive exception", "missing_topics": ["What exception controls?"]}'

    state = _state()
    state["parallel_round"] = 3
    state["planning_table"] = [
        PlanningStep(
            step_id=1,
            sub_question="What is the governing rule?",
            action_type="rag_search",
            status="completed",
            result="Rule result.",
            judge_verdict={"sufficient": "partial", "reason": "missing exception", "missing": "exception"},
        )
    ]
    state["evidence_store"] = [{"idx": "doc_1", "text": "Evidence text", "source": "mbe"}]

    monkeypatch.setattr("legal_rag.execution._llm_call", fake_exec_call)

    result = synthesizer_node(state)

    assert result["completeness_verdict"]["complete"] is False
    assert result["completeness_verdict"]["terminal"] is True
    assert result["completeness_verdict"]["terminal_reason"] == "max_rounds"
    assert route_after_synthesizer({"completeness_verdict": result["completeness_verdict"]}) == "__end__"
