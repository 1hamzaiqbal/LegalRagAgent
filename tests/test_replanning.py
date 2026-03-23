from legal_rag.execution import planner_node, replanner_node
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
        "replanning_brief": "",
        "step_traces": [],
        "run_artifact": {},
    }


def test_replanner_builds_follow_up_brief():
    state = _state()
    state["planning_table"] = [
        PlanningStep(
            step_id=1,
            sub_question="What is the exclusionary rule?",
            status="completed",
            result="The exclusionary rule suppresses evidence obtained in violation of the Fourth Amendment.",
            judge_verdict={"sufficient": "full", "reason": "complete"},
        ),
        PlanningStep(
            step_id=2,
            sub_question="What exceptions limit suppression?",
            status="completed",
            result="The answer identified the good-faith exception but missed attenuation.",
            judge_verdict={"sufficient": "partial", "reason": "missing one exception", "missing": "attenuation"},
        ),
    ]
    state["final_answer"] = "So far, the answer covers the exclusionary rule but not all limiting exceptions."
    state["completeness_verdict"] = {"complete": False, "missing_topics": ["What exceptions other than good faith can defeat suppression?"]}
    state["step_traces"] = [
        {
            "step_id": 2,
            "attempts": [
                {"action_type": "rag_search", "rewrite_attempt": 0},
                {"action_type": "rag_search", "rewrite_attempt": 1},
                {"action_type": "direct_answer", "rewrite_attempt": 0},
            ],
        }
    ]

    result = replanner_node(state)

    assert "Open gaps" in result["replanning_brief"]
    assert "What exceptions other than good faith can defeat suppression?" in result["replanning_brief"]
    assert "rag_search -> rag_search/rewrite1 -> direct_answer" in result["replanning_brief"]


def test_planner_follow_up_appends_new_steps(monkeypatch):
    state = _state()
    state["planning_table"] = [
        PlanningStep(
            step_id=1,
            sub_question="What is the exclusionary rule?",
            status="completed",
            result="Rule summary.",
            judge_verdict={"sufficient": "full", "reason": "complete"},
        )
    ]
    state["completeness_verdict"] = {"complete": False, "missing_topics": ["What exceptions limit suppression?"]}
    state["replanning_brief"] = "Open gaps:\n- What exceptions limit suppression?"

    monkeypatch.setattr(
        "legal_rag.execution._llm_call",
        lambda *args, **kwargs: """{
            "complexity": "simple",
            "steps": [
                {
                    "sub_question": "What exceptions limit application of the exclusionary rule?",
                    "authority_target": "exclusionary rule exceptions",
                    "retrieval_hints": ["good faith", "attenuation", "inevitable discovery"],
                    "action_type": "rag_search",
                    "max_retries": 2
                }
            ]
        }""",
    )

    result = planner_node(state)

    assert len(result["planning_table"]) == 2
    assert result["planning_table"][0].step_id == 1
    assert result["planning_table"][1].step_id == 2
    assert result["planning_table"][1].sub_question.startswith("What exceptions limit")
    assert result["replanning_brief"] == ""


def test_planner_follow_up_parse_failure_falls_back_to_missing_topics(monkeypatch):
    state = _state()
    state["planning_table"] = [
        PlanningStep(
            step_id=1,
            sub_question="What is the exclusionary rule?",
            status="completed",
            result="Rule summary.",
            judge_verdict={"sufficient": "full", "reason": "complete"},
        )
    ]
    state["completeness_verdict"] = {"complete": False, "missing_topics": ["What exceptions limit suppression?"]}
    state["replanning_brief"] = "Open gaps:\n- What exceptions limit suppression?"

    monkeypatch.setattr("legal_rag.execution._llm_call", lambda *args, **kwargs: "not-json")

    result = planner_node(state)

    assert len(result["planning_table"]) == 2
    assert result["planning_table"][1].step_id == 2
    assert result["planning_table"][1].sub_question == "What exceptions limit suppression?"
