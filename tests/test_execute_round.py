from legal_rag.execution import StepExecutionResult, execute_round_node
from legal_rag.models import LegalAgentState, PlanningStep
from legal_rag.profiles import get_profile


def _state(profile_name: str) -> LegalAgentState:
    return {
        "agent_metadata": {},
        "inputs": {"question": "Question?", "research_question": "Question?"},
        "run_config": {"max_steps": 7, "max_parallel_rounds": 3},
        "profile": get_profile(profile_name).to_state_dict(),
        "collections": ["legal_passages"],
        "planning_table": [
            PlanningStep(step_id=1, sub_question="Step 1"),
            PlanningStep(step_id=2, sub_question="Step 2"),
        ],
        "evidence_store": [{"idx": "existing", "text": "existing text", "source": "mbe", "step_ids": [99]}],
        "final_answer": "",
        "audit_log": [],
        "completeness_verdict": {},
        "parallel_round": 1,
        "step_traces": [],
        "run_artifact": {},
    }


def test_parallel_round_uses_same_snapshot_and_deduplicates(monkeypatch):
    snapshots = []

    def fake_execute(step, state, table_snapshot, evidence_snapshot, profile):
        snapshots.append(
            {
                "step_id": step.step_id,
                "snapshot_evidence_ids": [item["idx"] for item in evidence_snapshot],
                "snapshot_table_ids": [item.step_id for item in table_snapshot],
            }
        )
        updated = step.model_copy(
            update={
                "status": "completed",
                "result": f"result {step.step_id}",
                "confidence": 0.9,
                "judge_verdict": {"sufficient": "full", "reason": "ok"},
            }
        )
        return StepExecutionResult(
            step=updated,
            evidence=[
                {
                    "idx": "shared_doc",
                    "text": "same evidence",
                    "source": "caselaw",
                    "step_id": step.step_id,
                    "step_ids": [step.step_id],
                    "cross_encoder_score": 1.0,
                }
            ],
            trace={"step_id": step.step_id, "attempts": [], "round": 1},
        )

    monkeypatch.setattr("legal_rag.execution._execute_step_with_escalation", fake_execute)
    result = execute_round_node(_state("full_parallel"))

    planning_table = result["planning_table"]
    evidence_store = result["evidence_store"]
    assert [item["snapshot_evidence_ids"] for item in snapshots] == [["existing"], ["existing"]]
    assert all(step.evidence_ids == ["shared_doc"] for step in planning_table)
    assert len(evidence_store) == 2
    shared = next(item for item in evidence_store if item["idx"] == "shared_doc")
    assert shared["step_ids"] == [1, 2]
    traces = result["step_traces"]
    assert traces[0]["round"] == 1
    assert traces[0]["sub_question"] == "Step 1"
    assert traces[0]["evidence_ids"] == ["shared_doc"]


def test_sequential_round_executes_single_pending_step(monkeypatch):
    calls = []

    def fake_execute(step, state, table_snapshot, evidence_snapshot, profile):
        calls.append(step.step_id)
        updated = step.model_copy(
            update={
                "status": "completed",
                "result": "done",
                "confidence": 0.5,
                "judge_verdict": {"sufficient": "full", "reason": "ok"},
            }
        )
        return StepExecutionResult(step=updated, evidence=[], trace={"step_id": step.step_id, "attempts": [], "round": 1})

    monkeypatch.setattr("legal_rag.execution._execute_step_with_escalation", fake_execute)
    result = execute_round_node(_state("full_seq"))

    assert calls == [1]
    completed = [step for step in result["planning_table"] if step.status == "completed"]
    assert [step.step_id for step in completed] == [1]
