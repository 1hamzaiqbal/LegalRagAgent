import json

from legal_rag.artifacts import build_run_artifact, write_run_artifact
from legal_rag.models import ExecutionResult
from legal_rag.profiles import get_profile


def test_write_run_artifact_persists_artifact_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    artifact = {"profile_name": "full_parallel", "question": "Q?"}

    path = write_run_artifact(artifact)
    payload = json.loads((tmp_path / path).read_text())

    assert payload["artifact_path"] == path
    assert payload["profile_name"] == "full_parallel"


def test_build_run_artifact_copies_answer_and_terminal_flags():
    result = ExecutionResult(
        profile=get_profile("full_parallel"),
        final_answer="Answer.",
        completeness_verdict={"complete": False, "terminal_reason": "max_rounds"},
    )

    artifact = build_run_artifact(result, question="Q?")

    assert artifact["answered"] is False
    assert artifact["terminal_reason"] == "max_rounds"
