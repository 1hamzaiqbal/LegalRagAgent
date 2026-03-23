import json

from legal_rag.artifacts import write_run_artifact


def test_write_run_artifact_persists_artifact_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    artifact = {"profile_name": "full_parallel", "question": "Q?"}

    path = write_run_artifact(artifact)
    payload = json.loads((tmp_path / path).read_text())

    assert payload["artifact_path"] == path
    assert payload["profile_name"] == "full_parallel"
