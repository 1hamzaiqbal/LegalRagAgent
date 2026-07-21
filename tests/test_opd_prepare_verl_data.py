import json

from scripts.opd import prepare_verl_objective_data as prepare


def test_verl_dataset_preserves_exact_prompt_order_and_source(tmp_path, monkeypatch):
    task = tmp_path / "task.jsonl"
    rows = [
        {
            "record_id": f"M:{index}",
            "source": "M",
            "role": "student_opd",
            "prompt": [{"role": "user", "content": f"problem {index}"}],
            "solution": str(index),
        }
        for index in range(3)
    ]
    task.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}\n")
    prompt = tmp_path / "prompt.json"
    prompt.write_text("{}\n")
    contract = {
        "path": str(prompt.resolve()),
        "sha256": "a" * 64,
        "sequence_sha256": "b" * 64,
        "source": "M",
        "seed": 0,
        "consumed_prefix_rows": 1,
        "full_sequence_rows": 100,
    }
    monkeypatch.setattr(
        prepare,
        "validate_prompt_plan",
        lambda *args, **kwargs: (contract, [rows[2]]),
    )
    output, manifest = prepare.build_dataset(
        task_file=task,
        prepared_manifest=prepared,
        prompt_plan=prompt,
        source="M",
        seed=0,
        git_commit="c" * 40,
        diagnostic=True,
    )
    assert [row["extra_info"]["record_id"] for row in output] == ["M:2"]
    assert output[0]["data_source"] == "legalrag_opd_math_M"
    assert output[0]["reward_model"]["ground_truth"] == "2"
    assert manifest["diagnostic"] is True
    assert manifest["rows"] == 1
