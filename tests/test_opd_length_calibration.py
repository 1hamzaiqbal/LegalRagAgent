from __future__ import annotations

import json
from pathlib import Path

from scripts.opd_math.length_calibration import analyze_surface, parse_surface


def test_parse_surface_contract():
    family, arm, tokens, path = parse_surface("teacher:base:2048=/tmp/merged")
    assert (family, arm, tokens) == ("teacher", "base", 2048)
    assert path == Path("/tmp/merged")


def test_analyze_surface_requires_exact_registered_geometry(tmp_path: Path):
    directory = tmp_path / "merged"
    directory.mkdir()
    samples = [
        {
            "record_id": "a",
            "sample_idx": 0,
            "reward": 0.0,
            "reward_status": "prediction_parse_failed",
            "completion_tokens": 2048,
        },
        {
            "record_id": "a",
            "sample_idx": 1,
            "reward": 1.0,
            "reward_status": "correct",
            "completion_tokens": 100,
        },
    ]
    (directory / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "records": 1,
                "samples": 2,
                "samples_per_problem": 2,
                "decoding": {"max_new_tokens": 2048},
                "task_file": "/data/roles/O/student_opd.jsonl",
                "model": "model",
                "model_revision": "revision",
                "adapter": None,
            }
        ),
        encoding="utf-8",
    )
    directory.with_suffix(".custody.json").write_text("{}\n", encoding="utf-8")
    result = analyze_surface(
        directory,
        expected_tokens=2048,
        expected_records=1,
        expected_samples_per_record=2,
    )
    assert result["at_cap_fraction"] == 0.5
    assert result["parse_failures_at_cap"] == 1
    assert result["sample_identities"] == {("a", 0), ("a", 1)}
