import json

from scripts.opd_math.build_semantic_review_packet import build_packet
from scripts.opd_math.data_contract import write_jsonl


def test_review_packet_recovers_only_required_pairs(tmp_path):
    prepared = tmp_path / "prepared"
    (prepared / "audit").mkdir(parents=True)
    (prepared / "roles" / "M").mkdir(parents=True)
    records = [
        {
            "record_id": "M:one",
            "source": "M",
            "source_split": "train",
            "source_index": 1,
            "problem": "Find 1+1.",
            "answer": "2",
            "solution": r"\boxed{2}",
            "source_metadata": {"type": "Algebra"},
        },
        {
            "record_id": "M:two",
            "source": "M",
            "source_split": "train",
            "source_index": 2,
            "problem": "Compute 1+1.",
            "answer": "2",
            "solution": r"\boxed{2}",
            "source_metadata": {"type": "Algebra"},
        },
    ]
    role = prepared / "roles" / "M" / "teacher_train.jsonl"
    write_jsonl(role, records)
    write_jsonl(
        prepared / "audit" / "semantic_candidates.jsonl",
        [
            {
                "pair_id": "required",
                "left_record_id": "M:one",
                "right_record_id": "M:two",
                "jaccard": 0.9,
                "identical_numeric_sequence": True,
                "requires_review": True,
            },
            {
                "pair_id": "not-required",
                "left_record_id": "M:one",
                "right_record_id": "M:two",
                "jaccard": 0.86,
                "identical_numeric_sequence": False,
                "requires_review": False,
            },
        ],
    )
    manifest = {
        "files": {
            "roles/M/teacher_train.jsonl": {"rows": 2},
            "audit/semantic_candidates.jsonl": {"rows": 2},
        }
    }
    (prepared / "prepared_manifest.json").write_text(json.dumps(manifest) + "\n")

    packet = build_packet(prepared)

    assert len(packet) == 1
    assert packet[0]["pair_id"] == "required"
    assert packet[0]["left"]["problem"] == "Find 1+1."
    assert packet[0]["right"]["answer"] == "2"
