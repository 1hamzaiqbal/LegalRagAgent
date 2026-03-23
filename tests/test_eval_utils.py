import json

import pandas as pd

from eval import eval_utils


def test_select_qa_queries_by_labels_preserves_requested_order(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "subject": "TORTS",
                "full_q": "Question 1",
                "gold_idx": "mbe_1",
                "answer": "A",
                "choice_a": "A1",
                "choice_b": "B1",
                "choice_c": "C1",
                "choice_d": "D1",
                "gold_passage": "Gold 1",
            },
            {
                "subject": "CONTRACTS",
                "full_q": "Question 2",
                "gold_idx": "mbe_2",
                "answer": "B",
                "choice_a": "A2",
                "choice_b": "B2",
                "choice_c": "C2",
                "choice_d": "D2",
                "gold_passage": "Gold 2",
            },
        ],
        index=[10, 20],
    )
    monkeypatch.setattr(eval_utils, "load_qa_with_gold", lambda: frame)

    queries = eval_utils.select_qa_queries_by_labels(["qa_contracts_20", "qa_torts_10"])

    assert [query["label"] for query in queries] == ["qa_contracts_20", "qa_torts_10"]
    assert [query["question"] for query in queries] == ["Question 2", "Question 1"]


def test_load_eval_labels_supports_json_and_jsonl(tmp_path):
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps({"labels": ["qa_one", "qa_two"]}), encoding="utf-8")
    assert eval_utils.load_eval_labels(str(json_path)) == ["qa_one", "qa_two"]

    jsonl_path = tmp_path / "detail.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"label": "qa_three"}),
                json.dumps({"label": "qa_four"}),
            ]
        ),
        encoding="utf-8",
    )
    assert eval_utils.load_eval_labels(str(jsonl_path)) == ["qa_three", "qa_four"]
