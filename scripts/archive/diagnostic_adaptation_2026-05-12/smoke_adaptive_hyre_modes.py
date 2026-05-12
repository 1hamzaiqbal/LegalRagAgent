#!/usr/bin/env python3
"""API-free smoke checks for adaptive HyRE mode wiring.

This monkeypatches the LLM and retrieval functions, so it does not require API
keys, Chroma, embeddings, or a model server. It verifies that adaptive modes are
registered and that their retrieval-query shapes match the intended method
budget before launching expensive cluster jobs.
"""

from __future__ import annotations

import sys
from pathlib import Path
import os

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

import eval_harness as harness  # type: ignore  # noqa: E402
from eval_config import EVAL_MODES, EvalConfig  # type: ignore  # noqa: E402


def _fake_llm(_system: str, _user: str, label: str = "") -> str:
    if label.endswith("/snap_and_hyre"):
        return (
            "## Answer\n"
            "Candidate A matches because procedural default turns on the cited holding.\n"
            "Answer: (A)\n\n"
            "## Passage\n"
            "Procedural default holdings apply when a court enforces a preserved rule "
            "after a party failed to raise a required objection."
        )
    return "Answer: (A)"


def _casehold_row() -> pd.Series:
    return pd.Series({
        "question": "The court cites a holding about procedural default.",
        "choice_a": "Procedural default applies when an objection was required.",
        "choice_b": "A search is reasonable when supported by probable cause.",
        "choice_c": "A contract forms when offer and acceptance are present.",
        "choice_d": "Venue is proper where the defendant resides.",
        "choice_e": "Damages are limited by foreseeable loss.",
    })


def _housing_row() -> pd.Series:
    return pd.Series({
        "question": "Can a landlord terminate the tenancy without notice?",
        "state": "texas",
    })


def _run_mode(mode: str, dataset: str, row: pd.Series, expected_queries: int, expected_route: str) -> None:
    retrieval_calls: list[dict] = []

    def fake_retrieve(_row, queries, **kwargs):
        retrieval_calls.append({
            "queries": list(queries),
            "where": kwargs.get("where"),
            "rerank_query": kwargs.get("rerank_query"),
            "label_prefix": kwargs.get("label_prefix"),
        })
        return {
            "passages": ["dummy legal passage"],
            "evidence_store": [{"id": "dummy"}],
            "retrieved_ids": ["dummy"],
            "gold_retrieved": False,
        }

    harness._llm_call = _fake_llm
    harness._retrieve_and_format = fake_retrieve

    config = EvalConfig(mode=mode, dataset=dataset, retrieval_k=5)
    out = harness.MODE_RUNNERS[mode](row, config)
    assert out["hyre_route"] == expected_route, out
    assert out["adaptive_policy"].startswith("task_shape_bottleneck_v1"), out
    assert len(retrieval_calls) == 1, retrieval_calls
    queries = retrieval_calls[0]["queries"]
    assert len(queries) == expected_queries, (mode, queries)
    assert all("Answer:" not in query for query in queries[1:]), (mode, queries)


def _run_option_table_mode() -> None:
    retrieval_calls: list[dict] = []

    def fake_retrieve(_row, queries, **kwargs):
        retrieval_calls.append({
            "queries": list(queries),
            "where": kwargs.get("where"),
            "rerank_query": kwargs.get("rerank_query"),
            "label_prefix": kwargs.get("label_prefix"),
        })
        label = str(kwargs.get("label_prefix") or "")
        candidate = label.rsplit("_", 1)[-1]
        return {
            "passages": [f"dummy passage for {candidate}"],
            "evidence_store": [{
                "idx": f"dummy_{candidate}",
                "text": f"retrieved holding evidence for candidate {candidate}",
                "cross_encoder_score": 1.0,
            }],
            "retrieved_ids": [f"dummy_{candidate}"],
            "gold_retrieved": False,
        }

    harness._llm_call = _fake_llm
    harness._retrieve_and_format = fake_retrieve

    old_disable_ce = os.environ.get("DISABLE_CROSS_ENCODER")
    os.environ["DISABLE_CROSS_ENCODER"] = "1"
    try:
        config = EvalConfig(mode="adaptive_snap_hyre_option_table", dataset="casehold", retrieval_k=5)
        out = harness.MODE_RUNNERS["adaptive_snap_hyre_option_table"](_casehold_row(), config)
        assert out["hyre_route"] == "casehold_option_table", out
        assert out["adaptive_policy"] == "casehold_option_table_v1", out
        assert len(retrieval_calls) == 0, retrieval_calls
        assert len(out["candidate_score_table"]) == 5, out
        assert all(item["score_source"] == "lexical_overlap" for item in out["candidate_score_table"]), out
        assert out["gold_retrieved"] is False, out
        assert out["final_context_fields"] == [
            "question",
            "snap_reasoning",
            "hyde_passage",
            "candidate_score_table",
        ], out
    finally:
        if old_disable_ce is None:
            os.environ.pop("DISABLE_CROSS_ENCODER", None)
        else:
            os.environ["DISABLE_CROSS_ENCODER"] = old_disable_ce


def main() -> None:
    required_modes = {
        "adaptive_snap_hyre": 1,
        "adaptive_snap_hyre_anchor": 2,
        "adaptive_snap_hyre_diverse": 3,
    }
    missing = [mode for mode in required_modes if mode not in EVAL_MODES or mode not in harness.MODE_RUNNERS]
    if missing:
        raise SystemExit("missing adaptive mode(s): " + ", ".join(missing))

    for mode, expected_queries in required_modes.items():
        _run_mode(mode, "casehold", _casehold_row(), expected_queries, "option_grounding")

    # Housing should route through the state-filter branch when metadata exists.
    _run_mode("adaptive_snap_hyre_diverse", "housing", _housing_row(), 3, "state_filter")
    _run_option_table_mode()
    print("adaptive HyRE smoke checks passed")


if __name__ == "__main__":
    main()
