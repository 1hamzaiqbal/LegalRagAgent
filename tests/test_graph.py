import pytest

from legal_rag.runtime import build_graph


def test_build_graph_supports_full_profiles():
    assert build_graph("full_seq") is not None
    assert build_graph("full_parallel") is not None


def test_build_graph_rejects_baseline_profiles():
    with pytest.raises(ValueError):
        build_graph("llm_only")
