from legal_rag.profiles import PROFILE_NAMES, get_profile, list_profiles


def test_profile_registry_contains_expected_presets():
    assert {
        "llm_only",
        "simple_rag",
        "rewrite_rag",
        "full_seq",
        "full_parallel",
        "full_parallel_aspect",
    }.issubset(set(PROFILE_NAMES))


def test_get_profile_returns_resolved_profile():
    profile = get_profile("full_parallel_aspect")
    assert profile.name == "full_parallel_aspect"
    assert profile.rag_strategy == "aspect"
    assert profile.step_execution_mode == "parallel"
    assert profile.use_bm25 is False


def test_list_profiles_is_non_empty():
    names = {profile.name for profile in list_profiles()}
    assert "full_parallel" in names


def test_full_runtime_profiles_disable_bm25_for_iteration_speed():
    assert get_profile("full_seq").use_bm25 is False
    assert get_profile("full_parallel").use_bm25 is False
    assert get_profile("full_parallel_aspect").use_bm25 is False
