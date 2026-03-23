from playtests.cases import list_playtest_case_names, resolve_playtest_cases


def test_list_playtest_case_names_is_non_empty():
    names = list_playtest_case_names()
    assert names
    assert "bar_multihop_success" in names


def test_resolve_playtest_cases_returns_named_case_only():
    cases = resolve_playtest_cases("implied_warranty_remodel")
    assert len(cases) == 1
    assert cases[0]["profile"] == "full_parallel_aspect"


def test_resolve_playtest_cases_raises_helpful_error():
    try:
        resolve_playtest_cases("does_not_exist")
    except KeyError as exc:
        assert "Known cases" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown playtest case")
