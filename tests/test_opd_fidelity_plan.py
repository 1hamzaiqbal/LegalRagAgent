import copy

import pytest

from scripts.opd.fidelity_plan import DEFAULT_PLAN, load_fidelity_plan, validate_fidelity_plan


def test_fidelity_plan_binds_all_objectives_levels_and_non_authority():
    plan = load_fidelity_plan(DEFAULT_PLAN)
    assert plan["sources"] == ["M", "O"]
    assert plan["levels"]["full_custody_one_step"]["expected_runs"] == 12
    assert plan["levels"]["stored_real_model"]["status"] == "pending"
    assert plan["scientific_launch_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan.update(scientific_launch_authorized=True),
        lambda plan: plan.update(upstream_verl_commit="main"),
        lambda plan: plan.update(sources=["O"]),
        lambda plan: plan["levels"]["full_custody_one_step"].update(expected_runs=10),
        lambda plan: plan["stage_rules"].update(diagnostic_outcomes_may_select_objectives=True),
        lambda plan: plan["levels"]["stored_real_model"]["required_fields"].remove(
            "behavior_token_logprobs_from_generation_scores"
        ),
    ],
)
def test_fidelity_plan_fails_closed_on_drift(mutation):
    plan = load_fidelity_plan(DEFAULT_PLAN)
    payload = {key: copy.deepcopy(value) for key, value in plan.items() if key not in {"path", "sha256"}}
    mutation(payload)
    with pytest.raises(ValueError):
        validate_fidelity_plan(payload)
