import copy

import pytest

from scripts.opd.verl_objective_contract import load_plan, validate_plan


def test_upstream_verl_plan_is_pinned_and_not_launch_authority():
    plan = load_plan()
    assert plan["payload"]["scientific_launch_authorized"] is False
    assert plan["payload"]["objective_id"] == "k1_verl_upstream_clip10"
    assert plan["payload"]["fixed_config"]["rollouts_per_prompt"] == 4


def test_upstream_verl_plan_rejects_recipe_drift():
    payload = copy.deepcopy(load_plan()["payload"])
    payload["fixed_config"]["distillation_loss_max_clamp"] = 5.0
    with pytest.raises(ValueError, match="fixed recipe"):
        validate_plan(payload)
