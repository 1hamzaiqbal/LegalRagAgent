import math

from scripts.opd.verify_finite_state import run


def test_finite_state_verifier_covers_rejections_and_real_update(monkeypatch):
    monkeypatch.setattr(
        "scripts.opd.verify_finite_state.subprocess.run",
        lambda command, **kwargs: type(
            "Result",
            (),
            {"stdout": "a" * 40 + "\n" if "rev-parse" in command else ""},
        )(),
    )
    receipt = run()
    assert receipt["status"] == "passed"
    assert len(receipt["cases"]) == 10
    assert receipt["finite_case"]["parameter_update_l2"] > 0
    assert math.isfinite(receipt["finite_case"]["optimizer_state_signature"]["squared_l2"])
    assert receipt["scientific_launch_authorized"] is False
