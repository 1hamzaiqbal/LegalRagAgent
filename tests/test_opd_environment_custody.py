import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.opd import opd_train
from scripts.opd_math import evaluate_math as evaluation
from scripts.opd_math import student_results
from scripts.opd_math import verify_environment as verifier


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "hpc" / "slurm_opd_math_student_train.sh"
COMMIT = "a" * 40


def make_contract(tmp_path, monkeypatch, *, kind="train", packages=None):
    packages = packages or {"alpha": "1.0", "beta-package": "2.0"}
    environment = tmp_path / f"{kind}_environment"
    bin_dir = environment / "bin"
    bin_dir.mkdir(parents=True)
    environment_python = bin_dir / "python"
    environment_python.write_bytes(b"python")
    environment_python.chmod(0o755)
    freeze_payload = "".join(
        f"{name}=={version}\n" for name, version in sorted(packages.items())
    ).encode()
    (environment / "requirements.freeze.txt").write_bytes(freeze_payload)

    freeze_root = tmp_path / "environment_freezes" / COMMIT
    freeze_root.mkdir(parents=True)
    commit_freeze = freeze_root / f"{kind}.freeze.txt"
    commit_freeze.write_bytes(freeze_payload)

    monkeypatch.setattr(sys, "prefix", str(environment))
    monkeypatch.setattr(sys, "executable", str(environment_python))
    monkeypatch.setattr(
        verifier, "installed_distribution_versions", lambda: dict(packages)
    )
    return environment, commit_freeze, packages


def run_verification(environment, commit_freeze, *, kind="train", executable=None):
    return verifier.verify_environment(
        environment_root=environment,
        commit_freeze=commit_freeze,
        expected_commit=COMMIT,
        freeze_kind=kind,
        expected_executable=executable,
    )


def test_exact_environment_and_commit_freeze_pass(tmp_path, monkeypatch):
    environment, commit_freeze, packages = make_contract(tmp_path, monkeypatch)
    result = run_verification(environment, commit_freeze)
    assert result["status"] == "passed"
    assert result["installed_distribution_count"] == len(packages)
    assert result["commit_freeze"]["byte_identical_to_requirements_freeze"] is True
    assert result["requirements_freeze"]["sha256"] == result["commit_freeze"]["sha256"]


def test_upstream_verl_environment_and_commit_freeze_pass(tmp_path, monkeypatch):
    environment, commit_freeze, packages = make_contract(
        tmp_path, monkeypatch, kind="upstream_verl"
    )
    result = run_verification(
        environment, commit_freeze, kind="upstream_verl"
    )
    assert result["status"] == "passed"
    assert result["freeze_kind"] == "upstream_verl"
    assert result["installed_distribution_count"] == len(packages)
    assert result["expected_executable"] is None


@pytest.mark.parametrize(
    ("actual", "expected_detail"),
    [
        ({"alpha": "1.0"}, '"missing": ["beta-package"]'),
        (
            {"alpha": "1.0", "beta-package": "2.0", "gamma": "3.0"},
            '"extra": ["gamma"]',
        ),
        (
            {"alpha": "9.0", "beta-package": "2.0"},
            '"version_drift": {"alpha": {"freeze": "1.0", "installed": "9.0"}}',
        ),
    ],
)
def test_missing_extra_and_version_drift_fail_closed(
    tmp_path, monkeypatch, actual, expected_detail
):
    environment, commit_freeze, _ = make_contract(tmp_path, monkeypatch)
    monkeypatch.setattr(verifier, "installed_distribution_versions", lambda: actual)
    with pytest.raises(ValueError, match="installed distribution map differs") as error:
        run_verification(environment, commit_freeze)
    assert expected_detail in str(error.value)


def test_declared_environment_must_equal_live_sys_prefix(tmp_path, monkeypatch):
    environment, commit_freeze, _ = make_contract(tmp_path, monkeypatch)
    other = tmp_path / "other_environment"
    other.mkdir()
    monkeypatch.setattr(sys, "prefix", str(other))
    with pytest.raises(ValueError, match="not the live sys.prefix"):
        run_verification(environment, commit_freeze)


def test_commit_freeze_must_be_byte_identical(tmp_path, monkeypatch):
    environment, commit_freeze, _ = make_contract(tmp_path, monkeypatch)
    commit_freeze.write_text("alpha==1.0\nbeta-package==2.1\n")
    with pytest.raises(ValueError, match="not byte-identical"):
        run_verification(environment, commit_freeze)


def test_requirements_and_commit_freezes_must_be_regular(tmp_path, monkeypatch):
    environment, commit_freeze, _ = make_contract(tmp_path, monkeypatch)
    requirements = environment / "requirements.freeze.txt"
    real_requirements = environment / "real.freeze.txt"
    requirements.rename(real_requirements)
    requirements.symlink_to(real_requirements)
    with pytest.raises(ValueError, match="regular non-symlink"):
        run_verification(environment, commit_freeze)

    requirements.unlink()
    real_requirements.rename(requirements)
    real_commit_freeze = commit_freeze.with_name("real.freeze.txt")
    commit_freeze.rename(real_commit_freeze)
    commit_freeze.symlink_to(real_commit_freeze)
    with pytest.raises(ValueError, match="regular non-symlink"):
        run_verification(environment, commit_freeze)


def test_serve_executable_requires_exact_environment_python_shebang(
    tmp_path, monkeypatch
):
    environment, commit_freeze, _ = make_contract(
        tmp_path, monkeypatch, kind="serve"
    )
    vllm = environment / "bin" / "vllm"
    vllm.write_text("#!/usr/bin/env python\nprint('wrong interpreter')\n")
    vllm.chmod(0o755)
    with pytest.raises(ValueError, match="shebang does not invoke"):
        run_verification(environment, commit_freeze, kind="serve", executable=vllm)

    vllm.write_text(f"#!{environment / 'bin' / 'python'}\nprint('ok')\n")
    result = run_verification(
        environment, commit_freeze, kind="serve", executable=vllm
    )
    assert result["expected_executable"]["shebang"] == f"#!{environment}/bin/python"


def test_serve_verification_cannot_omit_vllm(tmp_path, monkeypatch):
    environment, commit_freeze, _ = make_contract(
        tmp_path, monkeypatch, kind="serve"
    )
    with pytest.raises(ValueError, match="requires.*bin/vllm"):
        run_verification(environment, commit_freeze, kind="serve")


def test_serve_drift_fails_trainer_custody_and_final_promotion(tmp_path, monkeypatch):
    freeze_root = tmp_path / "environment_freezes" / COMMIT
    freeze_root.mkdir(parents=True)
    train_freeze = freeze_root / "train.freeze.txt"
    serve_freeze = freeze_root / "serve.freeze.txt"
    train_freeze.write_text("alpha==1.0\n")
    serve_freeze.write_text("beta==2.0\n")

    def binding(path):
        return {"path": str(path.resolve()), "sha256": opd_train.sha256_file(path)}

    train_binding = binding(train_freeze)
    serve_binding = binding(serve_freeze)
    train_record = {
        "freeze_kind": "train",
        "expected_commit": COMMIT,
        "commit_freeze": {
            **train_binding,
            "byte_identical_to_requirements_freeze": True,
        },
    }
    serve_record = {
        "freeze_kind": "serve",
        "expected_commit": COMMIT,
        "commit_freeze": {
            **serve_binding,
            "byte_identical_to_requirements_freeze": True,
        },
    }
    contract = {
        "schema_version": 2,
        "git_commit": COMMIT,
        "verifier": {
            "path": str(opd_train.ENVIRONMENT_VERIFIER.resolve()),
            "sha256": opd_train.sha256_file(opd_train.ENVIRONMENT_VERIFIER),
        },
        "train_runtime_packages": opd_train.EXPECTED_TRAIN_PACKAGES,
        "train_freeze": train_binding,
        "train_verification": train_record,
        "serve_freeze": serve_binding,
        "serve_verification": serve_record,
    }
    monkeypatch.setattr(
        opd_train,
        "installed_package_versions",
        lambda expected: dict(opd_train.EXPECTED_TRAIN_PACKAGES),
    )
    calls = []

    def reverify(recorded, *, in_process=False):
        calls.append((recorded["freeze_kind"], in_process))
        if recorded["freeze_kind"] == "serve":
            raise ValueError("serve package or executable drift")
        return recorded

    monkeypatch.setattr(opd_train, "reverify_recorded_environment", reverify)
    assert not opd_train.environment_contract_unchanged(contract)
    assert calls == [("train", True), ("serve", False)]
    assert (
        opd_train.final_promotion_custody_failure_status(
            stable_final_artifact=True,
            intended_scientific_run=True,
            clean_stable_code=True,
            stable_environment_end=False,
        )
        == "failed_environment_custody_after_promotion"
    )


def test_heldout_reopens_train_in_process_and_serve_externally(tmp_path, monkeypatch):
    train_root, train_freeze, _ = make_contract(
        tmp_path / "train", monkeypatch, kind="train"
    )
    train_record = run_verification(train_root, train_freeze, kind="train")
    serve_root, serve_freeze, _ = make_contract(
        tmp_path / "serve", monkeypatch, kind="serve"
    )
    vllm = serve_root / "bin" / "vllm"
    vllm.write_text(f"#!{serve_root}/bin/python\n")
    vllm.chmod(0o755)
    serve_record = run_verification(
        serve_root, serve_freeze, kind="serve", executable=vllm
    )
    calls = []

    def reverify(recorded, *, in_process=False):
        calls.append((recorded["freeze_kind"], in_process))
        return dict(recorded)

    monkeypatch.setattr(student_results, "reverify_recorded_environment", reverify)
    student_results._validate_environment_verification(
        train_record,
        freeze=train_record["commit_freeze"],
        commit=COMMIT,
        kind="train",
    )
    student_results._validate_environment_verification(
        serve_record,
        freeze=serve_record["commit_freeze"],
        commit=COMMIT,
        kind="serve",
    )
    assert calls == [("train", True), ("serve", False)]


def test_freeze_parser_rejects_unmapped_requirements():
    with pytest.raises(ValueError, match="not an exact name==version"):
        verifier.parse_exact_freeze(
            b"alpha==1.0\nbeta @ https://example.invalid/beta.whl\n",
            label="freeze",
        )


def test_student_wrapper_syntax_and_environment_verification_order():
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    script = WRAPPER.read_text()
    trainer = '"$TRAIN_ENV/bin/python" "$REPO/scripts/opd/opd_train.py"'
    canonical_server_launch = (
        '  "$SERVE_ENV/bin/vllm" serve "$OPD_MATH_TEACHER_CHECKPOINT" \\\n'
        '    --host 127.0.0.1 \\\n'
        '    --served-model-name opd-math-teacher \\\n'
        '    --port "$PORT" \\\n'
        '    --max-model-len 4096 \\\n'
        '    --gpu-memory-utilization 0.55 >"$OUT.vllm.log" 2>&1 &'
    )
    assert canonical_server_launch in script
    assert script.count('"$TRAIN_ENV/bin/python" "$VERIFY_ENVIRONMENT"') == 2
    assert script.count('"$SERVE_ENV/bin/python" "$VERIFY_ENVIRONMENT"') == 2
    assert script.count("--freeze-kind train") == 2
    assert script.count("--freeze-kind serve") == 2
    assert script.count('--expected-executable "$SERVE_ENV/bin/vllm"') == 2
    assert '--train-environment-root "$TRAIN_ENV"' in script
    assert script.count('--serve-environment-root "$SERVE_ENV"') == 2
    assert script.index("Verifying live train environment") < script.index(trainer)
    assert script.index("Verifying live serve environment") < script.index(trainer)
    assert script.rindex("Re-verifying live train environment") > script.index(trainer)
    assert script.rindex("Re-verifying live serve environment") > script.index(trainer)
    serve_pre = script.index("Verifying live serve environment")
    mode_guard = script.rfind('if [[ "$MODE" == task_rl_k1_gap ]]; then', 0, serve_pre)
    assert mode_guard != -1


def test_verifier_cli_emits_machine_readable_pass(monkeypatch, capsys, tmp_path):
    environment, commit_freeze, _ = make_contract(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_environment.py",
            "--environment-root",
            str(environment),
            "--commit-freeze",
            str(commit_freeze),
            "--expected-commit",
            COMMIT,
            "--freeze-kind",
            "train",
        ],
    )
    verifier.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == verifier.SCHEMA
    assert payload["status"] == "passed"


def test_evaluator_binds_and_reverifies_exact_commit_environment(tmp_path, monkeypatch):
    expected = dict(evaluation.EXPECTED_EVALUATION_PACKAGES)
    environment, commit_freeze, _ = make_contract(
        tmp_path, monkeypatch, packages=expected
    )
    monkeypatch.setattr(evaluation, "package_versions", lambda: dict(expected))
    git = {"commit": COMMIT, "worktree_clean": True}
    contract = evaluation.validate_evaluation_environment_contract(
        Namespace(
            train_environment_root=environment,
            train_environment_freeze=commit_freeze,
        ),
        git,
    )
    assert contract["git_commit"] == COMMIT
    assert contract["train_environment_root"] == str(environment.resolve())
    assert contract["train_freeze"]["sha256"] == evaluation.sha256_file(
        commit_freeze
    )
    monkeypatch.setattr(
        evaluation,
        "reverify_recorded_environment",
        lambda recorded, *, in_process: dict(recorded),
    )
    assert evaluation.evaluation_environment_contract_unchanged(contract)


def test_evaluator_environment_rejects_missing_packages_and_symlink_freeze(
    tmp_path, monkeypatch
):
    expected = dict(evaluation.EXPECTED_EVALUATION_PACKAGES)
    environment, commit_freeze, _ = make_contract(
        tmp_path, monkeypatch, packages=expected
    )
    args = Namespace(
        train_environment_root=environment,
        train_environment_freeze=commit_freeze,
    )
    git = {"commit": COMMIT, "worktree_clean": True}
    missing = dict(expected)
    missing.pop(next(iter(missing)))
    monkeypatch.setattr(evaluation, "package_versions", lambda: missing)
    with pytest.raises(ValueError, match="live evaluation packages differ"):
        evaluation.validate_evaluation_environment_contract(args, git)

    monkeypatch.setattr(evaluation, "package_versions", lambda: dict(expected))
    real_freeze = commit_freeze.with_name("real-train.freeze.txt")
    commit_freeze.rename(real_freeze)
    commit_freeze.symlink_to(real_freeze)
    with pytest.raises(ValueError, match="regular non-symlink"):
        evaluation.validate_evaluation_environment_contract(args, git)


def test_evaluator_environment_rejects_symlinked_environment_root(
    tmp_path, monkeypatch
):
    expected = dict(evaluation.EXPECTED_EVALUATION_PACKAGES)
    environment, commit_freeze, _ = make_contract(
        tmp_path, monkeypatch, packages=expected
    )
    linked_environment = tmp_path / "linked-train-environment"
    linked_environment.symlink_to(environment, target_is_directory=True)
    monkeypatch.setattr(evaluation, "package_versions", lambda: dict(expected))
    with pytest.raises(ValueError, match="environment root.*non-symlink"):
        evaluation.validate_evaluation_environment_contract(
            Namespace(
                train_environment_root=linked_environment,
                train_environment_freeze=commit_freeze,
            ),
            {"commit": COMMIT, "worktree_clean": True},
        )
