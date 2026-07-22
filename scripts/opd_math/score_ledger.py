#!/usr/bin/env python3
"""Build and validate immutable score-ledger teacher gates.

The original teacher-gap gate re-ran Math-Verify over every completion whenever
the gate was consumed.  Math-Verify uses signal-bounded parsing and symbolic
comparison, so that made promotion depend on load-sensitive replay.  This
module separates measurement from custody:

* the evaluation-time score is retained as the primary observation;
* candidate-side verifier errors remain explicit bounded unknowns;
* a gold-only eligibility rule limits the symbolic metric to answers that are
  actually suitable for symbolic comparison;
* exact, hash-bound adjudications can amend a known score without editing the
  source evaluation; and
* downstream consumers rebuild the ledger and statistics from sealed bytes but
  never invoke the symbolic verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .quality_gates import (
        DEFAULT_TEACHER_MIN_RECORDS,
        TEACHER_GATE_TYPE,
        bootstrap_delta,
        canonical_json_sha256,
        sha256_file,
        sha256_tree,
    )
except ImportError:
    from quality_gates import (  # type: ignore
        DEFAULT_TEACHER_MIN_RECORDS,
        TEACHER_GATE_TYPE,
        bootstrap_delta,
        canonical_json_sha256,
        sha256_file,
        sha256_tree,
    )


LEDGER_SCHEMA_VERSION = 1
LEDGER_MANIFEST_TYPE = "opd_math_teacher_score_ledger_manifest_v1"
LEDGER_ROW_TYPE = "opd_math_teacher_score_ledger_row_v1"
SCORE_LEDGER_GATE_TYPE = "teacher_gap_score_ledger_v1"
MEASUREMENT_POLICY = "score_once_attest_many_with_bounded_unknowns_v1"
ELIGIBILITY_POLICY = "gold_only_symbolic_eligibility_v1"
UNCERTAINTY_POLICY = "binary_worst_case_assignment_v1"
DEFAULT_MIN_ELIGIBLE_COVERAGE = 0.75
ROOT = Path(__file__).resolve().parents[2]


def _json_object(path: Path, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("score-ledger builder requires an immutable Git commit")
    return {"commit": commit, "worktree_clean": not bool(status.strip())}


def classify_symbolic_gold(*, answer: str, solution: str) -> dict[str, Any]:
    """Classify symbolic-verifier eligibility using only registered gold text.

    The rule deliberately does not inspect model completions, arm identity, or
    observed rewards.  Long uncommanded words and prose-dominant answers are
    excluded because converting them into products of one-letter SymPy symbols
    is syntactic success, not semantic scoreability.
    """

    if not isinstance(answer, str) or not answer.strip():
        return {
            "policy": ELIGIBILITY_POLICY,
            "eligible": False,
            "reasons": ["missing_registered_answer"],
            "features": {
                "alphabetic_characters": 0,
                "digit_characters": 0,
                "maximum_uncommanded_alpha_run": 0,
            },
        }
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("registered solution must be non-empty text")

    without_commands = re.sub(r"\\[A-Za-z]+", "", answer)
    alpha_runs = re.findall(r"[A-Za-z]{5,}", without_commands)
    alphabetic = sum(character.isalpha() for character in without_commands)
    digits = sum(character.isdigit() for character in without_commands)
    reasons: list[str] = []
    if re.search(r"\\(?:text|mbox)\s*\{", answer):
        reasons.append("textual_latex_without_registered_string_ontology")
    if alpha_runs:
        reasons.append("long_uncommanded_alpha_run")
    if alphabetic >= 8 and alphabetic > digits:
        reasons.append("prose_dominant_registered_answer")
    return {
        "policy": ELIGIBILITY_POLICY,
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "features": {
            "alphabetic_characters": alphabetic,
            "digit_characters": digits,
            "maximum_uncommanded_alpha_run": max(map(len, alpha_runs), default=0),
        },
    }


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for row_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {label} at row {row_number}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row in {label} at row {row_number}: {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _bound_file(gate: dict[str, Any], key: str) -> Path:
    raw = gate.get(key)
    expected = gate.get(f"{key}_sha256")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ValueError(f"predecessor gate lacks absolute {key}")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"predecessor gate lacks valid {key}_sha256")
    path = Path(raw).resolve()
    if sha256_file(path) != expected:
        raise ValueError(f"predecessor gate binding changed: {key}={path}")
    return path


def _validate_predecessor_gate(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    gate = _json_object(path, "predecessor teacher gate")
    if gate.get("gate") != TEACHER_GATE_TYPE:
        raise ValueError("score ledger requires the canonical predecessor teacher gate")
    if gate.get("gate_strength") != "scientific":
        raise ValueError("score ledger requires a scientific predecessor gate")
    if gate.get("passed") is not True or gate.get("authorizes_scientific_merge") is not True:
        raise ValueError("predecessor teacher gate did not pass")
    paths = {
        key: _bound_file(gate, key)
        for key in (
            "task_file",
            "base_summary",
            "base_samples",
            "trained_summary",
            "trained_samples",
            "prepared_manifest",
            "source_manifest",
            "teacher_run_manifest",
            "teacher_training_task_file",
            "teacher_training_plan",
            "teacher_trainer_state",
            "teacher_trainer_log_history",
            "teacher_train_metrics",
        )
    }
    adapter = gate.get("trained_adapter")
    adapter_hash = gate.get("trained_adapter_tree_sha256")
    if not isinstance(adapter, str) or not Path(adapter).is_absolute():
        raise ValueError("predecessor gate lacks an absolute trained adapter")
    if sha256_tree(Path(adapter)) != adapter_hash:
        raise ValueError("predecessor teacher adapter changed")
    return gate, paths


def _sample_surface(path: Path, *, label: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    allowed = {
        "correct",
        "incorrect",
        "prediction_parse_failed",
        "verifier_error_zeroed",
    }
    for row_number, row in enumerate(_read_jsonl(path, label), start=1):
        record_id = row.get("record_id")
        sample_idx = row.get("sample_idx")
        status = row.get("reward_status")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"invalid record_id in {label} at row {row_number}")
        if not isinstance(sample_idx, int) or sample_idx < 0:
            raise ValueError(f"invalid sample_idx in {label} at row {row_number}")
        if status not in allowed:
            raise ValueError(f"unsupported reward status in {label} at row {row_number}: {status}")
        try:
            reward = float(row["reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid reward in {label} at row {row_number}") from exc
        if reward not in {0.0, 1.0} or not math.isfinite(reward):
            raise ValueError(f"non-binary reward in {label} at row {row_number}")
        if (status == "correct") != (reward == 1.0):
            raise ValueError(f"reward/status mismatch in {label} at row {row_number}")
        completion = row.get("completion_text")
        completion_hash = row.get("completion_sha256")
        if not isinstance(completion, str):
            raise ValueError(f"missing completion text in {label} at row {row_number}")
        actual_completion_hash = hashlib.sha256(completion.encode("utf-8")).hexdigest()
        if completion_hash != actual_completion_hash:
            raise ValueError(f"completion hash mismatch in {label} at row {row_number}")
        grouped.setdefault(record_id, []).append(
            {
                "sample_idx": sample_idx,
                "completion_sha256": completion_hash,
                "stored_reward": reward,
                "stored_reward_status": status,
                "effective_reward": reward,
                "effective_reward_status": status,
                "uncertain": status == "verifier_error_zeroed",
                "uncertainty_reason": (
                    "evaluation_verifier_error" if status == "verifier_error_zeroed" else None
                ),
                "adjudication_id": None,
            }
        )
    for record_id, samples in grouped.items():
        samples.sort(key=lambda row: row["sample_idx"])
        indices = [row["sample_idx"] for row in samples]
        if indices != list(range(len(indices))):
            raise ValueError(f"non-contiguous sample indices for {record_id} in {label}")
    return grouped


def _load_adjudications(
    paths: Iterable[Path],
    *,
    predecessor_gate: dict[str, Any],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    decisions: dict[tuple[str, str, int], dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    expected_samples = {
        "base": str(Path(predecessor_gate["base_samples"]).resolve()),
        "trained": str(Path(predecessor_gate["trained_samples"]).resolve()),
    }
    for raw_path in paths:
        path = Path(raw_path).resolve()
        payload = _json_object(path, "manual adjudication")
        if payload.get("record_type") != "opd_objective_family_manual_verifier_adjudication_v1":
            raise ValueError(f"unsupported adjudication schema: {path}")
        if payload.get("disclosure", {}).get("post_hoc") is not True:
            raise ValueError(f"adjudication must explicitly disclose post-hoc status: {path}")
        verdict = payload.get("manual_math_check", {}).get("verdict")
        if verdict not in {"correct", "incorrect"}:
            raise ValueError(f"adjudication lacks a binary manual verdict: {path}")
        scope = payload.get("scope")
        sample = payload.get("sample")
        if not isinstance(scope, dict) or not isinstance(sample, dict):
            raise ValueError(f"adjudication lacks scope/sample custody: {path}")
        if scope.get("task_file_sha256") != predecessor_gate.get("task_file_sha256"):
            raise ValueError(f"adjudication task binding differs from predecessor gate: {path}")
        bound_arms = [
            arm
            for arm, expected_path in expected_samples.items()
            if str(Path(scope.get(f"{arm}_samples", "")).resolve()) == expected_path
            and scope.get(f"{arm}_samples_sha256")
            == predecessor_gate.get(f"{arm}_samples_sha256")
        ]
        if len(bound_arms) != 2:
            raise ValueError(f"adjudication does not bind both frozen sample surfaces: {path}")
        classification = payload.get("decision", {}).get("classification")
        if isinstance(classification, str) and "_BASE_SAMPLE" in classification:
            arm = "base"
        elif isinstance(classification, str) and "_TRAINED_SAMPLE" in classification:
            arm = "trained"
        else:
            raise ValueError(
                f"adjudication must identify whether the reviewed sample is base or trained: {path}"
            )
        record_id = sample.get("record_id")
        sample_idx = sample.get("sample_idx")
        if not isinstance(record_id, str) or not isinstance(sample_idx, int):
            raise ValueError(f"adjudication has invalid sample identity: {path}")
        key = (arm, record_id, sample_idx)
        if key in decisions:
            raise ValueError(f"duplicate adjudication for {key}")
        decisions[key] = {
            "adjudication_id": payload.get("decision_id"),
            "completion_sha256": sample.get("completion_sha256"),
            "stored_reward": float(sample.get("stored_reward")),
            "stored_reward_status": sample.get("stored_reward_status"),
            "effective_reward": 1.0 if verdict == "correct" else 0.0,
            "effective_reward_status": f"manual_{verdict}",
            "reasoning_sha256": canonical_json_sha256(payload.get("manual_math_check")),
        }
        bindings.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "decision_id": payload.get("decision_id"),
                "post_hoc": True,
            }
        )
    return decisions, sorted(bindings, key=lambda item: item["path"])


def _surface(
    predecessor_gate_path: Path,
    adjudication_paths: Iterable[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predecessor_gate_path = Path(predecessor_gate_path).resolve()
    predecessor, paths = _validate_predecessor_gate(predecessor_gate_path)
    tasks = _read_jsonl(paths["task_file"], "registered task file")
    task_by_id: dict[str, dict[str, Any]] = {}
    for row in tasks:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in task_by_id:
            raise ValueError("registered task file has invalid or duplicate record IDs")
        task_by_id[record_id] = row
    base = _sample_surface(paths["base_samples"], label="base evaluation samples")
    trained = _sample_surface(paths["trained_samples"], label="trained evaluation samples")
    if set(base) != set(trained):
        raise ValueError("base and trained score surfaces have different record sets")
    if len(base) != predecessor.get("shared_records"):
        raise ValueError("predecessor shared-record count differs from score surfaces")
    if not set(base).issubset(task_by_id):
        raise ValueError("score surface contains records absent from registered task file")
    decisions, adjudication_bindings = _load_adjudications(
        adjudication_paths, predecessor_gate=predecessor
    )

    rows: list[dict[str, Any]] = []
    applied: set[tuple[str, str, int]] = set()
    for record_id in sorted(base):
        task = task_by_id[record_id]
        eligibility = classify_symbolic_gold(
            answer=task.get("answer"), solution=task.get("solution")
        )
        arms: dict[str, list[dict[str, Any]]] = {}
        for arm, grouped in (("base", base), ("trained", trained)):
            samples = [dict(sample) for sample in grouped[record_id]]
            for sample in samples:
                key = (arm, record_id, sample["sample_idx"])
                decision = decisions.get(key)
                if decision is None:
                    continue
                if (
                    sample["completion_sha256"] != decision["completion_sha256"]
                    or sample["stored_reward"] != decision["stored_reward"]
                    or sample["stored_reward_status"] != decision["stored_reward_status"]
                ):
                    raise ValueError(f"adjudication bytes do not match frozen sample: {key}")
                sample["effective_reward"] = decision["effective_reward"]
                sample["effective_reward_status"] = decision["effective_reward_status"]
                sample["uncertain"] = False
                sample["uncertainty_reason"] = None
                sample["adjudication_id"] = decision["adjudication_id"]
                sample["adjudication_reasoning_sha256"] = decision["reasoning_sha256"]
                applied.add(key)
            arms[arm] = samples
        rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "row_type": LEDGER_ROW_TYPE,
                "record_id": record_id,
                "eligibility": eligibility,
                "base": arms["base"],
                "trained": arms["trained"],
            }
        )
    if applied != set(decisions):
        raise ValueError(f"adjudications do not resolve to frozen samples: {set(decisions) - applied}")

    source_binding = {
        "predecessor_gate": str(predecessor_gate_path),
        "predecessor_gate_sha256": sha256_file(predecessor_gate_path),
        "task_file": str(paths["task_file"]),
        "task_file_sha256": sha256_file(paths["task_file"]),
        "base_summary": str(paths["base_summary"]),
        "base_summary_sha256": sha256_file(paths["base_summary"]),
        "base_samples": str(paths["base_samples"]),
        "base_samples_sha256": sha256_file(paths["base_samples"]),
        "trained_summary": str(paths["trained_summary"]),
        "trained_summary_sha256": sha256_file(paths["trained_summary"]),
        "trained_samples": str(paths["trained_samples"]),
        "trained_samples_sha256": sha256_file(paths["trained_samples"]),
    }
    return predecessor, rows, [source_binding, *adjudication_bindings]


def _grouped_surface(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    list[tuple[str, int]],
    list[tuple[str, int]],
]:
    base: dict[str, list[float]] = {}
    trained: dict[str, list[float]] = {}
    base_unknown: list[tuple[str, int]] = []
    trained_unknown: list[tuple[str, int]] = []
    for row in rows:
        if row["eligibility"]["eligible"] is not True:
            continue
        record_id = row["record_id"]
        base[record_id] = [float(sample["effective_reward"]) for sample in row["base"]]
        trained[record_id] = [float(sample["effective_reward"]) for sample in row["trained"]]
        for arm, target in (("base", base_unknown), ("trained", trained_unknown)):
            for sample in row[arm]:
                if sample["uncertain"]:
                    target.append((record_id, sample["sample_idx"]))
    if not base or set(base) != set(trained):
        raise ValueError("eligible score ledger has no complete paired record surface")
    return base, trained, sorted(base_unknown), sorted(trained_unknown)


def _assign(
    grouped: dict[str, list[float]], keys: list[tuple[str, int]], value: float
) -> dict[str, list[float]]:
    result = {record_id: list(rewards) for record_id, rewards in grouped.items()}
    for record_id, sample_idx in keys:
        result[record_id][sample_idx] = value
    return result


def compute_gate(
    *,
    predecessor: dict[str, Any],
    rows: list[dict[str, Any]],
    manifest_path: Path,
    manifest_sha256: str,
    ledger_path: Path,
    ledger_sha256: str,
    min_eligible_coverage: float,
) -> dict[str, Any]:
    if not 0 < min_eligible_coverage <= 1:
        raise ValueError("min eligible coverage must be in (0, 1]")
    base, trained, base_unknown, trained_unknown = _grouped_surface(rows)
    seed = predecessor.get("bootstrap_seed")
    draws = predecessor.get("bootstrap_draws")
    keys, delta, low, high = bootstrap_delta(base, trained, seed, draws)
    worst_base = _assign(base, base_unknown, 1.0)
    worst_trained = _assign(trained, trained_unknown, 0.0)
    _, worst_delta, worst_low, worst_high = bootstrap_delta(
        worst_base, worst_trained, seed, draws
    )
    best_base = _assign(base, base_unknown, 0.0)
    best_trained = _assign(trained, trained_unknown, 1.0)
    _, best_delta, best_low, best_high = bootstrap_delta(best_base, best_trained, seed, draws)
    total_records = len(rows)
    eligible_records = len(keys)
    coverage = eligible_records / total_records
    min_records = max(DEFAULT_TEACHER_MIN_RECORDS, int(predecessor["min_records"]))
    min_delta = float(predecessor["min_delta"])
    requirements = {
        "minimum_records_met": eligible_records >= min_records,
        "minimum_eligible_coverage_met": coverage >= min_eligible_coverage,
        "strict_delta_met": delta > min_delta,
        "positive_bootstrap_lower_bound_met": low > 0,
        "worst_case_strict_delta_met": worst_delta > min_delta,
        "worst_case_positive_bootstrap_lower_bound_met": worst_low > 0,
    }
    passed = all(requirements.values())
    base_accuracy = sum(sum(base[key]) / len(base[key]) for key in keys) / len(keys)
    trained_accuracy = sum(sum(trained[key]) / len(trained[key]) for key in keys) / len(keys)
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "gate": SCORE_LEDGER_GATE_TYPE,
        "gate_strength": "scientific",
        "passed": passed,
        "authorizes_scientific_merge": passed,
        "measurement_policy": MEASUREMENT_POLICY,
        "eligibility_policy": ELIGIBILITY_POLICY,
        "uncertainty_policy": UNCERTAINTY_POLICY,
        "predecessor_gate": predecessor["_manifest_path"],
        "predecessor_gate_sha256": predecessor["_manifest_sha256"],
        "score_ledger_manifest": str(Path(manifest_path).resolve()),
        "score_ledger_manifest_sha256": manifest_sha256,
        "score_ledger": str(Path(ledger_path).resolve()),
        "score_ledger_sha256": ledger_sha256,
        "total_registered_records": total_records,
        "eligible_records": eligible_records,
        "excluded_records": total_records - eligible_records,
        "eligible_coverage": coverage,
        "minimum_eligible_coverage": min_eligible_coverage,
        "shared_records": eligible_records,
        "base_accuracy": base_accuracy,
        "trained_accuracy": trained_accuracy,
        "paired_delta": delta,
        "bootstrap_95_ci": [low, high],
        "verifier_uncertainty_sensitivity": {
            "policy": UNCERTAINTY_POLICY,
            "base_error_samples": len(base_unknown),
            "trained_error_samples": len(trained_unknown),
            "worst_case_for_improvement": {
                "assignment": "base_errors_correct_trained_errors_incorrect",
                "paired_delta": worst_delta,
                "bootstrap_95_ci": [worst_low, worst_high],
            },
            "best_case_for_improvement": {
                "assignment": "base_errors_incorrect_trained_errors_correct",
                "paired_delta": best_delta,
                "bootstrap_95_ci": [best_low, best_high],
            },
        },
        "min_delta": min_delta,
        "min_records": min_records,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "requirements": requirements,
        "base_model": predecessor["base_model"],
        "base_model_revision": predecessor["base_model_revision"],
        "trained_adapter": predecessor["trained_adapter"],
        "trained_adapter_tree_sha256": predecessor["trained_adapter_tree_sha256"],
        "task_sources": predecessor["task_sources"],
        "task_roles": predecessor["task_roles"],
        "claim_boundary": (
            "Post-hoc symbolic-eligible, verifier-aligned teacher reward gap. "
            "This gate does not estimate accuracy on excluded semantic/prose answers and "
            "does not by itself establish student improvement."
        ),
    }


def _manifest_binding(bindings: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = bindings[0]
    adjudications = bindings[1:]
    return source, adjudications


def build_bundle(
    *,
    predecessor_gate_path: Path,
    adjudication_paths: Iterable[Path],
    output_dir: Path,
    min_eligible_coverage: float = DEFAULT_MIN_ELIGIBLE_COVERAGE,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite score-ledger bundle: {output_dir}")
    predecessor, rows, bindings = _surface(predecessor_gate_path, adjudication_paths)
    source_binding, adjudication_bindings = _manifest_binding(bindings)
    git = _git_state()
    if git["worktree_clean"] is not True:
        raise RuntimeError("score-ledger bundle requires a clean Git worktree")

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial.", dir=parent))
    try:
        ledger_path = temporary / "score_ledger.jsonl"
        ledger_text = _canonical_jsonl(rows)
        ledger_path.write_text(ledger_text)
        ledger_sha = sha256_file(ledger_path)
        manifest_path = temporary / "manifest.json"
        manifest = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "record_type": LEDGER_MANIFEST_TYPE,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_policy": MEASUREMENT_POLICY,
            "eligibility_policy": ELIGIBILITY_POLICY,
            "uncertainty_policy": UNCERTAINTY_POLICY,
            "minimum_eligible_coverage": min_eligible_coverage,
            "source_binding": source_binding,
            "adjudications": adjudication_bindings,
            "score_ledger": str(output_dir / ledger_path.name),
            "score_ledger_sha256": ledger_sha,
            "records": len(rows),
            "code": {
                "git": git,
                "builder": str(Path(__file__).resolve()),
                "builder_sha256": sha256_file(Path(__file__).resolve()),
            },
            "post_hoc_disclosure": (
                "The eligibility rule and score-ledger boundary were introduced after "
                "promotion replay failures. Source evaluation bytes remain unchanged."
            ),
        }
        manifest_path.write_text(_canonical_json(manifest) + "\n")
        manifest_sha = sha256_file(manifest_path)
        predecessor = dict(predecessor)
        predecessor["_manifest_path"] = str(Path(predecessor_gate_path).resolve())
        predecessor["_manifest_sha256"] = sha256_file(Path(predecessor_gate_path))
        gate_path = temporary / "gate.json"
        gate = compute_gate(
            predecessor=predecessor,
            rows=rows,
            manifest_path=output_dir / manifest_path.name,
            manifest_sha256=manifest_sha,
            ledger_path=output_dir / ledger_path.name,
            ledger_sha256=ledger_sha,
            min_eligible_coverage=min_eligible_coverage,
        )
        gate_path.write_text(_canonical_json(gate) + "\n")
        for path in (ledger_path, manifest_path, gate_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            path.chmod(0o444)
        os.rename(temporary, output_dir)
        output_dir.chmod(0o555)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "score_ledger": str(output_dir / "score_ledger.jsonl"),
        "score_ledger_sha256": sha256_file(output_dir / "score_ledger.jsonl"),
        "gate": str(output_dir / "gate.json"),
        "gate_sha256": sha256_file(output_dir / "gate.json"),
        "passed": gate["passed"],
    }


def recompute_score_ledger_gate(gate: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a ledger gate from sealed bytes without invoking Math-Verify."""

    if gate.get("gate") != SCORE_LEDGER_GATE_TYPE:
        raise ValueError("not a score-ledger teacher gate")
    manifest_path = Path(gate["score_ledger_manifest"]).resolve()
    ledger_path = Path(gate["score_ledger"]).resolve()
    if sha256_file(manifest_path) != gate.get("score_ledger_manifest_sha256"):
        raise ValueError("score-ledger manifest changed")
    if sha256_file(ledger_path) != gate.get("score_ledger_sha256"):
        raise ValueError("score ledger changed")
    manifest = _json_object(manifest_path, "score-ledger manifest")
    if manifest.get("record_type") != LEDGER_MANIFEST_TYPE:
        raise ValueError("unsupported score-ledger manifest")
    if manifest.get("measurement_policy") != MEASUREMENT_POLICY:
        raise ValueError("score-ledger measurement policy changed")
    code = manifest.get("code")
    if not isinstance(code, dict) or code.get("builder_sha256") != sha256_file(Path(__file__)):
        raise ValueError("score-ledger builder differs from manifest custody")
    source = manifest.get("source_binding")
    if not isinstance(source, dict):
        raise ValueError("score-ledger manifest lacks source binding")
    for key in (
        "predecessor_gate",
        "task_file",
        "base_summary",
        "base_samples",
        "trained_summary",
        "trained_samples",
    ):
        path = Path(source[key]).resolve()
        if sha256_file(path) != source.get(f"{key}_sha256"):
            raise ValueError(f"score-ledger source binding changed: {key}")
    adjudication_paths: list[Path] = []
    for item in manifest.get("adjudications", []):
        if not isinstance(item, dict):
            raise ValueError("score-ledger adjudication binding is malformed")
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item.get("sha256"):
            raise ValueError("score-ledger adjudication changed")
        adjudication_paths.append(path)
    predecessor, rows, _ = _surface(Path(source["predecessor_gate"]), adjudication_paths)
    expected_ledger = _canonical_jsonl(rows).encode("utf-8")
    if hashlib.sha256(expected_ledger).hexdigest() != gate.get("score_ledger_sha256"):
        raise ValueError("score ledger does not equal deterministic reconstruction")
    if ledger_path.read_bytes() != expected_ledger:
        raise ValueError("score ledger bytes differ from deterministic reconstruction")
    predecessor = dict(predecessor)
    predecessor["_manifest_path"] = str(Path(source["predecessor_gate"]).resolve())
    predecessor["_manifest_sha256"] = source["predecessor_gate_sha256"]
    return compute_gate(
        predecessor=predecessor,
        rows=rows,
        manifest_path=manifest_path,
        manifest_sha256=gate["score_ledger_manifest_sha256"],
        ledger_path=ledger_path,
        ledger_sha256=gate["score_ledger_sha256"],
        min_eligible_coverage=float(manifest["minimum_eligible_coverage"]),
    )


def validate_score_ledger_gate_for_merge(
    gate_path: Path,
    *,
    base_model: str,
    base_revision: str,
    adapter: Path,
) -> dict[str, Any]:
    gate_path = Path(gate_path).resolve()
    gate = _json_object(gate_path, "score-ledger teacher gate")
    if gate.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported score-ledger gate schema")
    if gate.get("gate") != SCORE_LEDGER_GATE_TYPE:
        raise ValueError("merge did not receive a score-ledger teacher gate")
    if gate.get("passed") is not True or gate.get("authorizes_scientific_merge") is not True:
        raise ValueError("score-ledger teacher gate did not pass")
    if gate.get("base_model") != base_model or gate.get("base_model_revision") != base_revision:
        raise ValueError("score-ledger gate model identity differs from merge request")
    adapter = Path(adapter).resolve()
    if str(adapter) != str(Path(gate.get("trained_adapter", "")).resolve()):
        raise ValueError("score-ledger gate adapter path differs from merge request")
    adapter_hash = sha256_tree(adapter)
    if adapter_hash != gate.get("trained_adapter_tree_sha256"):
        raise ValueError("score-ledger gate adapter bytes changed")
    recomputed = recompute_score_ledger_gate(gate)
    if recomputed != gate:
        changed = sorted(
            key for key in set(gate) | set(recomputed) if gate.get(key) != recomputed.get(key)
        )
        raise ValueError(
            "score-ledger gate differs from deterministic non-verifier recomputation; "
            f"changed_fields={changed[:20]}"
        )
    return {
        "gate": gate,
        "manifest": str(gate_path),
        "manifest_sha256": sha256_file(gate_path),
        "adapter": str(adapter),
        "adapter_tree_sha256": adapter_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-gate", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-eligible-coverage", type=float, default=DEFAULT_MIN_ELIGIBLE_COVERAGE
    )
    args = parser.parse_args()
    result = build_bundle(
        predecessor_gate_path=args.predecessor_gate,
        adjudication_paths=args.adjudication,
        output_dir=args.output_dir,
        min_eligible_coverage=args.min_eligible_coverage,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
