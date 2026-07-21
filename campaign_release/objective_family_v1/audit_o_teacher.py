#!/usr/bin/env python3
"""Seal an independent custody receipt for the fresh d89 O teacher.

This is an external campaign auditor, not a training or gate implementation.
It runs only after two byte-identical passing gate computations and a completed
teacher merge. It reopens the tracked gate/checkpoint validators, repeats the
physical hashes independently, binds terminal Slurm accounting, seals the
checkpoint and stdout read-only, and atomically publishes one no-clobber
receipt. It never authorizes a failed gate or changes a scientific threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


EXPECTED_COMMIT = "d89ba3d7be728d9ee3197f37d8a8836a4a9640c5"
EXPECTED_MODEL = "Qwen/Qwen3-8B"
EXPECTED_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_REPO = Path("/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math")
EXPECTED_TRAIN_ENV = "/engrfs/project/jacobsn/hiqbal/envs/opd_math_train"
EXPECTED_HF_HOME = "/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache"
EXPECTED_DATA_ROOT = "/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1_canonical_reviewed_19b24c2"
EXPECTED_GATE_STDOUT = "/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_gate_%j.out"
EXPECTED_MERGE_STDOUT = "/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_merge_%j.out"
GATE_LAUNCHER = "scripts/hpc/slurm_opd_math_quality_gate.sh"
MERGE_LAUNCHER = "scripts/hpc/slurm_opd_math_merge_teacher.sh"
EXPECTED_PRIMARY_GATE = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/gates/teacher_gap/O_gap_d89ba3d_v1.json")
EXPECTED_INDEPENDENT_GATE = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/gates/teacher_gap/O_gap_d89ba3d_v1_independent.json")
EXPECTED_CHECKPOINT = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/teachers/O/merged_d89ba3d_v1")
EXPECTED_AUDIT_RECEIPT = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/audits/objective_family/O_teacher_d89ba3d_v1.json")
EXPECTED_ADAPTER = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/teachers/O/run_108609/final_adapter")
EXPECTED_TEACHER_RUN_MANIFEST = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/teachers/O/run_108609/run_manifest.json")
EXPECTED_PROVENANCE_KEYS = {
    "schema_version", "schema", "status", "base_model", "base_revision", "adapter",
    "adapter_tree_sha256", "teacher_gap_manifest", "teacher_gap_manifest_sha256",
    "prepared_manifest", "prepared_manifest_sha256", "teacher_run_manifest",
    "teacher_run_manifest_sha256", "teacher_training_plan", "teacher_training_plan_sha256",
    "teacher_training_plan_config_sha256", "teacher_training_config_sha256",
    "teacher_training_packages", "teacher_training_environment", "teacher_trainer_state",
    "teacher_trainer_state_sha256", "teacher_trainer_log_history",
    "teacher_trainer_log_history_sha256", "teacher_train_metrics",
    "teacher_train_metrics_sha256", "teacher_trainer_log_max_step", "source_manifest",
    "source_manifest_sha256", "task_file_sha256", "task_sources", "task_roles", "decoding",
    "merge_code", "output_checkpoint", "output_checkpoint_tree_sha256", "tree_hash_algorithm",
    "tree_hash_excludes",
}


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def independent_tree_sha256(
    root: str | Path, *, exclude_relative_paths: tuple[str, ...] = ()
) -> str:
    raw_tree = Path(root)
    expect(not raw_tree.is_symlink(), f"checkpoint tree root is a symlink: {raw_tree}")
    tree = raw_tree.resolve()
    expect(tree.is_dir(), f"invalid checkpoint tree: {tree}")
    excluded = {Path(value).as_posix() for value in exclude_relative_paths}
    files: list[Path] = []
    for candidate in tree.rglob("*"):
        mode = candidate.lstat().st_mode
        expect(not stat.S_ISLNK(mode), f"checkpoint tree contains symlink: {candidate}")
        expect(
            stat.S_ISREG(mode) or stat.S_ISDIR(mode),
            f"checkpoint tree contains unbound special node: {candidate}",
        )
        relative = candidate.relative_to(tree).as_posix()
        if stat.S_ISREG(mode) and relative not in excluded:
            files.append(candidate)
    files.sort(key=lambda value: value.relative_to(tree).as_posix())
    expect(bool(files), "checkpoint tree is empty")
    digest = hashlib.sha256()
    digest.update(b"opd-math-tree-v1\0")
    for candidate in files:
        relative = candidate.relative_to(tree).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def regular_readonly(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    expect(not candidate.is_symlink() and candidate.is_file(), f"{label} must be a regular file")
    expect(
        candidate.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
        f"{label} must be sealed read-only",
    )
    return candidate.resolve()


def regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    expect(not candidate.is_symlink() and candidate.is_file(), f"{label} must be a regular file")
    return candidate.resolve()


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expect(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def git_state(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return {"commit": commit, "tracked_clean": not status.strip()}


def parse_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_tres(value: str, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        key, separator, raw_value = item.partition("=")
        expect(bool(separator) and key and raw_value and key not in result, f"invalid {label}")
        result[key] = raw_value
    return result


def parse_submit_line(value: str, *, repo: Path, launcher: str) -> dict[str, str]:
    tokens = shlex.split(value)
    expect(tokens and tokens[0] == "sbatch" and tokens[-1] == launcher, "Slurm submit line launcher drifted")
    expect(tokens.count("--parsable") == 1, "Slurm submission must contain exactly one --parsable")
    chdir = [token.removeprefix("--chdir=") for token in tokens if token.startswith("--chdir=")]
    exports = [token.removeprefix("--export=") for token in tokens if token.startswith("--export=")]
    expect(chdir == [str(repo)] and len(exports) == 1, "Slurm submission chdir/export drifted")
    raw_export = exports[0].split(",")
    expect(raw_export and raw_export[0] == "ALL", "Slurm submission must export from ALL")
    result: dict[str, str] = {}
    for item in raw_export[1:]:
        key, separator, raw_value = item.partition("=")
        expect(bool(separator) and key and key not in result, "Slurm export map is malformed")
        result[key] = raw_value
    allowed_options = {"--parsable", f"--chdir={repo}", f"--export={exports[0]}"}
    expect(set(tokens[1:-1]) == allowed_options and len(tokens[1:-1]) == 3, "Slurm submission has unregistered options")
    return result


def query_terminal_job(
    *,
    job_id: str,
    stdout_path: Path,
    expected_name: str,
    stdout_template: str,
    repo: Path,
    launcher: str,
    expected_exports: Mapping[str, str],
    expected_cpu: int,
    expected_mem: str,
    expected_minutes: int,
) -> dict[str, Any]:
    expect(re.fullmatch(r"[1-9][0-9]*", job_id) is not None, "scheduler job ID is invalid")
    fields = (
        "JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Submit,Start,End,"
        "AllocTRES,ReqTRES,StdOut,WorkDir,SubmitLine,Partition,Account,"
        "TimelimitRaw,NCPUS,NNodes"
    )
    command = ["sacct", "-X", "-n", "-P", "-j", job_id, f"--format={fields}"]
    raw = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    matches: list[list[str]] = []
    for line in raw.splitlines():
        parts = line.split("|", 17)
        if len(parts) == 18 and parts[0] == job_id:
            matches.append(parts)
    expect(len(matches) == 1, "sacct lacks one exact top-level scheduler row")
    (
        job, name, state, exit_code, elapsed, submit, start, end,
        alloc_tres, req_tres, recorded_stdout, workdir, submit_line,
        partition, account, time_limit, ncpus, nnodes,
    ) = matches[0]
    normalized = state.split()[0].split("+")[0]
    expect(normalized == "COMPLETED" and exit_code == "0:0", "scheduler row is not successful")
    expect(name == expected_name and elapsed.isdigit(), "scheduler job identity drifted")
    expect(workdir == str(repo) and partition == "general-cpu" and account == "engr-lab-jacobsn", "scheduler lane drifted")
    expect(time_limit == str(expected_minutes) and ncpus == str(expected_cpu) and nnodes == "1", "scheduler resource identity drifted")
    for label, value in (("requested TRES", req_tres), ("allocated TRES", alloc_tres)):
        tres = parse_tres(value, label)
        expect(tres.get("cpu") == str(expected_cpu) and tres.get("mem") == expected_mem and tres.get("node") == "1", f"{label} drifted")
    expect(recorded_stdout == stdout_template and recorded_stdout.count("%j") == 1, "scheduler stdout template drifted")
    expect("%" not in recorded_stdout.replace("%j", ""), "scheduler stdout contains an unregistered template token")
    expanded_stdout = Path(recorded_stdout.replace("%j", job_id))
    expect(expanded_stdout == stdout_path, "scheduler stdout path differs from expanded %j template")
    export_map = parse_submit_line(submit_line, repo=repo, launcher=launcher)
    expect(export_map == dict(expected_exports), "scheduler exported artifact identity drifted")
    parse_utc(submit, "scheduler submit")
    parse_utc(start, "scheduler start")
    parse_utc(end, "scheduler end")
    return {
        "job_id": job,
        "job_name": name,
        "state": normalized,
        "state_raw": state,
        "exit_code": exit_code,
        "elapsed_seconds": int(elapsed),
        "submit": submit,
        "start": start,
        "end": end,
        "alloc_tres": alloc_tres,
        "req_tres": req_tres,
        "stdout_template": recorded_stdout,
        "stdout": str(stdout_path),
        "workdir": workdir,
        "submit_line": submit_line,
        "partition": partition,
        "account": account,
        "time_limit_minutes": int(time_limit),
        "ncpus": int(ncpus),
        "nnodes": int(nnodes),
        "sacct_raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


def seal_tree_readonly(path: str | Path) -> None:
    raw_root = Path(path)
    expect(not raw_root.is_symlink(), "cannot seal a symlinked checkpoint tree")
    root = raw_root.resolve()
    expect(root.is_dir(), "cannot seal a missing checkpoint tree")
    for candidate in root.rglob("*"):
        mode = candidate.lstat().st_mode
        expect(not stat.S_ISLNK(mode), f"cannot seal tree containing symlink: {candidate}")
        expect(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f"cannot seal tree containing special node: {candidate}")
        current = stat.S_IMODE(mode)
        os.chmod(candidate, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    current = stat.S_IMODE(root.stat().st_mode)
    os.chmod(root, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def assert_tree_readonly(path: str | Path) -> None:
    root = Path(path)
    expect(root.is_dir() and not root.is_symlink(), "checkpoint tree disappeared after sealing")
    for candidate in (root, *root.rglob("*")):
        mode = candidate.lstat().st_mode
        expect(not stat.S_ISLNK(mode), f"sealed checkpoint contains symlink: {candidate}")
        expect(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f"sealed checkpoint contains special node: {candidate}")
        expect(
            mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
            f"checkpoint descendant remains writable: {candidate}",
        )


def write_new(path: str | Path, payload: Mapping[str, Any]) -> Path:
    raw_target = Path(path)
    expect(not raw_target.exists() and not raw_target.is_symlink(), f"audit output already exists: {raw_target}")
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    expect(not raw_target.parent.is_symlink(), "audit output parent may not be a symlink")
    target = raw_target.parent.resolve() / raw_target.name
    expect(not target.exists() and not target.is_symlink(), f"audit output already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.partial.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, target, follow_symlinks=False)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def copy_new_readonly(source: Path, destination: Path) -> Path:
    expect(source.is_file() and not source.is_symlink(), "audit log source is not regular")
    expect(not destination.exists() and not destination.is_symlink(), "audit log archive path is not fresh")
    parent = destination.parent
    expect(parent.is_dir() and not parent.is_symlink(), "audit log archive parent is not a real directory")
    parent_inode = (parent.lstat().st_dev, parent.lstat().st_ino)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial.", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fchmod(writer.fileno(), 0o444)
            os.fsync(writer.fileno())
        expect(sha256_file(temporary) == sha256_file(source), "audit log archive hash drifted")
        os.link(temporary, destination, follow_symlinks=False)
        expect(
            parent.is_dir()
            and not parent.is_symlink()
            and (parent.lstat().st_dev, parent.lstat().st_ino) == parent_inode,
            "audit log archive parent changed during copy",
        )
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def expected_gate_exports(gate: Mapping[str, Any], gate_path: Path, repo: Path) -> dict[str, str]:
    return {
        "OPD_MATH_REPO": str(repo),
        "OPD_MATH_TRAIN_ENV": EXPECTED_TRAIN_ENV,
        "OPD_MATH_DATA_ROOT": EXPECTED_DATA_ROOT,
        "OPD_MATH_GATE_KIND": "teacher_gap",
        "OPD_MATH_GATE_SOURCE": "O",
        "OPD_MATH_GATE_STRENGTH": "scientific",
        "OPD_MATH_GATE_BASE_SUMMARY": str(Path(str(gate["base_summary"])).resolve()),
        "OPD_MATH_GATE_BASE_SAMPLES": str(Path(str(gate["base_samples"])).resolve()),
        "OPD_MATH_GATE_TRAINED_SUMMARY": str(Path(str(gate["trained_summary"])).resolve()),
        "OPD_MATH_GATE_TRAINED_SAMPLES": str(Path(str(gate["trained_samples"])).resolve()),
        "OPD_MATH_GATE_BASE_MODEL": EXPECTED_MODEL,
        "OPD_MATH_GATE_BASE_REVISION": EXPECTED_REVISION,
        "OPD_MATH_GATE_TRAINED_ADAPTER": str(Path(str(gate["trained_adapter"])).resolve()),
        "OPD_MATH_GATE_TEACHER_RUN_MANIFEST": str(Path(str(gate["teacher_run_manifest"])).resolve()),
        "OPD_MATH_GATE_BOOTSTRAP_DRAWS": "10000",
        "OPD_MATH_SEED": "0",
        "OPD_MATH_GATE_OUTPUT": str(gate_path),
    }


def expected_merge_exports(
    *, repo: Path, gate_path: Path, adapter: Path, checkpoint: Path
) -> dict[str, str]:
    return {
        "OPD_MATH_REPO": str(repo),
        "OPD_MATH_TRAIN_ENV": EXPECTED_TRAIN_ENV,
        "OPD_MATH_HF_HOME": EXPECTED_HF_HOME,
        "OPD_MATH_MERGE_BASE_MODEL": EXPECTED_MODEL,
        "OPD_MATH_MERGE_BASE_REVISION": EXPECTED_REVISION,
        "OPD_MATH_MERGE_ADAPTER": str(adapter),
        "OPD_MATH_MERGE_GATE": str(gate_path),
        "OPD_MATH_MERGE_OUTPUT": str(checkpoint),
    }


def exact_success_marker(path: Path, marker: str, label: str) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    expect(sum(line == marker for line in lines) == 1, f"{label} lacks one exact artifact-bound success marker")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    raw_repo = Path(args.repo)
    expect(not raw_repo.is_symlink(), "audit repo may not be a symlink")
    repo = raw_repo.resolve()
    expect(repo == EXPECTED_REPO, "audit requires the canonical EIT d89 checkout")
    git_start = git_state(repo)
    expect(git_start == {"commit": EXPECTED_COMMIT, "tracked_clean": True}, "audit requires clean d89 checkout")
    auditor_path = regular_readonly(__file__, "O teacher auditor")
    auditor_hash_start = sha256_file(auditor_path)
    auditor_inode_start = (auditor_path.stat().st_dev, auditor_path.stat().st_ino)
    raw_output = Path(args.output)
    expect(not raw_output.exists() and not raw_output.is_symlink(), "audit output must be a fresh path before any sealing")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    expect(raw_output.parent.is_dir() and not raw_output.parent.is_symlink(), "audit output parent must be a real directory")
    canonical_output = raw_output.parent.resolve() / raw_output.name
    expect(not canonical_output.exists() and not canonical_output.is_symlink(), "canonical audit output must be fresh before any sealing")
    expect(canonical_output == EXPECTED_AUDIT_RECEIPT, "O teacher audit receipt path differs from the frozen boundary")
    archive_root = canonical_output.with_name(canonical_output.name + ".logs")
    expect(not archive_root.exists() and not archive_root.is_symlink(), "audit log archive root must be fresh")
    sys.path.insert(0, str(repo))
    from scripts.opd.objective_family_inputs import (  # type: ignore
        canonical_json_sha256 as tracked_canonical_json_sha256,
        sha256_tree as tracked_tree_sha256,
    )
    from scripts.opd.objective_family_preregistration import _validate_teacher  # type: ignore
    from scripts.opd.opd_train import _validate_teacher_provenance  # type: ignore
    from scripts.opd_math.quality_gates import recompute_teacher_gate  # type: ignore

    primary_path = regular_file(args.primary_gate, "primary O gate")
    independent_path = regular_file(args.independent_gate, "independent O gate")
    expect(
        primary_path == EXPECTED_PRIMARY_GATE
        and independent_path == EXPECTED_INDEPENDENT_GATE,
        "O teacher gate paths differ from the frozen boundary",
    )
    expect(primary_path != independent_path, "primary and independent O gate paths must differ")
    primary_stat = primary_path.stat()
    independent_stat = independent_path.stat()
    gate_inodes_start = {
        "primary": (primary_stat.st_dev, primary_stat.st_ino),
        "independent": (independent_stat.st_dev, independent_stat.st_ino),
    }
    expect(
        (primary_stat.st_dev, primary_stat.st_ino) != (independent_stat.st_dev, independent_stat.st_ino),
        "primary and independent O gates must not be hard links",
    )
    expect(primary_path.read_bytes() == independent_path.read_bytes(), "primary and independent O gates differ")
    primary = load_json(primary_path, "primary O gate")
    independent = load_json(independent_path, "independent O gate")
    for label, gate in (("primary", primary), ("independent", independent)):
        original = dict(gate)
        original.pop("manifest_sha256", None)
        expect(recompute_teacher_gate(original) == original, f"{label} O gate does not recompute")
        expect(
            gate.get("passed") is True
            and gate.get("authorizes_scientific_merge") is True
            and gate.get("task_sources") == ["O"]
            and gate.get("evaluation_git_commit") == EXPECTED_COMMIT,
            f"{label} O gate is not a passing same-commit gate",
        )
        expect(
            Path(str(gate.get("trained_adapter"))).resolve() == EXPECTED_ADAPTER
            and Path(str(gate.get("teacher_run_manifest"))).resolve()
            == EXPECTED_TEACHER_RUN_MANIFEST,
            f"{label} O gate is not bound to fresh teacher job 108609",
        )

    expect(args.primary_gate_job_id != args.independent_gate_job_id, "O gate computations must use distinct Slurm jobs")
    primary_gate_stdout = regular_file(args.primary_gate_stdout, "primary O gate stdout")
    independent_gate_stdout = regular_file(args.independent_gate_stdout, "independent O gate stdout")
    expect(primary_gate_stdout != independent_gate_stdout, "O gate stdout paths must differ")
    expect(
        (primary_gate_stdout.stat().st_dev, primary_gate_stdout.stat().st_ino)
        != (independent_gate_stdout.stat().st_dev, independent_gate_stdout.stat().st_ino),
        "O gate stdout files must not be hard links",
    )
    gate_stdout_inodes_start = {
        "primary": (primary_gate_stdout.stat().st_dev, primary_gate_stdout.stat().st_ino),
        "independent": (independent_gate_stdout.stat().st_dev, independent_gate_stdout.stat().st_ino),
    }
    exact_success_marker(
        primary_gate_stdout,
        f"PASS gate computation completed; inspect passed/strength before use: {primary_path}",
        "primary O gate stdout",
    )
    exact_success_marker(
        independent_gate_stdout,
        f"PASS gate computation completed; inspect passed/strength before use: {independent_path}",
        "independent O gate stdout",
    )
    primary_gate_scheduler = query_terminal_job(
        job_id=args.primary_gate_job_id,
        stdout_path=primary_gate_stdout,
        expected_name="opd_math_gate",
        stdout_template=EXPECTED_GATE_STDOUT,
        repo=repo,
        launcher=GATE_LAUNCHER,
        expected_exports=expected_gate_exports(primary, primary_path, repo),
        expected_cpu=2,
        expected_mem="8G",
        expected_minutes=240,
    )
    independent_gate_scheduler = query_terminal_job(
        job_id=args.independent_gate_job_id,
        stdout_path=independent_gate_stdout,
        expected_name="opd_math_gate",
        stdout_template=EXPECTED_GATE_STDOUT,
        repo=repo,
        launcher=GATE_LAUNCHER,
        expected_exports=expected_gate_exports(independent, independent_path, repo),
        expected_cpu=2,
        expected_mem="8G",
        expected_minutes=240,
    )

    raw_checkpoint = Path(args.checkpoint)
    expect(not raw_checkpoint.is_symlink(), "merged O checkpoint may not be a symlink")
    checkpoint = raw_checkpoint.resolve()
    expect(checkpoint.is_dir(), "merged O checkpoint is missing")
    expect(checkpoint == EXPECTED_CHECKPOINT, "merged O checkpoint path differs from the frozen boundary")
    provenance_path = checkpoint / "merge_provenance.json"
    expect(provenance_path.is_file() and not provenance_path.is_symlink(), "merge provenance is missing")
    provenance = load_json(provenance_path, "merge provenance")
    expect(set(provenance) == EXPECTED_PROVENANCE_KEYS, "merge provenance schema keys drifted")
    checkpoint_hash = independent_tree_sha256(
        checkpoint, exclude_relative_paths=("merge_provenance.json",)
    )
    expect(
        tracked_tree_sha256(checkpoint, exclude_relative_paths=("merge_provenance.json",))
        == checkpoint_hash,
        "independent and tracked checkpoint tree hashes disagree",
    )
    identity = {
        "teacher_source": "O",
        "base_model": EXPECTED_MODEL,
        "base_revision": EXPECTED_REVISION,
        "teacher_gap_manifest": str(primary_path),
        "teacher_gap_manifest_sha256": sha256_file(primary_path),
        "teacher_gap_payload_sha256": canonical_json_sha256(primary),
        "merged_checkpoint": str(checkpoint),
        "merged_checkpoint_tree_sha256": checkpoint_hash,
        "merge_provenance_manifest_sha256": sha256_file(provenance_path),
        "merge_provenance_payload_sha256": canonical_json_sha256(provenance),
    }
    expect(
        tracked_canonical_json_sha256(primary) == identity["teacher_gap_payload_sha256"]
        and tracked_canonical_json_sha256(provenance) == identity["merge_provenance_payload_sha256"],
        "independent and tracked canonical JSON hashes disagree",
    )
    strong_args = SimpleNamespace(
        teacher_base_model=EXPECTED_MODEL,
        teacher_base_revision=EXPECTED_REVISION,
        teacher_checkpoint=str(checkpoint),
    )
    primary_for_provenance = dict(primary)
    primary_for_provenance["manifest_sha256"] = sha256_file(primary_path)
    strong_provenance = _validate_teacher_provenance(
        str(provenance_path), primary_for_provenance, strong_args
    )
    expect(strong_provenance.get("manifest_sha256") == sha256_file(provenance_path), "strong teacher provenance hash drifted")

    adapter = Path(str(primary["trained_adapter"]))
    expect(not adapter.is_symlink(), "gate-bound teacher adapter may not be a symlink")
    adapter = adapter.resolve()
    expect(adapter.is_dir(), "gate-bound teacher adapter is missing")
    merge_stdout = regular_file(args.merge_stdout, "merge stdout")
    merge_stdout_inode_start = (merge_stdout.stat().st_dev, merge_stdout.stat().st_ino)
    stdout_identities = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in (primary_gate_stdout, independent_gate_stdout, merge_stdout)
    }
    expect(len(stdout_identities) == 3, "gate and merge stdout files must have distinct inodes")
    exact_success_marker(
        merge_stdout,
        f"PASS scientifically gated teacher merge: {checkpoint}",
        "merge stdout",
    )
    merge_scheduler = query_terminal_job(
        job_id=args.merge_job_id,
        stdout_path=merge_stdout,
        expected_name="opd_math_merge",
        stdout_template=EXPECTED_MERGE_STDOUT,
        repo=repo,
        launcher=MERGE_LAUNCHER,
        expected_exports=expected_merge_exports(
            repo=repo,
            gate_path=primary_path,
            adapter=adapter,
            checkpoint=checkpoint,
        ),
        expected_cpu=8,
        expected_mem="64G",
        expected_minutes=120,
    )
    latest_gate_end = max(
        parse_utc(primary_gate_scheduler["end"], "primary gate end"),
        parse_utc(independent_gate_scheduler["end"], "independent gate end"),
    )
    expect(
        latest_gate_end <= parse_utc(merge_scheduler["submit"], "merge submit"),
        "teacher merge was submitted before both independent O gate jobs completed",
    )

    before = {
        "primary_gate": sha256_file(primary_path),
        "independent_gate": sha256_file(independent_path),
        "primary_gate_stdout": sha256_file(primary_gate_stdout),
        "independent_gate_stdout": sha256_file(independent_gate_stdout),
        "checkpoint_tree": checkpoint_hash,
        "provenance": sha256_file(provenance_path),
        "merge_stdout": sha256_file(merge_stdout),
        "auditor": auditor_hash_start,
    }

    seal_tree_readonly(checkpoint)
    for path in (primary_path, independent_path, primary_gate_stdout, independent_gate_stdout, merge_stdout):
        os.chmod(path, 0o444)
    assert_tree_readonly(checkpoint)
    for label, path in (
        ("primary O gate", primary_path),
        ("independent O gate", independent_path),
        ("primary O gate stdout", primary_gate_stdout),
        ("independent O gate stdout", independent_gate_stdout),
        ("merge stdout", merge_stdout),
    ):
        sealed = regular_readonly(path, label)
        expect(stat.S_IMODE(sealed.lstat().st_mode) == 0o444, f"{label} mode is not exactly 0444")

    expect(primary_path.read_bytes() == independent_path.read_bytes(), "O gates changed while sealing")
    primary_final = load_json(primary_path, "primary O gate after sealing")
    independent_final = load_json(independent_path, "independent O gate after sealing")
    for label, gate in (("primary", primary_final), ("independent", independent_final)):
        original = dict(gate)
        original.pop("manifest_sha256", None)
        expect(recompute_teacher_gate(original) == original, f"{label} O gate changed before publication")
    expect(_validate_teacher(identity, commit=EXPECTED_COMMIT) == identity, "teacher identity changed before publication")
    primary_final_for_provenance = dict(primary_final)
    primary_final_for_provenance["manifest_sha256"] = sha256_file(primary_path)
    _validate_teacher_provenance(
        str(provenance_path), primary_final_for_provenance, strong_args
    )
    exact_success_marker(primary_gate_stdout, f"PASS gate computation completed; inspect passed/strength before use: {primary_path}", "primary O gate stdout")
    exact_success_marker(independent_gate_stdout, f"PASS gate computation completed; inspect passed/strength before use: {independent_path}", "independent O gate stdout")
    exact_success_marker(merge_stdout, f"PASS scientifically gated teacher merge: {checkpoint}", "merge stdout")
    final_primary_gate_scheduler = query_terminal_job(
        job_id=args.primary_gate_job_id,
        stdout_path=primary_gate_stdout,
        expected_name="opd_math_gate",
        stdout_template=EXPECTED_GATE_STDOUT,
        repo=repo,
        launcher=GATE_LAUNCHER,
        expected_exports=expected_gate_exports(primary_final, primary_path, repo),
        expected_cpu=2,
        expected_mem="8G",
        expected_minutes=240,
    )
    final_independent_gate_scheduler = query_terminal_job(
        job_id=args.independent_gate_job_id,
        stdout_path=independent_gate_stdout,
        expected_name="opd_math_gate",
        stdout_template=EXPECTED_GATE_STDOUT,
        repo=repo,
        launcher=GATE_LAUNCHER,
        expected_exports=expected_gate_exports(independent_final, independent_path, repo),
        expected_cpu=2,
        expected_mem="8G",
        expected_minutes=240,
    )
    expect(
        final_primary_gate_scheduler == primary_gate_scheduler
        and final_independent_gate_scheduler == independent_gate_scheduler,
        "terminal O gate scheduler rows changed before publication",
    )
    final_scheduler = query_terminal_job(
        job_id=args.merge_job_id,
        stdout_path=merge_stdout,
        expected_name="opd_math_merge",
        stdout_template=EXPECTED_MERGE_STDOUT,
        repo=repo,
        launcher=MERGE_LAUNCHER,
        expected_exports=expected_merge_exports(repo=repo, gate_path=primary_path, adapter=adapter, checkpoint=checkpoint),
        expected_cpu=8,
        expected_mem="64G",
        expected_minutes=120,
    )
    expect(final_scheduler == merge_scheduler, "terminal merge scheduler row changed before publication")
    archive_root.mkdir(parents=False, exist_ok=False)
    archive_root_mode = archive_root.lstat().st_mode
    expect(stat.S_ISDIR(archive_root_mode) and not stat.S_ISLNK(archive_root_mode), "audit archive root is not a real directory")
    archive_root_inode = (archive_root.lstat().st_dev, archive_root.lstat().st_ino)
    archived_primary_gate_stdout = copy_new_readonly(
        primary_gate_stdout, archive_root / "primary_gate_stdout.log"
    )
    archived_independent_gate_stdout = copy_new_readonly(
        independent_gate_stdout, archive_root / "independent_gate_stdout.log"
    )
    archived_merge_stdout = copy_new_readonly(
        merge_stdout, archive_root / "merge_stdout.log"
    )
    os.chmod(archive_root, 0o555)
    archive_fd = os.open(archive_root, os.O_RDONLY)
    try:
        os.fsync(archive_fd)
    finally:
        os.close(archive_fd)
    after = {
        "primary_gate": sha256_file(primary_path),
        "independent_gate": sha256_file(independent_path),
        "primary_gate_stdout": sha256_file(primary_gate_stdout),
        "independent_gate_stdout": sha256_file(independent_gate_stdout),
        "checkpoint_tree": independent_tree_sha256(checkpoint, exclude_relative_paths=("merge_provenance.json",)),
        "provenance": sha256_file(provenance_path),
        "merge_stdout": sha256_file(merge_stdout),
        "auditor": sha256_file(__file__),
    }
    expect(after == before, "teacher audit inputs changed before receipt publication")
    gate_inodes_final = {
        "primary": (primary_path.stat().st_dev, primary_path.stat().st_ino),
        "independent": (independent_path.stat().st_dev, independent_path.stat().st_ino),
    }
    gate_stdout_inodes_final = {
        "primary": (primary_gate_stdout.stat().st_dev, primary_gate_stdout.stat().st_ino),
        "independent": (independent_gate_stdout.stat().st_dev, independent_gate_stdout.stat().st_ino),
    }
    expect(gate_inodes_final == gate_inodes_start, "O gate inode custody changed before publication")
    expect(
        gate_stdout_inodes_final == gate_stdout_inodes_start
        and (merge_stdout.stat().st_dev, merge_stdout.stat().st_ino) == merge_stdout_inode_start,
        "O gate/merge stdout inode custody changed before publication",
    )
    expect(len(set(gate_inodes_final.values())) == 2, "O gate inodes collapsed before publication")
    expect(
        len({*gate_stdout_inodes_final.values(), (merge_stdout.stat().st_dev, merge_stdout.stat().st_ino)}) == 3,
        "O gate/merge stdout inodes collapsed before publication",
    )
    expect(
        (auditor_path.stat().st_dev, auditor_path.stat().st_ino) == auditor_inode_start,
        "O teacher auditor inode changed before publication",
    )
    for archived, source_hash in (
        (archived_primary_gate_stdout, after["primary_gate_stdout"]),
        (archived_independent_gate_stdout, after["independent_gate_stdout"]),
        (archived_merge_stdout, after["merge_stdout"]),
    ):
        sealed_archive = regular_readonly(archived, "archived audit stdout")
        expect(
            sha256_file(sealed_archive) == source_hash
            and stat.S_IMODE(sealed_archive.lstat().st_mode) == 0o444,
            "archived audit stdout hash/mode drifted",
        )
    archive_root_final_mode = archive_root.lstat().st_mode
    expect(
        stat.S_ISDIR(archive_root_final_mode)
        and not stat.S_ISLNK(archive_root_final_mode)
        and (archive_root.lstat().st_dev, archive_root.lstat().st_ino) == archive_root_inode
        and stat.S_IMODE(archive_root_final_mode) == 0o555,
        "audit log archive root identity/mode drifted",
    )
    assert_tree_readonly(checkpoint)
    for label, path in (
        ("O teacher auditor", auditor_path),
        ("primary O gate", primary_path),
        ("independent O gate", independent_path),
        ("primary O gate live stdout", primary_gate_stdout),
        ("independent O gate live stdout", independent_gate_stdout),
        ("O teacher merge live stdout", merge_stdout),
        ("primary O gate archived stdout", archived_primary_gate_stdout),
        ("independent O gate archived stdout", archived_independent_gate_stdout),
        ("O teacher merge archived stdout", archived_merge_stdout),
    ):
        regular_readonly(path, label)
    git_end = git_state(repo)
    expect(git_end == git_start, "d89 Git custody changed during teacher audit")
    return {
        "schema_version": 1,
        "receipt": "opd_math_objective_family_o_teacher_independent_audit_v1",
        "status": "passed_and_sealed",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "git": git_end,
        "auditor": {"path": str(auditor_path), "sha256": after["auditor"]},
        "primary_gate": {
            "path": str(primary_path),
            "sha256": after["primary_gate"],
            "stdout": {
                "scheduler_path": str(primary_gate_stdout),
                "archive_path": str(archived_primary_gate_stdout),
                "sha256": after["primary_gate_stdout"],
            },
            "scheduler": primary_gate_scheduler,
        },
        "independent_gate": {
            "path": str(independent_path),
            "sha256": after["independent_gate"],
            "stdout": {
                "scheduler_path": str(independent_gate_stdout),
                "archive_path": str(archived_independent_gate_stdout),
                "sha256": after["independent_gate_stdout"],
            },
            "scheduler": independent_gate_scheduler,
        },
        "gates_byte_identical": True,
        "gates_distinct_paths_inodes_jobs": True,
        "gates_recomputed_exactly": True,
        "teacher_identity": identity,
        "checkpoint_tree_hash_independently_reproduced": True,
        "strong_teacher_provenance_validator_passed": True,
        "tracked_teacher_validator_passed": True,
        "merge_scheduler_terminal": merge_scheduler,
        "merge_submitted_after_both_gates_completed": True,
        "merge_stdout": {
            "scheduler_path": str(merge_stdout),
            "archive_path": str(archived_merge_stdout),
            "sha256": after["merge_stdout"],
        },
        "checkpoint_sealed_read_only": True,
        "all_bound_gate_and_stdout_artifacts_sealed_read_only": True,
        "stable_custody_revalidated_before_publication": True,
        "heldout_student_outcomes_inspected": False,
        "claim_boundary": "Teacher gate and merged-checkpoint custody only; no OPD student-performance result.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--primary-gate", required=True)
    parser.add_argument("--primary-gate-job-id", required=True)
    parser.add_argument("--primary-gate-stdout", required=True)
    parser.add_argument("--independent-gate", required=True)
    parser.add_argument("--independent-gate-job-id", required=True)
    parser.add_argument("--independent-gate-stdout", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--merge-job-id", required=True)
    parser.add_argument("--merge-stdout", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit(args)
    output = write_new(args.output, payload)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
