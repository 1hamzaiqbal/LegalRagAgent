#!/usr/bin/env python3
"""Fail-closed live-environment custody check for OPD-math jobs.

Run this script with the Python interpreter from the environment being checked.
It intentionally uses only the Python standard library so that a damaged or
incomplete scientific environment cannot hide behind a second verifier env.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path


SCHEMA = "opd_math_environment_verification_v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FREEZE_LINE_RE = re.compile(
    r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==(?P<version>[^\s]+)"
)


def canonical_distribution_name(name: str) -> str:
    """Apply the PEP 503 normalization used for distribution identity."""

    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if not normalized:
        raise ValueError("distribution name is empty after normalization")
    return normalized


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(
    path: str | Path, *, label: str, require_executable: bool = False
) -> tuple[Path, bytes]:
    """Read one non-symlink regular file without following a leaf symlink."""

    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {candidate}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
    if require_executable and before.st_mode & 0o111 == 0:
        raise ValueError(f"{label} is not executable: {candidate}")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ValueError(
            f"could not open {label} without following symlinks: {candidate}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} ceased to be a regular file: {candidate}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while it was being opened: {candidate}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if (
        (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        raise ValueError(f"{label} changed while it was being read: {candidate}")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise ValueError(f"{label} size changed while it was being read: {candidate}")
    return candidate.resolve(strict=True), payload


def parse_exact_freeze(payload: bytes, *, label: str) -> dict[str, str]:
    """Parse a freeze containing only one exact ``name==version`` per line."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc

    versions: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = FREEZE_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"{label} line {line_number} is not an exact name==version requirement"
            )
        name = canonical_distribution_name(match.group("name"))
        if name in versions:
            raise ValueError(f"{label} contains duplicate distribution {name!r}")
        versions[name] = match.group("version")
    return versions


def installed_distribution_versions() -> dict[str, str]:
    """Return the complete normalized distribution map visible to this Python."""

    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        raw_version = distribution.version
        if not raw_name or not raw_version:
            raise ValueError("an installed distribution lacks exact Name/Version metadata")
        name = canonical_distribution_name(raw_name)
        if name in versions:
            raise ValueError(
                f"multiple installed distributions normalize to the same name: {name!r}"
            )
        versions[name] = raw_version
    return versions


def _assert_exact_distribution_map(
    *, expected: dict[str, str], actual: dict[str, str]
) -> None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    version_drift = {
        name: {"freeze": expected[name], "installed": actual[name]}
        for name in sorted(set(expected) & set(actual))
        if expected[name] != actual[name]
    }
    if missing or extra or version_drift:
        raise ValueError(
            "installed distribution map differs from requirements.freeze.txt: "
            + json.dumps(
                {
                    "missing": missing,
                    "extra": extra,
                    "version_drift": version_drift,
                },
                sort_keys=True,
            )
        )


def _validate_expected_executable(
    executable: str | Path, *, environment_root: Path
) -> dict[str, str]:
    expected = Path(executable)
    expected_parent = (environment_root / "bin").resolve(strict=True)
    if expected.parent.resolve(strict=True) != expected_parent:
        raise ValueError(
            f"expected executable must be directly under {expected_parent}: {expected}"
        )
    executable_path, payload = _read_regular_file(
        expected, label="expected environment executable", require_executable=True
    )
    first_line = payload.splitlines()[0] if payload.splitlines() else b""
    try:
        shebang = first_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("expected environment executable has a non-UTF-8 shebang") from exc
    environment_python = environment_root / "bin" / "python"
    required_shebang = f"#!{environment_python}"
    if shebang != required_shebang:
        raise ValueError(
            "expected environment executable shebang does not invoke the selected "
            f"environment bin/python: expected={required_shebang!r}, actual={shebang!r}"
        )
    return {
        "path": str(executable_path),
        "sha256": sha256_bytes(payload),
        "shebang": shebang,
    }


def verify_environment(
    *,
    environment_root: str | Path,
    commit_freeze: str | Path,
    expected_commit: str,
    freeze_kind: str,
    expected_executable: str | Path | None = None,
) -> dict:
    """Validate live prefix, exact packages, freeze custody, and console script."""

    if not isinstance(expected_commit, str) or COMMIT_RE.fullmatch(expected_commit) is None:
        raise ValueError("expected commit must be exactly 40 lowercase hexadecimal characters")
    if not isinstance(freeze_kind, str) or freeze_kind not in {
        "train",
        "serve",
        "upstream_verl",
    }:
        raise ValueError("freeze kind must be train, serve, or upstream_verl")

    try:
        declared_root = Path(environment_root).resolve(strict=True)
        live_root = Path(sys.prefix).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ValueError(f"could not resolve declared/live environment prefix: {exc}") from exc
    if declared_root != live_root:
        raise ValueError(
            f"environment root is not the live sys.prefix: declared={declared_root}, "
            f"live={live_root}"
        )

    required_serve_executable = declared_root / "bin" / "vllm"
    if freeze_kind == "serve":
        if expected_executable is None:
            raise ValueError("serve verification requires the selected environment bin/vllm")
        if Path(expected_executable).absolute() != required_serve_executable:
            raise ValueError(
                "serve verification expected executable must be the selected "
                f"environment bin/vllm: {required_serve_executable}"
            )
    elif expected_executable is not None:
        raise ValueError(
            f"{freeze_kind} verification does not accept a serve executable"
        )

    environment_python = declared_root / "bin" / "python"
    live_executable = Path(sys.executable).absolute()
    expected_python = environment_python.absolute()
    if live_executable != expected_python:
        raise ValueError(
            "verifier is not running through the selected environment bin/python: "
            f"live={live_executable}, expected={expected_python}"
        )

    requirements_path, requirements_payload = _read_regular_file(
        declared_root / "requirements.freeze.txt",
        label="environment requirements.freeze.txt",
    )
    frozen_versions = parse_exact_freeze(
        requirements_payload, label="environment requirements.freeze.txt"
    )
    installed_versions = installed_distribution_versions()
    _assert_exact_distribution_map(expected=frozen_versions, actual=installed_versions)

    commit_path = Path(commit_freeze)
    required_name = f"{freeze_kind}.freeze.txt"
    try:
        resolved_parent = commit_path.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"could not resolve commit-specific freeze parent: {exc}") from exc
    if (
        commit_path.name != required_name
        or resolved_parent.name != expected_commit
        or resolved_parent.parent.name != "environment_freezes"
    ):
        raise ValueError(
            "commit-specific freeze must be named "
            f"environment_freezes/{expected_commit}/{required_name}"
        )
    commit_path, commit_payload = _read_regular_file(
        commit_path, label="commit-specific environment freeze"
    )
    if commit_payload != requirements_payload:
        raise ValueError(
            "commit-specific environment freeze is not byte-identical to the live "
            "environment requirements.freeze.txt"
        )

    package_map_payload = json.dumps(
        installed_versions, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    result = {
        "schema_version": 1,
        "schema": SCHEMA,
        "status": "passed",
        "environment_root": str(declared_root),
        "live_python": str(Path(sys.executable).absolute()),
        "expected_commit": expected_commit,
        "freeze_kind": freeze_kind,
        "installed_distribution_count": len(installed_versions),
        "installed_distribution_map_sha256": sha256_bytes(package_map_payload),
        "requirements_freeze": {
            "path": str(requirements_path),
            "sha256": sha256_bytes(requirements_payload),
        },
        "commit_freeze": {
            "path": str(commit_path),
            "sha256": sha256_bytes(commit_payload),
            "byte_identical_to_requirements_freeze": True,
        },
        "expected_executable": None,
    }
    if expected_executable is not None:
        result["expected_executable"] = _validate_expected_executable(
            expected_executable, environment_root=declared_root
        )
    return result


def run_external_environment_verification(
    *,
    environment_root: str | Path,
    commit_freeze: str | Path,
    expected_commit: str,
    freeze_kind: str,
    expected_executable: str | Path | None = None,
) -> dict:
    """Run this verifier with the selected environment's own Python."""

    if not isinstance(expected_commit, str) or COMMIT_RE.fullmatch(expected_commit) is None:
        raise ValueError("expected commit must be exactly 40 lowercase hexadecimal characters")
    if freeze_kind not in {"train", "serve", "upstream_verl"}:
        raise ValueError("freeze kind must be train, serve, or upstream_verl")
    try:
        root = Path(environment_root).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ValueError(f"could not resolve selected environment root: {exc}") from exc
    command = [
        str(root / "bin" / "python"),
        str(Path(__file__).resolve()),
        "--environment-root",
        str(root),
        "--commit-freeze",
        str(commit_freeze),
        "--expected-commit",
        expected_commit,
        "--freeze-kind",
        freeze_kind,
    ]
    if expected_executable is not None:
        command.extend(("--expected-executable", str(expected_executable)))
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"selected environment verifier could not run: {exc}") from exc
    if process.returncode != 0:
        detail = process.stderr.strip().splitlines()
        suffix = detail[-1] if detail else "no error detail"
        raise ValueError(
            f"selected environment verifier failed with exit {process.returncode}: {suffix}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("selected environment verifier did not emit exactly one JSON record")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("selected environment verifier emitted invalid JSON") from exc
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != 1
        or result.get("schema") != SCHEMA
        or result.get("status") != "passed"
    ):
        raise ValueError("selected environment verifier emitted an invalid pass record")
    return result


def reverify_recorded_environment(recorded: dict, *, in_process: bool = False) -> dict:
    """Rerun and require byte-for-byte-equivalent structured verification."""

    if not isinstance(recorded, dict) or recorded.get("schema") != SCHEMA:
        raise ValueError("recorded environment verification has an invalid schema")
    commit_freeze = recorded.get("commit_freeze")
    executable = recorded.get("expected_executable")
    if not isinstance(commit_freeze, dict):
        raise ValueError("recorded environment verification lacks a commit freeze")
    if executable is not None and not isinstance(executable, dict):
        raise ValueError("recorded environment verification has an invalid executable")
    kwargs = {
        "environment_root": recorded.get("environment_root"),
        "commit_freeze": commit_freeze.get("path"),
        "expected_commit": recorded.get("expected_commit"),
        "freeze_kind": recorded.get("freeze_kind"),
        "expected_executable": None if executable is None else executable.get("path"),
    }
    fresh = (
        verify_environment(**kwargs)
        if in_process
        else run_external_environment_verification(**kwargs)
    )
    if fresh != recorded:
        raise ValueError("live environment verification differs from its initial identity")
    return fresh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-root", required=True)
    parser.add_argument("--commit-freeze", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--freeze-kind",
        required=True,
        choices=("train", "serve", "upstream_verl"),
    )
    parser.add_argument("--expected-executable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_environment(
        environment_root=args.environment_root,
        commit_freeze=args.commit_freeze,
        expected_commit=args.expected_commit,
        freeze_kind=args.freeze_kind,
        expected_executable=args.expected_executable,
    )
    print(json.dumps(result, separators=(",", ":"), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
