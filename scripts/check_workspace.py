#!/usr/bin/env python3
"""Fast, dependency-free integrity checks for an active LegalRagAgent tree."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_BRANCHES = {"codex/three_dial", "codex/opd_distillation", "codex/opd_math_pipeline"}
REQUIRED = (
    "ACTIVE_TRACK.md",
    "CLAUDE.md",
    "docs/README.md",
    "docs/OPERATIONS.md",
    "docs/signoff_log.md",
    "wiki/index.md",
    "wiki/snapshots/research-state-2026-07-17.md",
    "wiki/tracks/three-dial.md",
    "wiki/tracks/opd-distillation.md",
    "wiki/literature/index.md",
)
FORBIDDEN_TOP_LEVEL = ("analysis", "ideas", "literature", "paper", "reports")
ENTRYPOINTS = (
    "README.md",
    "CLAUDE.md",
    "ACTIVE_TRACK.md",
    "docs/README.md",
    "docs/OPERATIONS.md",
    "docs/worktree_map_2026-07-17.md",
    "docs/archive_manifest_2026-07-17.md",
    "wiki/index.md",
    "wiki/START_HERE.md",
    "wiki/snapshots/research-state-2026-07-17.md",
    "wiki/tracks/three-dial.md",
    "wiki/tracks/opd-distillation.md",
    "wiki/tracks/scope-old.md",
)
STALE_POINTERS = (
    "/Users/hamzaiqbal/grad/LegalRagAgent_recovery_20260717",
    "/Users/hamzaiqbal/grad/LegalRagAgent_archive/LegalRagAgent-scope-old-20260717.zip",
    "/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre",
    "/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-scope-old",
)
OPD_MATH_REQUIRED = (
    "wiki/tracks/opd-math-source-transfer.md",
    "wiki/snapshots/opd-program-goal-2026-07-20.md",
    "configs/opd_math/source_manifest.json",
    "configs/opd_math/objective_registry.json",
    "configs/opd_math/objective_family_student_plan.json",
    "configs/opd_math/fidelity_plan.json",
    "evidence/july_2026/opd_finite_state_108548.json",
    "configs/opd_math/deepmath_qualification_plan.json",
    "configs/opd_math/deepmath_inventory_plan.json",
    "configs/opd_math/deepmath_collision_audit_plan.json",
    "configs/opd_math/teacher_training_plan.json",
    "requirements/opd-math.txt",
    "requirements/opd-math-serve.txt",
    "scripts/opd_math/README.md",
    "scripts/opd_math/prepare_data.py",
    "scripts/opd_math/train_teacher_grpo.py",
    "scripts/opd_math/evaluate_math.py",
    "scripts/opd_math/quality_gates.py",
    "scripts/opd_math/tokenizer_contract.py",
    "scripts/opd_math/server_scoring_probe.py",
    "scripts/opd_math/deepmath_qualification.py",
    "scripts/opd_math/materialize_deepmath_inventory.py",
    "scripts/opd_math/audit_deepmath_inventory.py",
    "scripts/opd_math/finalize_deepmath_audit.py",
    "scripts/opd/objective_registry.py",
    "scripts/opd/fidelity_plan.py",
    "scripts/opd/verify_finite_state.py",
    "scripts/hpc/setup_opd_math_env.sh",
    "scripts/hpc/setup_opd_math_serve_env.sh",
    "scripts/hpc/slurm_opd_math_cache_models.sh",
    "scripts/hpc/slurm_opd_math_deepmath_download.sh",
    "scripts/hpc/slurm_opd_math_deepmath_inventory.sh",
    "scripts/hpc/slurm_opd_math_deepmath_audit.sh",
    "scripts/hpc/slurm_opd_math_deepmath_finalize.sh",
    "scripts/hpc/slurm_opd_math_env_preflight.sh",
    "scripts/hpc/slurm_opd_math_serve_preflight.sh",
)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def check_required(errors: list[str]) -> None:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")
    for relative in FORBIDDEN_TOP_LEVEL:
        if (ROOT / relative).exists():
            errors.append(f"predecessor directory returned at repo root: {relative}/")


def check_entrypoint_links(errors: list[str]) -> None:
    for relative in ENTRYPOINTS:
        source = ROOT / relative
        if not source.is_file():
            continue
        for raw_target in MARKDOWN_LINK_RE.findall(source.read_text(errors="replace")):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"broken Markdown link: {relative} -> {raw_target}")


def wiki_index() -> dict[str, set[Path]]:
    index: dict[str, set[Path]] = defaultdict(set)
    wiki_root = ROOT / "wiki"
    for item in wiki_root.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(wiki_root)
        without_suffix = relative.with_suffix("").as_posix()
        index[without_suffix].add(item)
        index[item.stem].add(item)
        index[relative.as_posix()].add(item)
    return index


def check_wikilinks(errors: list[str], warnings: list[str]) -> None:
    index = wiki_index()
    missing: set[tuple[str, str]] = set()
    ambiguous: set[str] = set()
    for source in (ROOT / "wiki").rglob("*.md"):
        relative = source.relative_to(ROOT).as_posix()
        for raw in WIKILINK_RE.findall(source.read_text(errors="replace")):
            target = raw.split("|", 1)[0].split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://")):
                continue
            normalized = target.removeprefix("wiki/").removesuffix(".md")
            matches = index.get(normalized) or index.get(Path(normalized).name)
            if not matches:
                missing.add((relative, target))
            elif len(matches) > 1 and "/" not in normalized:
                ambiguous.add(target)
    for source, target in sorted(missing):
        errors.append(f"missing wikilink: {source} -> [[{target}]]")
    if ambiguous:
        warnings.append(
            "ambiguous basename wikilinks: " + ", ".join(sorted(ambiguous)[:12])
        )


def check_generated_policy(errors: list[str]) -> None:
    generated = ROOT / "docs/generated"
    allowed = re.compile(r"leakage_audit_.+_2026-07-02_points\.jsonl$")
    for item in generated.rglob("*.jsonl"):
        if not allowed.fullmatch(item.name):
            errors.append(
                "large generated JSONL belongs in EIT artifact storage: "
                + item.relative_to(ROOT).as_posix()
            )


def check_branch(branch: str, errors: list[str], warnings: list[str]) -> None:
    if branch not in ACTIVE_BRANCHES:
        warnings.append(f"historical/non-active branch checked out: {branch or '(detached)'}")
        return
    active_text = (ROOT / "ACTIVE_TRACK.md").read_text(errors="replace")
    if branch == "codex/three_dial" and "three-dial" not in active_text.lower():
        errors.append("ACTIVE_TRACK.md does not describe the three-dial lane")
    if branch in ("codex/opd_distillation", "codex/opd_math_pipeline"):
        if "opd" not in active_text.lower():
            errors.append("ACTIVE_TRACK.md does not describe the OPD lane")
        if not (ROOT / "scripts/hpc/slurm_opd_gated_smoke.sh").is_file():
            errors.append("OPD branch is missing its gated smoke launcher")
    if branch == "codex/opd_math_pipeline":
        for relative in OPD_MATH_REQUIRED:
            if not (ROOT / relative).is_file():
                errors.append(f"OPD math branch is missing required surface: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-clean", action="store_true", help="fail when Git has local changes"
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    branch = git("branch", "--show-current")

    check_required(errors)
    check_entrypoint_links(errors)
    check_wikilinks(errors, warnings)
    check_generated_policy(errors)
    check_branch(branch, errors, warnings)

    for relative in ENTRYPOINTS:
        source = ROOT / relative
        if not source.is_file():
            continue
        text = source.read_text(errors="replace")
        for pointer in STALE_POINTERS:
            if pointer in text:
                errors.append(f"stale local recovery pointer in {relative}: {pointer}")

    status = git("status", "--porcelain")
    if status:
        message = "working tree has local changes"
        if args.strict_clean:
            errors.append(message)
        else:
            warnings.append(message)

    print(f"branch: {branch or '(detached)'}")
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"FAIL: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"PASS: workspace contract holds ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
