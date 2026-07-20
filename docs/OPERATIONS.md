# LegalRagAgent operations guide

This is the practical runbook for choosing a surface, placing new artifacts,
checking a change, and recovering historical material. Research claims belong
in the snapshot, track pages, and signoff log; this file governs mechanics.

## Choose one active lane

| Work | Mac path | EIT path | Branch |
|---|---|---|---|
| Three-dial science, evaluation, and marginal-utility control | `/Users/hamzaiqbal/grad/LegalRagAgent` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-three-dial` | `codex/three_dial` |
| OPD/distillation and teacher-student engineering | `/Users/hamzaiqbal/grad/LegalRagAgent-opd-distillation` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-distillation` | `codex/opd_distillation` |
| Bounded OPD math source-transfer experiment | `/Users/hamzaiqbal/grad/LegalRagAgent-opd-math` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math` | `codex/opd_math_pipeline` |

Do not launch new work from `codex/scope_old`, an archive branch, or a path
under `/engrfs/project/jacobsn/hiqbal/archives/`. Historical checkouts are
restored only for inspection and removed afterward.

## Fast orientation

```bash
git status --short --branch
uv run python scripts/check_workspace.py
sed -n '1,220p' ACTIVE_TRACK.md
sed -n '1,240p' wiki/snapshots/research-state-2026-07-17.md
```

The workspace checker validates required entrypoints, forbidden predecessor
directories, current Markdown links, Obsidian wikilinks, generated-artifact
policy, branch-specific files, and stale local recovery pointers. Add
`--strict-clean` in CI or before handoff when a dirty tree must fail.

## Artifact placement contract

| Artifact | Location | Git policy |
|---|---|---|
| Reusable code, configs, and tests | `eval/`, `scripts/`, `utils/`, `tests/` | Track |
| Current compact result/citation evidence | `evidence/<campaign>/` | Track with manifest |
| Experiment ledger and necessary row logs | `logs/` | Track only when named by a signoff/audit gate |
| Compact generated summaries, tables, and plots | `docs/generated/` | Track selectively |
| Large point-level/generated JSONL | EIT `artifacts/legalrag/` | Do not add to active Git |
| Downloaded datasets, caches, vector stores, checkpoints | EIT scratch/project data or ignored local paths | Do not add to active Git |
| Papers and related repositories | EIT `literature/legalrag/` plus wiki manifests | PDFs/repos stay on EIT |
| Superseded project trees and recovery bundles | EIT `archives/LegalRagAgent_archive/` plus named Git archive refs | Never mix into active worktrees |

The pre-trim May/July generated directory is checksummed at:

`/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/2026-07-17/docs-generated-pre-trim/`

## Change and evidence workflow

1. Confirm the active branch and read its `ACTIVE_TRACK.md`.
2. Name the question, reader, evidence action/set, budget, and outcome schema.
3. Reuse or extend a source log; do not promote a narrative file to evidence.
4. Put compact citable outputs under `evidence/` or `docs/generated/` and add a
   manifest/signoff pointer.
5. Run focused tests, the workspace checker, and `git diff --check`.
6. Commit on exactly one active lane. Port shared documentation/interfaces by
   cherry-pick; do not merge an entire lane into the other.
7. Push to GitHub, then fast-forward the matching EIT checkout.

## Validation ladder

```bash
# Always
uv run python scripts/check_workspace.py
git diff --check

# Evaluation changes
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -q

# OPD changes
uv run python scripts/opd/test_opd_loss.py

# OPD math changes
uv run pytest -q \
  tests/test_opd_math_data.py \
  tests/test_opd_finalize_semantic_reviews.py \
  tests/test_opd_reward_loss.py \
  tests/test_teacher_client_token_ids.py \
  tests/test_opd_tokenizer_contract.py \
  tests/test_opd_evaluation_shards.py \
  tests/test_opd_hpc_wrappers.py \
  tests/test_opd_quality_gates.py \
  tests/test_opd_merge_custody.py \
  tests/test_opd_teacher_recipe.py \
  tests/test_opd_run_contract.py \
  tests/test_opd_environment_custody.py \
  tests/test_opd_evaluation_timing_plan.py \
  tests/test_opd_student_results.py \
  tests/test_opd_server_process_binding.py
```

GPU or API smoke completion proves plumbing only. Scientific promotion still
requires the task-specific gate recorded in `docs/signoff_log.md`.

For the OPD-math campaign, legacy v1 evaluations cannot authorize new science.
The M teacher failed its gap gate and is closed: never launch `M_M` or `M_O`.
The active recovery requires fresh M/O raw-student support, one strict O
teacher/gap, `baseline_M`, `O_M`, `baseline_O`, `O_O`, and the tracked
four-arm conditional readout, all under exact-environment freezes and one final
Git commit. The O full-gap plan requires paired base/trained timing evidence
and at least five shared shards; every primary student arm requires a
preregistered stable run ID. The final preregistration pins the selected O
teacher, both support gates, all four paths, and the checksummed readout bundle
before any student arm is launched. Any later tracked commit reopens that
campaign boundary. See
`wiki/snapshots/opd-math-verifier-recovery-2026-07-20.md`.

Launch hold: the four-arm successor above was never sealed or launched. The
later objective-family request is recorded in
`wiki/snapshots/opd-objective-family-expansion-2026-07-20.md` and requires a
new immutable campaign namespace. The strict verifier, support, O-teacher, and
custody machinery are reusable prerequisites; the four-arm preregistration is
not. Do not launch a 100-step expanded student arm until the declarative
objective registry, cross-veRL fidelity tests, three-seed preregistration, and
generalized outcome-blind readout pass on one final commit.

## Recovery

- Git history: `codex/scope_old`, `codex/archive/pre_cleanup_20260717`, and
  `codex/archive/early_agentic_20260717`.
- Cleanup ZIPs:
  `/engrfs/project/jacobsn/hiqbal/archives/LegalRagAgent_archive/2026-07-17/local-cleanup/archives/`
- Full recovery ZIP:
  `/engrfs/project/jacobsn/hiqbal/archives/LegalRagAgent_archive/2026-07-17/local-cleanup/recovery-package/LegalRagAgent-recovery-20260717.zip`

Restore into a new empty directory. Never unpack or apply a historical patch
over either active worktree.
