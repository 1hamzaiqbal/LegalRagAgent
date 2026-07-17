# Script map

Start from `docs/OPERATIONS.md`; filenames alone are not an execution contract.

## Current lanes

- `opd/`: teacher client, loss definitions, training skeleton, and CPU tests.
- `bandit/`: July allocation-policy baselines and analysis.
- `judge_pilot/`: reusable evidence-selection/judge builders and evaluators.
  Its ignored `data/` directory is local experiment state, not source.
- `hpc/`: Slurm launchers. Read the named completion audit and confirm paths,
  models, caches, and offline settings before submission.
- `local/`: machine-specific orchestration helpers; inspect defaults before
  use.

## Root scripts and `archive/`

Most root scripts reconstruct SCOPE/Snap-HyRE-era analyses. They remain for
lineage and selective reuse, but they are not a current run queue. A script is
launchable only when an active track page or completion plan names it and its
inputs still exist. `archive/` is historical and must never be used for new
jobs.

## Checks

```bash
uv run python scripts/check_workspace.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -q
uv run python scripts/opd/test_opd_loss.py
```
