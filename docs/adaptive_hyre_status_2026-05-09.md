# Adaptive HyRE Status (2026-05-09)

This is the current handoff for the legal-only adaptive Snap-HyDE/HyRE sweep.

## Current State

- Branch: `codex/final-report-snap-hyde`
- Latest pushed commit when refreshed: `d12df38`.
- Repo-local status when checked: clean
- Cluster worktree for this wave:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre`.
- The original cluster checkout
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent` was dirty, so the adaptive
  sweep was launched from the separate clean worktree.

The harness and cluster path are ready, but the goal is not complete until full
legal adaptive runs land and pass audit. Current postprocess coverage still
marks the adaptive modes as missing for full legal runs.

## Live Launch State

The first `cluster-vllm` wave was submitted as jobs `66812`-`66815` and failed
before evals. Gemma 4 26B did not fit on the A40 vLLM allocations under the
current settings; Housing also saw an uncorrectable ECC error during CUDA
startup. Treat jobs `66812`-`66815` as failed deployment evidence, not method
results.

The serial fallback wave used cluster retrieval/Chroma plus OpenRouter Gemma 4
26B (`or-gemma4-26b`). The first Housing fallback job (`66822`) was cancelled
after repeated CUDA ECC failures in retrieval embedding on `a40-2206`; the
replacement Housing job excluded that node. The serial API jobs were then
cancelled because one dataset job serialized all modes and was unlikely to
finish inside the 8h wallclock.

| Dataset | Job | Provider | State |
|---|---:|---|---|
| barexam | 66821 | `or-gemma4-26b` | cancelled; serial job too slow |
| housing | 66826 | `or-gemma4-26b` | cancelled; superseded by matrix |
| casehold | 66823 | `or-gemma4-26b` | cancelled; serial job too slow |
| legalbench_scalr | 66824 | `or-gemma4-26b` | cancelled; serial job too slow |

Manifest:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_submit_20260509_171016.tsv`.

Replacement Housing manifest:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_submit_20260509_171813.tsv`.

The active fast matrix uses one dataset/mode per SLURM job at `N=50`, provider
`or-gemma4-26b`, and `--exclude=a40-2206`.

Manifest:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_mode_matrix_20260509_172126.tsv`.

Job range: `66827`-`66849`.

Refresh at 2026-05-09 17:29 CT:

- Jobs `66827`-`66833` and `66835`-`66843` were running with normal provider
  preflight and early per-row output.
- Jobs `66844`-`66849` (LegalBench-SCALR) were pending on
  `QOSMaxJobsPerUserLimit`.
- Job `66834` (`housing` / `snap_hyre_state`) failed during Chroma
  `PersistentClient` startup with a disk I/O error while other Housing jobs were
  concurrently reading the large Housing collection.
- Replacement job `66851` was submitted for `housing` / `snap_hyre_state` only,
  with `--exclude=a40-2206` and
  `--dependency=afterany:66833:66835:66836:66837`.

Local runs remain possible for smoke tests, but they are not preferred for this
wave because the cluster has the complete legal Chroma collections.

Refresh at 2026-05-09 17:56 CT:

- Postprocess now supports `--tag-contains`; this wave should be summarized with
  `--tag-contains adaptive-hyre-or-gemma4-26b` so older same-provider logs do
  not pollute the active matrix.
- Current clean, tag-scoped N=50 parity signals:
  - `barexam`: `rag_simple` 86.0%, `adaptive_snap_hyre` 86.0%;
    `snap_hyre_option` has one parse failure and should not be treated as clean.
  - `housing`: `rag_state_filter` 62.0%, `adaptive_snap_hyre` 62.0%;
    `adaptive_snap_hyre_anchor` and `adaptive_snap_hyre_diverse` are clean at
    60.0%.
  - `casehold`: `rag_simple` 70.0%, `adaptive_snap_hyre_anchor` 70.0%;
    `adaptive_snap_hyre` is clean at 64.0%.
- Missing/active pieces:
  - `housing` / `snap_hyre_state`: replacement job `66851` is running.
  - `casehold` / `adaptive_snap_hyre_diverse`: original job `66843` hung after
    8/50 and was cancelled; replacement job `66854` is running.
  - `legalbench_scalr`: `rag_simple` completed at 72.0%; jobs `66845`-`66849`
    are still running for the remaining modes.

Refresh at 2026-05-09 18:48 CT:

- The N=50 tag-scoped adaptive HyRE matrix is complete and passes the strict
  readiness gate:

```bash
PROVIDER=or-gemma4-26b MIN_N=50 TAG_CONTAINS=adaptive-hyre-or-gemma4-26b \
  EVAL_VENV=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv \
  scripts/check_adaptive_hyre_readiness.sh
```

- Generated summaries copied into this repo:
  `docs/adaptive_hyre_sweep_latest.md` and
  `docs/adaptive_hyre_sweep_latest.json`.
- Harness fix landed after this run started: detail logs now include dataset/tag
  in the filename and store `tag` in each detail row. This was needed because
  concurrent same-mode jobs can otherwise collide on
  `eval_<mode>_<provider>_<minute>_detail.jsonl`.
- Clean N=50 frontier:
  - `barexam`: parity, `rag_simple` 86.0% vs `adaptive_snap_hyre` 86.0%.
  - `housing`: parity, `snap_hyre_state` 64.0% vs
    `adaptive_snap_hyre_diverse` 64.0%.
  - `casehold`: parity, `rag_simple` 70.0% vs
    `adaptive_snap_hyre_diverse` 70.0%.
  - `legalbench_scalr`: lead, `rag_snap_hyde_2call` 76.0% vs
    `adaptive_snap_hyre_anchor` 78.0%.
- Treat this as an N=50 iteration signal, not full-corpus evidence. The useful
  next step is to scale the strongest frontier policies to N=200 with the fixed
  detail-log naming.

N=200 targeted follow-up launched at 2026-05-09 18:52 CT with fixed detail-log
naming:

| Dataset | Mode | Job |
|---|---|---:|
| barexam | `rag_simple` | 66866 |
| barexam | `adaptive_snap_hyre` | 66867 |
| housing | `rag_state_filter` | 66868 |
| housing | `snap_hyre_state` | 66869 |
| housing | `adaptive_snap_hyre_diverse` | 66870 |
| casehold | `rag_simple` | 66871 |
| casehold | `adaptive_snap_hyre_diverse` | 66872 |
| legalbench_scalr | `rag_simple` | 66873 |
| legalbench_scalr | `rag_snap_hyde_2call` | 66874 |
| legalbench_scalr | `adaptive_snap_hyre_anchor` | 66875 |

Manifest:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_n200_targeted_20260509_185235.tsv`.

Targeted N=200 postprocessing should use `--expected-run` for the ten launched
dataset/mode pairs. Do not use the full adaptive readiness gate for this wave,
because it intentionally scales only the N=50 frontier policies and controls.

## Methods To Run

Option-style datasets (`barexam`, `casehold`, `legalbench_scalr`):

```bash
RUN_SPECS="rag_simple rag_snap_hyde_2call snap_hyre_option adaptive_snap_hyre adaptive_snap_hyre_anchor adaptive_snap_hyre_diverse"
```

HousingQA:

```bash
RUN_SPECS="rag_state_filter snap_hyre_state adaptive_snap_hyre adaptive_snap_hyre_anchor adaptive_snap_hyre_diverse"
```

The three adaptive policy probes are:

- `adaptive_snap_hyre`: task-shape bottleneck route, one HyRE retrieval query.
- `adaptive_snap_hyre_anchor`: same route, HyRE plus raw task anchor.
- `adaptive_snap_hyre_diverse`: same route, HyRE plus raw task anchor plus
  sanitized snap-reasoning anchor.
- `adaptive_snap_hyre_v2`: controller distilled from the N=50/N=200 evidence:
  Housing uses state-filtered diverse anchors, CaseHOLD uses option-grounded
  diverse anchors, SCALR uses plain two-call Snap-HyDE because option anchoring
  was unstable, and BarExam uses option-grounded HyRE.

All four are intended to stay at two LLM calls per question.

## Cluster Launch

From the cluster worktree:

```bash
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre
git fetch hamza codex/final-report-snap-hyde
git pull --ff-only hamza codex/final-report-snap-hyde
REPO=$PWD \
DATA_REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent \
EVAL_VENV=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv \
CHROMA_DB_DIR=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/chroma_db \
USE_VLLM=0 \
PROVIDER=or-gemma4-26b \
scripts/hpc/submit_adaptive_hyre_legal_sweep.sh
```

For a subset:

```bash
REPO=$PWD \
DATA_REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent \
EVAL_VENV=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv \
CHROMA_DB_DIR=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/chroma_db \
USE_VLLM=0 \
PROVIDER=or-gemma4-26b \
scripts/hpc/submit_adaptive_hyre_legal_sweep.sh housing casehold
```

For the current fast matrix shape:

```bash
REPO=$PWD \
DATA_REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent \
EVAL_VENV=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv \
CHROMA_DB_DIR=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/chroma_db \
N_QUESTIONS=50 \
USE_VLLM=0 \
PROVIDER=or-gemma4-26b \
SBATCH_EXTRA_ARGS="--exclude=a40-2206" \
scripts/hpc/submit_adaptive_hyre_mode_matrix.sh
```

The launch wrapper runs:

1. branch freshness checks
2. dirty-checkout guard
3. `scripts/hpc/prepare_adaptive_hyre_cluster.sh`
4. `scripts/hpc/submit_adaptive_hyre_legal_sweep.sh`

`prepare_adaptive_hyre_cluster.sh` checks mode registration, script syntax,
API-free adaptive smoke tests, and the four legal Chroma collections unless
`CHECK_CHROMA=0` is set.

## Monitoring And Evidence

```bash
PROVIDER=or-gemma4-26b \
EVAL_VENV=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv \
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs \
LOCAL_LOG_DIR=logs \
scripts/hpc/monitor_adaptive_hyre_sweep.sh
```

Successful dataset jobs write:

- `$LOG_DIR/adaptive_hyre_<provider>_<dataset>_n<N>_k<K>_<job>.md`
- `$LOG_DIR/adaptive_hyre_<provider>_<dataset>_n<N>_k<K>_<job>.json`

The JSON includes latest logs, adaptive coverage, and the adaptive parity
frontier (`LEADS`, `PARITY`, `GAP`, or `MISSING`).

Before treating a result as usable evidence:

```bash
python scripts/analyze_detail_flags.py logs/<detail_log>.jsonl
python scripts/audit_adaptive_hyre_logs.py logs/<detail_log>.jsonl
python scripts/postprocess_adaptive_hyre_sweep.py --min-n 20 \
  --dataset legalbench_scalr \
  --provider or-gemma4-26b \
  --output docs/adaptive_hyre_sweep_latest.md \
  --json-output docs/adaptive_hyre_sweep_latest.json
```

Use a strict readiness gate for automation:

```bash
python scripts/postprocess_adaptive_hyre_sweep.py --min-n 20 --dataset legalbench_scalr --provider or-gemma4-26b --require-ready
```

The readiness gate requires every expected adaptive mode to be present and to
pass `scripts/audit_adaptive_hyre_logs.py`.

For the targeted N=200 wave, use targeted readiness instead:

```bash
python scripts/postprocess_adaptive_hyre_sweep.py --min-n 200 \
  --provider or-gemma4-26b \
  --tag-contains adaptive-hyre-or-gemma4-26b \
  --expected-run barexam:rag_simple \
  --expected-run barexam:adaptive_snap_hyre \
  --expected-run housing:rag_state_filter \
  --expected-run housing:snap_hyre_state \
  --expected-run housing:adaptive_snap_hyre_diverse \
  --expected-run casehold:rag_simple \
  --expected-run casehold:adaptive_snap_hyre_diverse \
  --expected-run legalbench_scalr:rag_simple \
  --expected-run legalbench_scalr:rag_snap_hyde_2call \
  --expected-run legalbench_scalr:adaptive_snap_hyre_anchor \
  --require-ready
```

To run the strict gate over all four legal datasets:

```bash
PROVIDER=or-gemma4-26b MIN_N=50 scripts/check_adaptive_hyre_readiness.sh
```

## Completion Checklist

Do not treat the adaptive HyRE iteration goal as complete until all of these are
true:

1. Cluster launch produced submit manifests under `$LOG_DIR` for the intended
   legal datasets.
2. Each dataset has a persisted markdown and JSON postprocess summary scoped to
   that dataset and the active provider (`or-gemma4-26b` for the current
   fallback wave).
3. This readiness command exits zero for every legal dataset:

```bash
scripts/check_adaptive_hyre_readiness.sh
```

4. The JSON parity frontier has a non-`MISSING` adaptive policy for each
   dataset.
5. Any reported `GAP` case has a concrete interpretation: retrieval miss,
   option conversion, metadata filtering, answer parsing, or model/cost issue.
6. Smoke logs, local-only checks, and incomplete partial-job summaries remain
   excluded from result claims.

Do not promote smoke logs or local-only checks as result claims.
