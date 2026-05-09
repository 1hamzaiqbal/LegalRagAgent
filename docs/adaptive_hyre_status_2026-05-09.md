# Adaptive HyRE Status (2026-05-09)

This is the current handoff for the legal-only adaptive Snap-HyDE/HyRE sweep.

## Current State

- Branch: `codex/final-report-snap-hyde`
- Latest pushed commit at this handoff: `0a0a0d1`
- Repo-local status when checked: clean
- Direct cluster submission from this local environment is blocked:
  `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`.

The harness and cluster path are ready, but the goal is not complete until full
legal adaptive runs land and pass audit. Current postprocess coverage still
marks the adaptive modes as missing for full legal runs.

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

All three are intended to stay at two LLM calls per question.

## Cluster Launch

From the cluster checkout:

```bash
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
git fetch origin
git switch codex/final-report-snap-hyde
git pull --ff-only
AUTO_PULL=1 scripts/hpc/launch_adaptive_hyre_sweep.sh
```

For a subset:

```bash
AUTO_PULL=1 scripts/hpc/launch_adaptive_hyre_sweep.sh housing casehold
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
  --output docs/adaptive_hyre_sweep_latest.md \
  --json-output docs/adaptive_hyre_sweep_latest.json
```

Do not promote smoke logs or local-only checks as result claims.
