#!/bin/bash
# Pull, preflight, and submit the adaptive HyRE legal sweep from the cluster.

set -euo pipefail

EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/final-report-snap-hyde}
AUTO_PULL=${AUTO_PULL:-0}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}

branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected branch $EXPECTED_BRANCH, found $branch" >&2
  echo "Run: git fetch origin && git switch $EXPECTED_BRANCH && git pull --ff-only" >&2
  exit 2
fi

if [[ "$ALLOW_DIRTY" != "1" && -n "$(git status --short)" ]]; then
  echo "ERROR: checkout is dirty before launch" >&2
  git status --short >&2
  exit 2
fi

git fetch origin "$EXPECTED_BRANCH"
local_head=$(git rev-parse HEAD)
remote_head=$(git rev-parse "origin/$EXPECTED_BRANCH")
if [[ "$local_head" != "$remote_head" ]]; then
  if [[ "$AUTO_PULL" == "1" ]]; then
    git pull --ff-only origin "$EXPECTED_BRANCH"
  else
    echo "ERROR: local HEAD differs from origin/$EXPECTED_BRANCH" >&2
    echo "local=$local_head" >&2
    echo "remote=$remote_head" >&2
    echo "Set AUTO_PULL=1 to fast-forward automatically." >&2
    exit 2
  fi
fi

scripts/hpc/prepare_adaptive_hyre_cluster.sh "$@"
scripts/hpc/submit_adaptive_hyre_legal_sweep.sh "$@"
