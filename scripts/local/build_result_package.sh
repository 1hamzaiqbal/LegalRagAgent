#!/usr/bin/env bash
# Build source-gated Snap-HyRE package status tables and optional plots.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

UV="${UV:-uv}"
TAG_PREFIX="${TAG_PREFIX:-local-snap-hyre}"
OUT_DIR="${OUT_DIR:-docs/generated/snap_hyre_package}"

"$UV" run python scripts/build_snap_hyre_package.py \
  --tag-prefix "$TAG_PREFIX" \
  --out-dir "$OUT_DIR"

