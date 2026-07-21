#!/usr/bin/env bash
# Bind an unchanged, exactly verified pinned-veRL environment to the current commit.
set -euo pipefail

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
VERL="${OPD_MATH_VERL_CHECKOUT:-/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/verl}"
ENV_DIR="${OPD_MATH_VERL_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_verl}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$VERL" rev-parse HEAD)" = "6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"
test -z "$(git -C "$VERL" status --porcelain=v1 --untracked-files=no)"
test -x "$ENV_DIR/bin/python"
test -f "$ENV_DIR/requirements.freeze.txt"
test ! -L "$ENV_DIR/requirements.freeze.txt"
test "$(stat -c %a "$ENV_DIR/requirements.freeze.txt")" = 444

COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"
COMMIT_FREEZE="$FREEZE_ROOT/upstream_verl.freeze.txt"
if [[ -e "$COMMIT_FREEZE" || -L "$COMMIT_FREEZE" ]]; then
  echo "Refusing to replace commit-specific pinned-veRL freeze: $COMMIT_FREEZE" >&2
  exit 2
fi
mkdir -p "$FREEZE_ROOT"
cp "$ENV_DIR/requirements.freeze.txt" "$COMMIT_FREEZE"
chmod 0444 "$COMMIT_FREEZE"

"$ENV_DIR/bin/python" "$REPO/scripts/opd_math/verify_environment.py" \
  --environment-root "$ENV_DIR" \
  --commit-freeze "$COMMIT_FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind upstream_verl

echo "Existing pinned-veRL environment verified and bound to commit $COMMIT"
