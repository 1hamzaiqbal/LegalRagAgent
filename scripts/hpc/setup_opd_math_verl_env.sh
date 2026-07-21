#!/usr/bin/env bash
# Build the isolated pinned-veRL execution environment on persistent EIT storage.
set -euo pipefail

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
VERL="${OPD_MATH_VERL_CHECKOUT:-/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/verl}"
ENV_DIR="${OPD_MATH_VERL_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_verl}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
UV_CACHE_DIR="${OPD_MATH_VERL_UV_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/uv_cache_opd_math_verl}"
PINNED_VERL_COMMIT="6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$VERL" rev-parse HEAD)" = "$PINNED_VERL_COMMIT"
test -z "$(git -C "$VERL" status --porcelain=v1 --untracked-files=no)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"
COMMIT_FREEZE="$FREEZE_ROOT/upstream_verl.freeze.txt"

if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
  echo "Refusing to alter existing pinned-veRL environment: $ENV_DIR" >&2
  exit 2
fi
if [[ -e "$COMMIT_FREEZE" || -L "$COMMIT_FREEZE" ]]; then
  echo "Refusing to replace commit-specific pinned-veRL freeze: $COMMIT_FREEZE" >&2
  exit 2
fi

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
test -n "$UV_BIN"
mkdir -p "$UV_CACHE_DIR" "$(dirname "$ENV_DIR")" "$FREEZE_ROOT"
export UV_CACHE_DIR

"$UV_BIN" venv "$ENV_DIR" --python 3.11
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  "vllm==0.12.0" \
  "math-verify[antlr4_13_2]==0.9.0" \
  "safetensors==0.7.0" \
  "$VERL"

"$ENV_DIR/bin/python" -c 'import importlib.metadata as m,re,sys; values={}; [(values.__setitem__(re.sub(r"[-_.]+","-",d.metadata["Name"]).lower(),d.version)) for d in m.distributions()]; sys.stdout.write("".join(f"{k}=={values[k]}\n" for k in sorted(values)))' \
  >"$ENV_DIR/requirements.freeze.txt"
cp "$ENV_DIR/requirements.freeze.txt" "$COMMIT_FREEZE"
chmod 0444 "$ENV_DIR/requirements.freeze.txt" "$COMMIT_FREEZE"

"$ENV_DIR/bin/python" "$REPO/scripts/opd_math/verify_environment.py" \
  --environment-root "$ENV_DIR" \
  --commit-freeze "$COMMIT_FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind upstream_verl

echo "Pinned-veRL environment created and frozen for LegalRagAgent commit $COMMIT"
