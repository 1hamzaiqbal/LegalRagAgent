#!/usr/bin/env bash
# Create the isolated TRL training environment. Run once on EIT with network.
set -euo pipefail

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
UV_CACHE_DIR="${OPD_MATH_UV_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/uv_cache_opd_math}"

test -f "$REPO/requirements/opd-math.txt"
if [[ -e "$ENV_DIR" && "${OPD_MATH_ALLOW_ENV_UPDATE:-0}" != "1" ]]; then
  echo "Refusing to alter existing environment: $ENV_DIR" >&2
  echo "Set OPD_MATH_ALLOW_ENV_UPDATE=1 only after inspecting it." >&2
  exit 2
fi

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
test -n "$UV_BIN"
mkdir -p "$UV_CACHE_DIR" "$(dirname "$ENV_DIR")"
export UV_CACHE_DIR

if [[ ! -e "$ENV_DIR" ]]; then
  "$UV_BIN" venv "$ENV_DIR" --python 3.11
fi
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" -r "$REPO/requirements/opd-math.txt"
"$UV_BIN" pip freeze --python "$ENV_DIR/bin/python" | sort >"$ENV_DIR/requirements.freeze.txt"
echo "Environment installed. GPU imports are intentionally deferred to the Slurm preflight."
