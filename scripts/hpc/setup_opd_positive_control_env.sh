#!/usr/bin/env bash
# Build the exact isolated environment declared by pinned upstream OPSD.
set -euo pipefail

REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
UV_CACHE_DIR="${OPD_POSITIVE_UV_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/uv_cache_opd_positive_control}"
UV_BIN="/home/compute/hiqbal/.local/bin/uv"
PYTHON_BIN="/home/compute/hiqbal/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10"

test -z "$(git -C "$REPO" status --porcelain=v1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"
COMMIT_FREEZE="$FREEZE_ROOT/upstream_opsd.freeze.txt"

if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
  echo "Refusing to alter existing positive-control environment: $ENV_DIR" >&2
  exit 2
fi
if [[ -e "$COMMIT_FREEZE" || -L "$COMMIT_FREEZE" ]]; then
  echo "Refusing to replace commit-specific environment freeze: $COMMIT_FREEZE" >&2
  exit 2
fi
test -x "$UV_BIN"
test -x "$PYTHON_BIN"
mkdir -p "$(dirname "$ENV_DIR")" "$FREEZE_ROOT" "$UV_CACHE_DIR"
export UV_CACHE_DIR

"$UV_BIN" venv "$ENV_DIR" --python "$PYTHON_BIN"
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  "torch==2.8.0" \
  "accelerate==1.11.0" \
  "transformers==4.57.1" \
  "trl==0.26.0" \
  "datasets==3.6.0" \
  "deepspeed==0.18.2" \
  "peft==0.17.1" \
  "bitsandbytes==0.48.2" \
  "wandb==0.22.3" \
  "vllm==0.11.0" \
  "xformers==0.0.32.post1" \
  "triton==3.4.0" \
  "einops==0.8.1" \
  "safetensors==0.5.3" \
  "sentencepiece==0.1.99" \
  "tiktoken==0.9.0" \
  "math-verify==0.8.0" \
  "ninja" "packaging" "wheel" "setuptools"
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  --no-build-isolation "flash-attn==2.8.3"

"$ENV_DIR/bin/python" -c 'import importlib.metadata as m,re,sys; values={}; [(values.__setitem__(re.sub(r"[-_.]+","-",d.metadata["Name"]).lower(),d.version)) for d in m.distributions() if d.metadata.get("Name")]; sys.stdout.write("".join(f"{k}=={values[k]}\n" for k in sorted(values)))' \
  >"$ENV_DIR/requirements.freeze.txt"
cp "$ENV_DIR/requirements.freeze.txt" "$COMMIT_FREEZE"
chmod 0444 "$ENV_DIR/requirements.freeze.txt" "$COMMIT_FREEZE"

"$ENV_DIR/bin/python" "$REPO/scripts/opd/verify_positive_control_environment.py" \
  --environment-root "$ENV_DIR" \
  --freeze "$COMMIT_FREEZE"

echo "Pinned upstream OPSD environment created for LegalRagAgent commit $COMMIT"
