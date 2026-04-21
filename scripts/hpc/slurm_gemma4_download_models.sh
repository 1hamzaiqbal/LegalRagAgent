#!/bin/bash
#SBATCH -p general
#SBATCH -A engr-lab-jacobsn
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -J gemma4-download
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Download Gemma 4 sizes not yet cached on the cluster.
# Currently cached:
#   - google/gemma-4-E4B-it   (~15G)
#   - google/gemma-4-26B-A4B-it (~49G)
#
# Downloads with this script:
#   - google/gemma-4-E2B-it   (smaller MoE, for faster iteration baselines)
#   - google/gemma-4-31B-it   (largest public Gemma 4, fits H100 80GB unquantized)
#
# CPU-only job (no GPU needed). Runs on a general partition node.
# Must NOT set HF_HUB_OFFLINE — we need network to download.

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean}
GEMMA_VENV=${GEMMA_VENV:-/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}

MODELS=(
  "google/gemma-4-E2B-it"
  "google/gemma-4-31B-it"
)

mkdir -p "$LOG_DIR" "$HF_CACHE"
cd "$REPO"

# Intentionally unset offline flags: we need network here.
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true
unset HF_DATASETS_OFFLINE || true
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Starting Gemma 4 downloads to $HF_CACHE"
df -h "$HF_CACHE" | tail -1

for MODEL in "${MODELS[@]}"; do
  echo
  echo "[$(date -Is)] === Downloading $MODEL ==="
  "$GEMMA_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
import os, sys
model = "$MODEL"
cache = os.environ["HUGGINGFACE_HUB_CACHE"]
print(f"Target model: {model}")
print(f"Cache dir:    {cache}")
path = snapshot_download(
    repo_id=model,
    cache_dir=cache,
    resume_download=True,
    max_workers=8,
)
print(f"Downloaded to: {path}")
PY
  echo "[$(date -Is)] === Done: $MODEL ==="
  du -sh "$HF_CACHE/models--${MODEL//\//--}" 2>/dev/null || true
done

echo
echo "[$(date -Is)] All downloads complete. Cache contents:"
ls -la "$HF_CACHE/" | grep "^d" | grep gemma
df -h "$HF_CACHE" | tail -1
