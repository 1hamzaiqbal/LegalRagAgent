#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J embed-wave2
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Wave 2: Build 5 additional embedding collections on local /tmp, copy to NFS.
# Models: stella-1.5b, gte-qwen2-1.5b, jina-v3, arctic-l-v2, nomic-v2-moe
#
# Uses gemma4 venv for stella-1.5b (needs transformers>=5.5.0).
# Uses primary venv for the rest.
#
# Each model gets its own ChromaDB directory to avoid corruption.

set -uo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
EVAL_VENV="$REPO/.venv"
GEMMA_VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
LOCAL_BASE="/tmp/chroma_build_$$"

cd "$REPO"

export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/engrfs/tmp/jacobsn/hiqbal_legalrag/cache"
export TORCH_HOME="/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch"
mkdir -p "$XDG_CACHE_HOME" "$LOCAL_BASE"

# Ensure gemma venv has embedding deps (for stella-1.5b fallback)
echo "[$(date '+%F %T')] Checking gemma venv for sentence-transformers..."
if ! "$GEMMA_VENV/bin/python" -c "import sentence_transformers" 2>/dev/null; then
  echo "[$(date '+%F %T')] Installing sentence-transformers in gemma venv..."
  "$GEMMA_VENV/bin/pip" install --quiet sentence-transformers chromadb pandas 2>&1 | tail -3
fi

# Models to build (wave 2)
# stella-1.5b needs transformers>=5.5.0 (gemma venv), others use primary venv
MODELS="gte-qwen2-1.5b jina-v3 arctic-l-v2 nomic-v2-moe stella-1.5b"
FAILED=""

for model in $MODELS; do
  LOCAL_DIR="$LOCAL_BASE/$model"
  mkdir -p "$LOCAL_DIR"

  # Target on NFS — separate directory per model
  NFS_DIR="$REPO/chroma_db_${model//-/_}"

  # Skip if already built
  if [ -d "$NFS_DIR" ]; then
    echo "[$(date '+%F %T')] SKIP $model: $NFS_DIR already exists"
    continue
  fi

  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] Building: $model"
  echo "  Local: $LOCAL_DIR"
  echo "  NFS target: $NFS_DIR"
  echo "======================================================================"

  # Try primary venv first. If model needs newer transformers (stella-1.5b),
  # fall back to gemma venv which has transformers>=5.5.0 + sentence-transformers.
  BUILD_OK=0

  source "$EVAL_VENV/bin/activate"
  if CHROMA_DB_DIR="$LOCAL_DIR" python utils/fast_embed.py barexam --model "$model"; then
    BUILD_OK=1
  else
    echo "[$(date '+%F %T')] Primary venv failed for $model, trying gemma venv..."
    rm -rf "$LOCAL_DIR"
    mkdir -p "$LOCAL_DIR"
    source "$GEMMA_VENV/bin/activate"
    if CHROMA_DB_DIR="$LOCAL_DIR" python utils/fast_embed.py barexam --model "$model"; then
      BUILD_OK=1
    fi
  fi

  if [ "$BUILD_OK" -eq 1 ]; then
    echo "[$(date '+%F %T')] Build succeeded, copying to NFS..."
    cp -r "$LOCAL_DIR" "$NFS_DIR"
    echo "[$(date '+%F %T')] SUCCESS: $model → $NFS_DIR"
    # Verify (use primary venv which always has chromadb)
    source "$EVAL_VENV/bin/activate"
    CHROMA_DB_DIR="$NFS_DIR" python utils/fast_embed.py status
  else
    echo "[$(date '+%F %T')] FAILED: $model (both venvs)"
    FAILED="$FAILED $model"
  fi

  # Clean up local to save space
  rm -rf "$LOCAL_DIR"
done

echo ""
echo "======================================================================"
echo "[$(date '+%F %T')] Wave 2 builds complete"
echo "======================================================================"

if [ -n "$FAILED" ]; then
  echo "FAILED models:$FAILED"
  exit 1
else
  echo "All wave 2 builds succeeded!"
fi
