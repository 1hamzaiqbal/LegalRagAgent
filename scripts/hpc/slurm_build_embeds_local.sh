#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J embed-local
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Build embedding collections on LOCAL disk (/tmp) to avoid NFS+ChromaDB issues,
# then copy back to NFS. Each model gets its own chroma directory.
#
# This uses CHROMA_DB_DIR env var added to fast_embed.py and rag_utils.py.
# At eval time: CHROMA_DB_DIR=./chroma_db_<model> python eval/eval_harness.py ...

set -uo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV="$REPO/.venv"
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
LOCAL_BASE="/tmp/chroma_build_$$"

source "$VENV/bin/activate"
cd "$REPO"

export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/engrfs/tmp/jacobsn/hiqbal_legalrag/cache"
mkdir -p "$XDG_CACHE_HOME" "$LOCAL_BASE"

# First rebuild the baseline on local disk
MODELS="gte-large legal-bert stella-400m bge-m3"
FAILED=""

for model in $MODELS; do
  LOCAL_DIR="$LOCAL_BASE/$model"
  mkdir -p "$LOCAL_DIR"

  # Target on NFS — separate directory per model
  if [ "$model" = "gte-large" ]; then
    NFS_DIR="$REPO/chroma_db"
  else
    NFS_DIR="$REPO/chroma_db_${model//-/_}"
  fi

  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] Building: $model"
  echo "  Local: $LOCAL_DIR"
  echo "  NFS target: $NFS_DIR"
  echo "======================================================================"

  if CHROMA_DB_DIR="$LOCAL_DIR" python utils/fast_embed.py barexam --model "$model"; then
    echo "[$(date '+%F %T')] Build succeeded, copying to NFS..."
    rm -rf "$NFS_DIR"
    cp -r "$LOCAL_DIR" "$NFS_DIR"
    echo "[$(date '+%F %T')] SUCCESS: $model → $NFS_DIR"
    # Verify
    CHROMA_DB_DIR="$NFS_DIR" python utils/fast_embed.py status
  else
    echo "[$(date '+%F %T')] FAILED: $model"
    FAILED="$FAILED $model"
  fi

  # Clean up local to save space
  rm -rf "$LOCAL_DIR"
done

echo ""
echo "======================================================================"
echo "[$(date '+%F %T')] All builds complete"
echo "======================================================================"

if [ -n "$FAILED" ]; then
  echo "FAILED models:$FAILED"
  exit 1
else
  echo "All builds succeeded!"
  echo ""
  echo "To eval with a specific embedder:"
  echo "  CHROMA_DB_DIR=./chroma_db_legal_bert EVAL_EMBEDDING_MODEL=nlpaueb/legal-bert-base-uncased \\"
  echo "    python eval/eval_harness.py --mode snap_hyde ..."
fi
