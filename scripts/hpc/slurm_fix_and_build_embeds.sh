#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J embed-fix
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Rebuild all ChromaDB embeddings from scratch
# Baseline (gte-large) + 3 comparison models, all sequential
# Total estimated: ~8h (2.2h baseline + ~6h for 3 models)

set -uo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV="$REPO/.venv"
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache

source "$VENV/bin/activate"
cd "$REPO"

export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/engrfs/tmp/jacobsn/hiqbal_legalrag/cache"
mkdir -p "$XDG_CACHE_HOME"

echo "[$(date '+%F %T')] Building all embedding collections from scratch"
echo "[$(date '+%F %T')] chroma_db/ should be empty (deleted corrupt DB)"

# Baseline first, then comparison models
MODELS="gte-large legal-bert stella-400m bge-m3"
FAILED=""

for model in $MODELS; do
  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] Building: $model"
  echo "======================================================================"

  if python utils/fast_embed.py barexam --model "$model"; then
    echo "[$(date '+%F %T')] SUCCESS: $model"
  else
    echo "[$(date '+%F %T')] FAILED: $model (continuing with next)"
    FAILED="$FAILED $model"
  fi
done

echo ""
echo "======================================================================"
echo "[$(date '+%F %T')] Final status"
echo "======================================================================"
python utils/fast_embed.py status

if [ -n "$FAILED" ]; then
  echo "FAILED models:$FAILED"
  exit 1
else
  echo "All builds succeeded!"
fi
