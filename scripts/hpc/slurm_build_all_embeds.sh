#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J embed-all
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Build ALL embedding collections SEQUENTIALLY to avoid ChromaDB SQLite
# concurrent write issues on NFS. One writer at a time.
#
# Models: legal-bert (~1.5h), stella-400m (~2h), bge-m3 (~2.5h)
# Total estimated: ~6h
#
# stella-1.5b skipped for now (DynamicCache compat issue with transformers 4.57.6)

set -uo pipefail  # no -e so we continue on individual model failures

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

MODELS="legal-bert stella-400m bge-m3"
FAILED=""

for model in $MODELS; do
  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] Building embeddings: $model"
  echo "======================================================================"

  if python utils/fast_embed.py barexam --model "$model"; then
    echo "[$(date '+%F %T')] SUCCESS: $model"
  else
    echo "[$(date '+%F %T')] FAILED: $model"
    FAILED="$FAILED $model"
  fi
done

echo ""
echo "======================================================================"
echo "[$(date '+%F %T')] All builds complete"
echo "======================================================================"
python utils/fast_embed.py status

if [ -n "$FAILED" ]; then
  echo ""
  echo "FAILED models:$FAILED"
  exit 1
fi
