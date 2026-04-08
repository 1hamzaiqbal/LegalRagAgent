#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Parameterized embedding build script
# Usage: sbatch --job-name=embed-<name> slurm_build_embed.sh <short_name>
# Example: sbatch --job-name=embed-legal-bert slurm_build_embed.sh legal-bert
#
# Available models: legal-bert, stella-1.5b, bge-m3, stella-400m,
#                   arctic-l-v2, jina-v3, gte-qwen2-1.5b
#
# Estimated build times (686K barexam passages):
#   legal-bert (110M):   ~1.5h
#   stella-400m (400M):  ~2h
#   bge-m3 (568M):       ~2.5h
#   stella-1.5b (1.5B):  ~6h
#   gte-qwen2-1.5b (1.5B): ~6h

set -euo pipefail

EMBED_MODEL="${1:?Usage: sbatch slurm_build_embed.sh <embed_short_name>}"

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

echo "[$(date '+%F %T')] Building barexam embeddings with model: $EMBED_MODEL"
echo "[$(date '+%F %T')] Available models:"
python utils/fast_embed.py barexam --list-models 2>&1 | head -20

python utils/fast_embed.py barexam --model "$EMBED_MODEL"

echo "[$(date '+%F %T')] Embedding build complete for $EMBED_MODEL"
python utils/fast_embed.py status
