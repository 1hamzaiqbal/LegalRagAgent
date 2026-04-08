#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -J embed-barexam
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Build ChromaDB embeddings for barexam corpus (686K passages)
# Uses gte-large-en-v1.5 (default, already cached)
# Estimated: 7-13h on A40, writes to ./chroma_db/
# This unblocks all RAG eval modes (rag_simple, rag_snap_hyde, ce_threshold, etc.)

set -euo pipefail

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

echo "[$(date '+%F %T')] Starting barexam embedding build"
echo "[$(date '+%F %T')] Model: gte-large-en-v1.5 (default)"
echo "[$(date '+%F %T')] Corpus: datasets/barexam_qa/barexam_qa_train.csv"

python utils/fast_embed.py barexam

echo "[$(date '+%F %T')] Embedding build complete"
python utils/fast_embed.py status
