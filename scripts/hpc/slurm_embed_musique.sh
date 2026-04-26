#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J embed-musique
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Embed MuSiQue's 48k validation paragraphs into ChromaDB collection
# `musique_passages` using the default GTE-large-en-v1.5 (matches BarExam).
# Estimate: ~5-10 min on A40 (vs ~2.2h for BarExam 686k).

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV="$REPO/.venv"
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache

source "$VENV/bin/activate"
cd "$REPO"

export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/engrfs/tmp/jacobsn/hiqbal_legalrag/cache
mkdir -p "$XDG_CACHE_HOME"

# Make sure dataset CSV is on the cluster (it's gitignored — pull from local)
if [[ ! -f datasets/musique/passages.csv ]]; then
  echo "[$(date '+%F %T')] ERROR: datasets/musique/passages.csv not found." >&2
  echo "Download locally first: uv run python utils/download_new_datasets.py musique" >&2
  echo "Then scp datasets/musique/ wustl:$REPO/datasets/" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Embedding MuSiQue corpus → musique_passages"
python utils/fast_embed.py musique

echo "[$(date '+%F %T')] Embedding complete"
python utils/fast_embed.py status
