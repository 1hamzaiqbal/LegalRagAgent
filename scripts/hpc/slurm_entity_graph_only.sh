#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 0
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J entity-graph
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Build entity graph ONLY (no case summaries, no GPU needed).
# Uses spaCy for NLP entity extraction — CPU only.
# Previous run crashed on save (dict.most_common bug, now fixed).

set -uo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV="$REPO/.venv"

cd "$REPO"

export PYTHONUNBUFFERED=1

source "$VENV/bin/activate"

echo "[$(date '+%F %T')] Building NLP entity graph with spaCy..."
python utils/build_entity_graph.py --spacy

echo "[$(date '+%F %T')] Checking output..."
python utils/build_entity_graph.py --status

echo "[$(date '+%F %T')] Entity graph build complete"
