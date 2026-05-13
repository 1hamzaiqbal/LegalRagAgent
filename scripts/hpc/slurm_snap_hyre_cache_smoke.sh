#!/bin/bash
# Smoke-test the comprehensive Snap-HyRE cache path on a compute node.
#
# This is intentionally tiny: it validates imports, dataset/chroma visibility,
# and one raw-question retrieval-cache row per headline legal benchmark. It does
# not call an LLM and should run before any full-corpus jobs.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J snap-hyre-cache-smoke
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-snap-hyre-comprehensive}
DATA_REPO=${DATA_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
EVAL_VENV=${EVAL_VENV:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}
CACHE_DIR=${CACHE_DIR:-$REPO/caches/retrieval/smoke}

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$CACHE_DIR"
ln -sfn "$DATA_REPO/datasets" "$REPO/datasets"
ln -sfn "$DATA_REPO/chroma_db" "$REPO/chroma_db"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TORCH_HOME="$TORCH_HOME"
export TRITON_CACHE_DIR="/tmp/hiqbal-triton/${SLURM_JOB_ID:-local}"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
mkdir -p "$TRITON_CACHE_DIR" "$UV_CACHE_DIR"
export CHROMA_DB_DIR="$CHROMA_DB_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export DISABLE_CROSS_ENCODER=1

source "$EVAL_VENV/bin/activate"

echo "[$(date -Is)] repo=$REPO"
echo "[$(date -Is)] data_repo=$DATA_REPO"
echo "[$(date -Is)] chroma=$CHROMA_DB_DIR"
git rev-parse --short HEAD
git status --short --branch

python -m py_compile \
  eval/eval_config.py \
  eval/eval_harness.py \
  llm_config.py \
  rag_utils.py \
  scripts/build_retrieval_cache.py \
  scripts/audit_retrieval_cache.py

python - <<'PY'
import sqlite3
from pathlib import Path

db = Path("chroma_db/chroma.sqlite3")
required = {"legal_passages", "housing_statutes", "casehold_holdings", "legalbench_scalr_holdings"}
con = sqlite3.connect(db)
names = {row[0] for row in con.execute("select name from collections")}
con.close()
print("[preflight] collections=" + ",".join(sorted(names)))
missing = required - names
if missing:
    raise SystemExit("[preflight] missing collections: " + ",".join(sorted(missing)))
PY

python - <<'PY'
import sys
sys.path.insert(0, "eval")
from eval_harness import MODE_RUNNERS

if "snap_hyre" not in MODE_RUNNERS:
    raise SystemExit("snap_hyre mode missing")
print("[preflight] snap_hyre runner=" + MODE_RUNNERS["snap_hyre"].__name__)
PY

for dataset in barexam housing casehold legalbench_scalr; do
  out="$CACHE_DIR/${dataset}_raw_question_k3_smoke.jsonl"
  echo
  echo "[$(date -Is)] cache smoke dataset=$dataset out=$out"
  python scripts/build_retrieval_cache.py \
    --dataset "$dataset" \
    --questions 1 \
    --query-type raw_question \
    --max-k 3 \
    --out "$out"
  python scripts/audit_retrieval_cache.py \
    --cache "$out" \
    --dataset "$dataset" \
    --query-type raw_question \
    --min-k 3 \
    --ks 1,3
done

echo "[$(date -Is)] Snap-HyRE cache smoke complete."
