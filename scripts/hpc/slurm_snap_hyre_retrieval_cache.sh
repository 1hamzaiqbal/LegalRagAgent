#!/bin/bash
# Build and audit full-dataset retrieval-id caches for top-k selection.
#
# Default pass is API-free and builds deterministic raw-question plus
# golden-neighbor caches. Set QUERY_TYPES=hyre_cache and HYRE_MODELS=... after
# Snap-HyRE generation caches exist.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J snap-hyre-ret-cache
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
CACHE_DIR=${CACHE_DIR:-$REPO/caches/retrieval/full}
HYRE_CACHE_ROOT=${HYRE_CACHE_ROOT:-$REPO/caches/hyre}
HYRE_CACHE_PATTERN=${HYRE_CACHE_PATTERN:-"$HYRE_CACHE_ROOT/{dataset}_{model}_{mode}.jsonl"}
QUESTIONS=${QUESTIONS:-full}
MAX_K=${MAX_K:-10}
KS=${KS:-1,3,5,10}
ALIGN_MIN_EXISTS=${ALIGN_MIN_EXISTS:-0.95}

if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASETS_ARR=(${DATASETS})
else
  DATASETS_ARR=(barexam housing casehold legalbench_scalr)
fi

if [[ -n "${QUERY_TYPES:-}" ]]; then
  # shellcheck disable=SC2206
  QUERY_TYPES_ARR=(${QUERY_TYPES})
else
  QUERY_TYPES_ARR=(raw_question golden_neighbors)
fi

if [[ -n "${HYRE_MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  HYRE_MODELS_ARR=(${HYRE_MODELS})
else
  HYRE_MODELS_ARR=()
fi

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

source "$EVAL_VENV/bin/activate"

echo "[$(date -Is)] repo=$REPO commit=$(git rev-parse --short HEAD)"
echo "[$(date -Is)] chroma=$CHROMA_DB_DIR cache_dir=$CACHE_DIR questions=$QUESTIONS max_k=$MAX_K"
echo "[$(date -Is)] datasets=${DATASETS_ARR[*]} query_types=${QUERY_TYPES_ARR[*]} hyre_models=${HYRE_MODELS_ARR[*]:-none}"
git status --short --branch

python -m py_compile \
  eval/eval_config.py \
  eval/eval_harness.py \
  rag_utils.py \
  scripts/build_retrieval_cache.py \
  scripts/audit_retrieval_cache.py \
  scripts/audit_retrieval_id_alignment.py \
  scripts/compile_retrieval_cache_matrix.py

outputs=()

for dataset in "${DATASETS_ARR[@]}"; do
  alignment_report="$CACHE_DIR/retrieval_id_alignment_${dataset}.txt"
  echo
  echo "[$(date -Is)] audit retrieval-id alignment dataset=$dataset report=$alignment_report"
  if python scripts/audit_retrieval_id_alignment.py \
    --dataset "$dataset" \
    --questions "$QUESTIONS" \
    --min-exists "$ALIGN_MIN_EXISTS" > "$alignment_report" 2>&1; then
    echo "[$(date -Is)] alignment OK dataset=$dataset"
  else
    echo "[$(date -Is)] WARNING: alignment failed dataset=$dataset; retrieval Hit/MRR is not promotable without a qrel fix"
    cat "$alignment_report"
  fi

  for query_type in "${QUERY_TYPES_ARR[@]}"; do
    case "$query_type" in
      raw_question|golden_neighbors)
        out="$CACHE_DIR/${dataset}_${query_type}_k${MAX_K}.jsonl"
        echo
        echo "[$(date -Is)] build dataset=$dataset query_type=$query_type out=$out"
        python scripts/build_retrieval_cache.py \
          --dataset "$dataset" \
          --questions "$QUESTIONS" \
          --query-type "$query_type" \
          --max-k "$MAX_K" \
          --out "$out"
        python scripts/audit_retrieval_cache.py \
          --cache "$out" \
          --dataset "$dataset" \
          --query-type "$query_type" \
          --min-k "$MAX_K" \
          --ks "$KS"
        outputs+=("$out")
        ;;
      hyde_cache|hyre_cache)
        if [[ "${#HYRE_MODELS_ARR[@]}" -eq 0 ]]; then
          echo "[$(date -Is)] ERROR: HYRE_MODELS is required when QUERY_TYPES includes $query_type"
          exit 2
        fi
        generation_mode=snap_hyre
        if [[ "$query_type" == "hyde_cache" ]]; then
          generation_mode=rag_hyde
        fi
        for model in "${HYRE_MODELS_ARR[@]}"; do
          hyre_cache=${HYRE_CACHE_PATTERN//\{dataset\}/$dataset}
          hyre_cache=${hyre_cache//\{model\}/$model}
          hyre_cache=${hyre_cache//\{mode\}/$generation_mode}
          if [[ ! -f "$hyre_cache" ]]; then
            echo "[$(date -Is)] ERROR: missing HyRE cache $hyre_cache"
            exit 2
          fi
          out="$CACHE_DIR/${dataset}_${model}_${generation_mode}_k${MAX_K}.jsonl"
          echo
          echo "[$(date -Is)] build dataset=$dataset query_type=$query_type model=$model generation_mode=$generation_mode hyre_cache=$hyre_cache out=$out"
          python scripts/build_retrieval_cache.py \
            --dataset "$dataset" \
            --questions "$QUESTIONS" \
            --query-type "$query_type" \
            --hyre-cache-path "$hyre_cache" \
            --max-k "$MAX_K" \
            --out "$out"
          python scripts/audit_retrieval_cache.py \
            --cache "$out" \
            --dataset "$dataset" \
            --query-type "$query_type" \
            --min-k "$MAX_K" \
            --ks "$KS"
          outputs+=("$out")
        done
        ;;
      *)
        echo "[$(date -Is)] ERROR: unknown query_type=$query_type"
        exit 2
        ;;
    esac
  done
done

if [[ "${#outputs[@]}" -gt 0 ]]; then
  matrix_md="$CACHE_DIR/retrieval_cache_matrix.md"
  matrix_csv="$CACHE_DIR/retrieval_cache_matrix.csv"
  cache_args=()
  for out in "${outputs[@]}"; do
    cache_args+=(--cache "$out")
  done
  python scripts/compile_retrieval_cache_matrix.py \
    "${cache_args[@]}" \
    --ks "$KS" \
    --min-k "$MAX_K" \
    --out-md "$matrix_md" \
    --out-csv "$matrix_csv"
  echo "[$(date -Is)] wrote matrix $matrix_md"
  echo "[$(date -Is)] wrote matrix $matrix_csv"
fi

echo "[$(date -Is)] retrieval cache job complete."
