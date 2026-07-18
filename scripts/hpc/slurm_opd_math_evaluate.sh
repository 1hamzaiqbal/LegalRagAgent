#!/bin/bash
#SBATCH --job-name=opd_math_eval
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_eval_%j.out

set -euo pipefail
: "${OPD_MATH_EVAL_SOURCE:?Set M or O}"
: "${OPD_MATH_EVAL_ROLE:?Set teacher_skill_dev, target_gap_dev, student_support, source_holdout, or external_M_test}"
: "${OPD_MATH_EVAL_MODEL:?Set the pinned base model name}"
: "${OPD_MATH_EVAL_MODEL_REVISION:?Set the pinned base model revision}"
: "${OPD_MATH_EVAL_MAX_RECORDS:?Set an explicit record budget; use 0 only for the full role file}"
: "${OPD_MATH_EVAL_LABEL:?Set a filesystem-safe output label such as M_base or M_trained}"

case "$OPD_MATH_EVAL_SOURCE" in M|O) ;; *) echo "invalid OPD_MATH_EVAL_SOURCE" >&2; exit 2 ;; esac
case "$OPD_MATH_EVAL_ROLE" in
  teacher_skill_dev|target_gap_dev) TASK_REL="roles/$OPD_MATH_EVAL_SOURCE/teacher_gap_dev.jsonl" ;;
  student_support) TASK_REL="roles/$OPD_MATH_EVAL_SOURCE/student_opd.jsonl" ;;
  source_holdout) TASK_REL="roles/$OPD_MATH_EVAL_SOURCE/source_holdout.jsonl" ;;
  external_M_test)
    test "$OPD_MATH_EVAL_SOURCE" = M
    TASK_REL="eval/M_test.jsonl"
    ;;
  *) echo "invalid OPD_MATH_EVAL_ROLE" >&2; exit 2 ;;
esac
if [[ "$OPD_MATH_EVAL_ROLE" == student_support || "$OPD_MATH_EVAL_ROLE" == source_holdout ]]; then
  EVAL_TEMPERATURE="${OPD_MATH_EVAL_TEMPERATURE:-1.0}"
  EVAL_TOP_P="${OPD_MATH_EVAL_TOP_P:-1.0}"
  EVAL_TOP_K="${OPD_MATH_EVAL_TOP_K:-0}"
  EVAL_MAX_NEW_TOKENS="${OPD_MATH_EVAL_MAX_NEW_TOKENS:-512}"
else
  EVAL_TEMPERATURE="${OPD_MATH_EVAL_TEMPERATURE:-0.7}"
  EVAL_TOP_P="${OPD_MATH_EVAL_TOP_P:-0.8}"
  EVAL_TOP_K="${OPD_MATH_EVAL_TOP_K:-20}"
  EVAL_MAX_NEW_TOKENS="${OPD_MATH_EVAL_MAX_NEW_TOKENS:-1024}"
fi
[[ "$OPD_MATH_EVAL_LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe OPD_MATH_EVAL_LABEL" >&2; exit 2; }

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
DATA_ROOT="${OPD_MATH_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
TASK="$DATA_ROOT/$TASK_REL"
OUT="$RUN_ROOT/evaluations/$OPD_MATH_EVAL_ROLE/$OPD_MATH_EVAL_LABEL/run_${SLURM_JOB_ID}"

test -x "$ENV_DIR/bin/python"
test -f "$TASK"
test -f "$DATA_ROOT/prepared_manifest.json"
mkdir -p "$(dirname "$OUT")"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source "$ENV_DIR/bin/activate"

ARGS=(
  --model "$OPD_MATH_EVAL_MODEL"
  --model-revision "$OPD_MATH_EVAL_MODEL_REVISION"
  --task-file "$TASK"
  --output-dir "$OUT"
  --max-records "$OPD_MATH_EVAL_MAX_RECORDS"
  --samples-per-problem "${OPD_MATH_EVAL_SAMPLES_PER_PROBLEM:-4}"
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
  --temperature "$EVAL_TEMPERATURE"
  --top-p "$EVAL_TOP_P"
  --top-k "$EVAL_TOP_K"
  --seed "${OPD_MATH_SEED:-0}"
  --write-completions
  --local-files-only
)
if [[ -n "${OPD_MATH_EVAL_ADAPTER:-}" ]]; then
  test -f "$OPD_MATH_EVAL_ADAPTER/adapter_config.json"
  ARGS+=(--adapter "$OPD_MATH_EVAL_ADAPTER")
fi
python "$REPO/scripts/opd_math/evaluate_math.py" "${ARGS[@]}"
test -f "$OUT/summary.json"
test -f "$OUT/samples.jsonl"
echo "PASS evaluation artifact only; no gate inferred: $OUT"
