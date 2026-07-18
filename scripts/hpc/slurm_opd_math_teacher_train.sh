#!/bin/bash
#SBATCH --job-name=opd_math_tfull
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_teacher_full_%j.out

set -euo pipefail
: "${OPD_MATH_TEACHER_LIMIT:?Set the matched teacher-example limit from prepared_manifest.json}"

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
DATA_ROOT="${OPD_MATH_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
SOURCE="${OPD_MATH_TEACHER_SOURCE:-M}"
BUDGET_MODE="${OPD_MATH_BUDGET_MODE:-primary_matched}"
MODEL="${OPD_MATH_TEACHER_MODEL:-Qwen/Qwen3-8B}"
REVISION="${OPD_MATH_TEACHER_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
TRAINING_PLAN="$REPO/configs/opd_math/teacher_training_plan.json"
OUT="$RUN_ROOT/teachers/$SOURCE/run_${SLURM_JOB_ID}"

test "$SOURCE" = M || test "$SOURCE" = O
case "$BUDGET_MODE" in primary_matched|dose_response) ;; *) echo "invalid OPD_MATH_BUDGET_MODE" >&2; exit 2 ;; esac
test -x "$ENV_DIR/bin/python"
test -f "$REPO/configs/opd_math/source_manifest.json"
test -f "$TRAINING_PLAN"
test -f "$DATA_ROOT/prepared_manifest.json"
test -f "$DATA_ROOT/roles/$SOURCE/teacher_train.jsonl"
mkdir -p "$(dirname "$OUT")"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source "$ENV_DIR/bin/activate"
python "$REPO/scripts/opd_math/train_teacher_grpo.py" \
  --model "$MODEL" \
  --model-revision "$REVISION" \
  --source "$SOURCE" \
  --budget-mode "$BUDGET_MODE" \
  --task-file "$DATA_ROOT/roles/$SOURCE/teacher_train.jsonl" \
  --prepared-manifest "$DATA_ROOT/prepared_manifest.json" \
  --source-manifest "$REPO/configs/opd_math/source_manifest.json" \
  --training-plan "$TRAINING_PLAN" \
  --output-dir "$OUT" \
  --limit "$OPD_MATH_TEACHER_LIMIT" \
  --max-steps "${OPD_MATH_MAX_STEPS:-100}" \
  --num-generations "${OPD_MATH_NUM_GENERATIONS:-4}" \
  --gradient-accumulation-steps "${OPD_MATH_GRAD_ACCUM:-4}" \
  --max-prompt-tokens "${OPD_MATH_MAX_PROMPT_TOKENS:-1536}" \
  --max-completion-length "${OPD_MATH_MAX_COMPLETION:-1024}" \
  --learning-rate "${OPD_MATH_TEACHER_LEARNING_RATE:-2e-5}" \
  --lora-r "${OPD_MATH_TEACHER_LORA_R:-16}" \
  --seed "${OPD_MATH_SEED:-0}" \
  --require-informative-reward \
  --local-files-only
test -f "$OUT/final_adapter/adapter_config.json"
test -f "$OUT/train_metrics.json"
test -f "$OUT/trainer_log_history.json"
echo "Teacher training completed; held-out quality is still unestablished: $OUT"
