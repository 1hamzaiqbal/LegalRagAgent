#!/bin/bash
#SBATCH --job-name=opd_math_teacher
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_teacher_%j.out

set -euo pipefail
: "${OPD_MATH_DATA_ROOT:?Set an explicit audit or reviewed canonical data root}"
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
DATA_ROOT="$OPD_MATH_DATA_ROOT"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
SOURCE="${OPD_MATH_TEACHER_SOURCE:-M}"
MODEL="${OPD_MATH_TEACHER_MODEL:-Qwen/Qwen3-8B}"
REVISION="${OPD_MATH_TEACHER_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"
TRAIN_FREEZE="$FREEZE_ROOT/train.freeze.txt"
VERIFY_ENVIRONMENT="$REPO/scripts/opd_math/verify_environment.py"
OUT="$RUN_ROOT/smoke/teacher_${SOURCE}_${SLURM_JOB_ID}"

test "$SOURCE" = M || test "$SOURCE" = O
test -x "$ENV_DIR/bin/python"
test -f "$REPO/configs/opd_math/source_manifest.json"
test -f "$REPO/configs/opd_math/teacher_training_plan.json"
test -f "$TRAIN_FREEZE"
test -f "$DATA_ROOT/prepared_manifest.json"
test -f "$DATA_ROOT/roles/$SOURCE/teacher_train.jsonl"
mkdir -p "$RUN_ROOT/smoke"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source "$ENV_DIR/bin/activate"
echo "Verifying live train environment before teacher smoke"
"$ENV_DIR/bin/python" "$VERIFY_ENVIRONMENT" \
  --environment-root "$ENV_DIR" \
  --commit-freeze "$TRAIN_FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind train
python "$REPO/scripts/opd_math/train_teacher_grpo.py" \
  --model "$MODEL" \
  --model-revision "$REVISION" \
  --source "$SOURCE" \
  --budget-mode primary_matched \
  --task-file "$DATA_ROOT/roles/$SOURCE/teacher_train.jsonl" \
  --prepared-manifest "$DATA_ROOT/prepared_manifest.json" \
  --source-manifest "$REPO/configs/opd_math/source_manifest.json" \
  --training-plan "$REPO/configs/opd_math/teacher_training_plan.json" \
  --output-dir "$OUT" \
  --train-environment-root "$ENV_DIR" \
  --train-environment-freeze "$TRAIN_FREEZE" \
  --limit "${OPD_MATH_TEACHER_SMOKE_LIMIT:-16}" \
  --max-steps 1 \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-prompt-tokens 2304 \
  --max-completion-length 256 \
  --seed "${OPD_MATH_SEED:-0}" \
  --smoke \
  --local-files-only
echo "Re-verifying live train environment after teacher smoke"
"$ENV_DIR/bin/python" "$VERIFY_ENVIRONMENT" \
  --environment-root "$ENV_DIR" \
  --commit-freeze "$TRAIN_FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind train
test -f "$OUT/final_adapter/adapter_config.json"
test -f "$OUT/train_metrics.json"
test -f "$OUT/trainer_log_history.json"
echo "PASS teacher GRPO plumbing only: $OUT"
