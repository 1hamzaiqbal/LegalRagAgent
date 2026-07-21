#!/bin/bash
#SBATCH --job-name=opd_objfam
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_objective_family_%j.out

set -euo pipefail
: "${OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root}"
: "${OPD_MATH_OBJECTIVE_ID:?Set one registered local objective ID}"
: "${OPD_MATH_STUDENT_SOURCE:?Set M or O}"
: "${OPD_MATH_SEED:?Set registered seed 0, 1, or 2}"
: "${OPD_MATH_CAMPAIGN_KIND:?Set diagnostic or scientific}"
: "${OPD_MATH_OBJECTIVE_OUT:?Set the fresh preregistered output directory}"
: "${OPD_MATH_STUDENT_SUPPORT_MANIFEST:?Set the same-commit support gate}"
: "${OPD_MATH_OBJECTIVE_PROMPT_PLAN:?Set the exact source/seed prompt plan}"
: "${OPD_MATH_OBJECTIVE_INITIALIZATION_MANIFEST:?Set the exact seed adapter manifest}"

OBJECTIVE_ID="$OPD_MATH_OBJECTIVE_ID"
case "$OBJECTIVE_ID" in
  task_rl|task_rl_k1_ungated_clip5|task_rl_k1_ungated_unclipped|task_rl_k1_gated_clip5_beta5|k1_bare_verl_compatible_clip10) ;;
  k1_verl_upstream_clip10)
    echo "upstream veRL has a separate pinned launcher" >&2
    exit 2
    ;;
  *) echo "invalid local objective-family ID" >&2; exit 2 ;;
esac
case "$OPD_MATH_STUDENT_SOURCE" in M|O) ;; *) echo "invalid student source" >&2; exit 2 ;; esac
case "$OPD_MATH_SEED" in 0|1|2) ;; *) echo "invalid objective-family seed" >&2; exit 2 ;; esac
case "$OPD_MATH_CAMPAIGN_KIND" in diagnostic|scientific) ;; *) echo "invalid campaign kind" >&2; exit 2 ;; esac

SOURCE="$OPD_MATH_STUDENT_SOURCE"
SEED="$OPD_MATH_SEED"
RUN_KEY="${OBJECTIVE_ID}__${SOURCE}__seed${SEED}"
RUN_ID="${OPD_MATH_OBJECTIVE_RUN_ID:-$RUN_KEY}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe run ID" >&2; exit 2; }
[[ "$OPD_MATH_OBJECTIVE_OUT" = /* ]] || { echo "objective output must be absolute" >&2; exit 2; }

if [[ "$OPD_MATH_CAMPAIGN_KIND" == diagnostic ]]; then
  test "$SEED" = 0 || { echo "one-step fidelity diagnostics use seed 0 only" >&2; exit 2; }
  STEPS=1
else
  STEPS=100
  : "${OPD_MATH_OBJECTIVE_PREREGISTRATION:?Scientific run requires sealed preregistration}"
  : "${OPD_MATH_OBJECTIVE_LAUNCH_PLAN:?Scientific run requires sealed launch plan}"
fi

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
TRAIN_ENV="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
SERVE_ENV="${OPD_MATH_SERVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_serve}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
DATA_ROOT="$OPD_MATH_DATA_ROOT"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
OUT="$OPD_MATH_OBJECTIVE_OUT"
PRELAUNCH_RECEIPT="$OUT.prelaunch.json"
TASK="$DATA_ROOT/roles/$SOURCE/student_opd.jsonl"
PREPARED="$DATA_ROOT/prepared_manifest.json"
LAUNCHER="$REPO/scripts/hpc/slurm_opd_math_objective_family_train.sh"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"
TRAIN_FREEZE="$FREEZE_ROOT/train.freeze.txt"
SERVE_FREEZE="$FREEZE_ROOT/serve.freeze.txt"
VERIFY_ENVIRONMENT="$REPO/scripts/opd_math/verify_environment.py"
PREREG_TOOL="$REPO/scripts/opd/objective_family_preregistration.py"
STUDENT="Qwen/Qwen3-1.7B"
STUDENT_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PORT=""
URL=""
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

test -z "$(git -C "$REPO" status --porcelain=v1)" || {
  echo "objective-family execution requires a clean EIT checkout" >&2
  exit 2
}
test -x "$TRAIN_ENV/bin/python"
test -f "$TASK"
test -f "$PREPARED"
test -f "$OPD_MATH_STUDENT_SUPPORT_MANIFEST"
test -f "$OPD_MATH_OBJECTIVE_PROMPT_PLAN"
test -f "$OPD_MATH_OBJECTIVE_INITIALIZATION_MANIFEST"
test -f "$TRAIN_FREEZE"
test -f "$LAUNCHER"
for artifact in \
  "$OUT" \
  "$PRELAUNCH_RECEIPT" \
  "$OUT.vllm.log" \
  "$OUT.server_models.json" \
  "$OUT.tokenizer_contract.json" \
  "$OUT.server_scoring_contract.json"; do
  if [[ -e "$artifact" || -L "$artifact" ]]; then
    echo "Refusing to reuse objective-family artifact path: $artifact" >&2
    exit 2
  fi
done

"$TRAIN_ENV/bin/python" "$VERIFY_ENVIRONMENT" \
  --environment-root "$TRAIN_ENV" \
  --commit-freeze "$TRAIN_FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind train

if [[ "$OPD_MATH_CAMPAIGN_KIND" == scientific ]]; then
  "$TRAIN_ENV/bin/python" "$PREREG_TOOL" prelaunch \
    --preregistration "$OPD_MATH_OBJECTIVE_PREREGISTRATION" \
    --launch-plan "$OPD_MATH_OBJECTIVE_LAUNCH_PLAN" \
    --run-key "$RUN_KEY" \
    --run-id "$RUN_ID" \
    --scheduler-job-id "$SLURM_JOB_ID" \
    --output "$PRELAUNCH_RECEIPT"
fi

export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TRAIN_ARGS=(
  --objective-id "$OBJECTIVE_ID"
  --student-source "$SOURCE"
  --task-file "$TASK"
  --task-limit "${OPD_MATH_TASK_LIMIT:-2161}"
  --budget-mode primary_matched
  --campaign-run-id "$RUN_ID"
  --scheduler-job-id "$SLURM_JOB_ID"
  --prepared-manifest "$PREPARED"
  --student "$STUDENT"
  --student-revision "$STUDENT_REVISION"
  --student-support-manifest "$OPD_MATH_STUDENT_SUPPORT_MANIFEST"
  --objective-family-prompt-plan "$OPD_MATH_OBJECTIVE_PROMPT_PLAN"
  --objective-family-initialization-manifest "$OPD_MATH_OBJECTIVE_INITIALIZATION_MANIFEST"
  --objective-family-launcher "$LAUNCHER"
  --train-environment-root "$TRAIN_ENV"
  --train-environment-freeze "$TRAIN_FREEZE"
  --out-dir "$OUT"
  --steps "$STEPS"
  --group-size 4
  --micro-prompts 1
  --max-new-tokens 512
  --max-prompt-tokens 1536
  --lr 1e-5
  --lora 32
  --grad-clip 1.0
  --seed "$SEED"
  --temperature 1.0
  --top-p 1.0
  --top-k 0
  --gradient-checkpointing
  --min-informative-group-fraction 0.05
  --require-parameter-update
  --local-files-only
)

if [[ "$OPD_MATH_CAMPAIGN_KIND" == diagnostic ]]; then
  TRAIN_ARGS+=(--objective-family-diagnostic)
else
  TRAIN_ARGS+=(--prelaunch-receipt "$PRELAUNCH_RECEIPT")
fi

if [[ "$OBJECTIVE_ID" != task_rl ]]; then
  : "${OPD_MATH_TEACHER_CHECKPOINT:?Teacher-scored objective requires passing O checkpoint}"
  : "${OPD_MATH_TEACHER_GAP_MANIFEST:?Teacher-scored objective requires passing O gap}"
  : "${OPD_MATH_TEACHER_PROVENANCE_MANIFEST:?Teacher-scored objective requires O provenance}"
  : "${OPD_MATH_TEACHER_BASE_MODEL:?Set the O teacher base model}"
  : "${OPD_MATH_TEACHER_BASE_REVISION:?Set the O teacher base revision}"
  test -x "$SERVE_ENV/bin/vllm"
  test -d "$OPD_MATH_TEACHER_CHECKPOINT"
  test -f "$OPD_MATH_TEACHER_GAP_MANIFEST"
  test -f "$OPD_MATH_TEACHER_PROVENANCE_MANIFEST"
  test -f "$SERVE_FREEZE"
  "$SERVE_ENV/bin/python" "$VERIFY_ENVIRONMENT" \
    --environment-root "$SERVE_ENV" \
    --commit-freeze "$SERVE_FREEZE" \
    --expected-commit "$COMMIT" \
    --freeze-kind serve \
    --expected-executable "$SERVE_ENV/bin/vllm"
  PORT="$("$TRAIN_ENV/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
  URL="http://127.0.0.1:$PORT"
  "$SERVE_ENV/bin/vllm" serve "$OPD_MATH_TEACHER_CHECKPOINT" \
    --host 127.0.0.1 \
    --served-model-name opd-math-teacher \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.55 >"$OUT.vllm.log" 2>&1 &
  SERVER_PID=$!
  READY=0
  for _ in $(seq 1 600); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -100 "$OUT.vllm.log"; exit 1; fi
    if curl -sf --max-time 2 "$URL/health" >/dev/null 2>&1; then READY=1; break; fi
    sleep 2
  done
  test "$READY" = 1
  kill -0 "$SERVER_PID"
  curl -sf --max-time 10 "$URL/v1/models" >"$OUT.server_models.json"
  "$TRAIN_ENV/bin/python" "$REPO/scripts/opd_math/tokenizer_contract.py" \
    --teacher "$OPD_MATH_TEACHER_CHECKPOINT" \
    --student "$STUDENT" \
    --student-revision "$STUDENT_REVISION" \
    --server-url "$URL" \
    --server-model opd-math-teacher \
    --output "$OUT.tokenizer_contract.json" \
    --local-files-only
  "$TRAIN_ENV/bin/python" "$REPO/scripts/opd_math/server_scoring_probe.py" \
    --tokenizer "$STUDENT" \
    --tokenizer-revision "$STUDENT_REVISION" \
    --server-url "$URL" \
    --server-model opd-math-teacher \
    --server-pid "$SERVER_PID" \
    --teacher-checkpoint "$OPD_MATH_TEACHER_CHECKPOINT" \
    --teacher-provenance-manifest "$OPD_MATH_TEACHER_PROVENANCE_MANIFEST" \
    --teacher-server-max-model-len 4096 \
    --serve-environment-root "$SERVE_ENV" \
    --output "$OUT.server_scoring_contract.json" \
    --local-files-only
  TRAIN_ARGS+=(
    --teacher-url "$URL"
    --teacher-model opd-math-teacher
    --teacher-checkpoint "$OPD_MATH_TEACHER_CHECKPOINT"
    --teacher-server-max-model-len 4096
    --teacher-base-model "$OPD_MATH_TEACHER_BASE_MODEL"
    --teacher-base-revision "$OPD_MATH_TEACHER_BASE_REVISION"
    --teacher-gap-manifest "$OPD_MATH_TEACHER_GAP_MANIFEST"
    --teacher-provenance-manifest "$OPD_MATH_TEACHER_PROVENANCE_MANIFEST"
    --tokenizer-contract "$OUT.tokenizer_contract.json"
    --server-scoring-contract "$OUT.server_scoring_contract.json"
    --serve-environment-root "$SERVE_ENV"
    --serve-environment-freeze "$SERVE_FREEZE"
  )
fi

"$TRAIN_ENV/bin/python" "$REPO/scripts/opd/opd_train.py" "${TRAIN_ARGS[@]}"
"$TRAIN_ENV/bin/python" "$VERIFY_ENVIRONMENT" \
  --environment-root "$TRAIN_ENV" \
  --commit-freeze "$TRAIN_FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind train
if [[ "$OBJECTIVE_ID" != task_rl ]]; then
  "$SERVE_ENV/bin/python" "$VERIFY_ENVIRONMENT" \
    --environment-root "$SERVE_ENV" \
    --commit-freeze "$SERVE_FREEZE" \
    --expected-commit "$COMMIT" \
    --freeze-kind serve \
    --expected-executable "$SERVE_ENV/bin/vllm"
fi
echo "Objective-family run completed; held-out evaluation remains forbidden until campaign-wide release: $OUT"
