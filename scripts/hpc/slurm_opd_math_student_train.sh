#!/bin/bash
#SBATCH --job-name=opd_math_sfull
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_student_full_%j.out

set -euo pipefail
: "${OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root}"
: "${OPD_MATH_STUDENT_MODE:?Set task_rl or task_rl_k1_gap explicitly}"
: "${OPD_MATH_STUDENT_STEPS:?Set an explicit student optimizer-step budget}"
: "${OPD_MATH_TASK_LIMIT:?Set the matched student-pool limit from prepared_manifest.json}"
: "${OPD_MATH_BUDGET_MODE:?Set primary_matched or dose_response}"
: "${OPD_MATH_STUDENT_SUPPORT_MANIFEST:?Set the passing student-support manifest}"

MODE="$OPD_MATH_STUDENT_MODE"
case "$MODE" in task_rl|task_rl_k1_gap) ;; *) echo "invalid OPD_MATH_STUDENT_MODE" >&2; exit 2 ;; esac
case "$OPD_MATH_BUDGET_MODE" in primary_matched|dose_response) ;; *) echo "invalid OPD_MATH_BUDGET_MODE" >&2; exit 2 ;; esac

if [[ "$MODE" == task_rl ]]; then
  : "${OPD_MATH_STUDENT_SOURCE:?task_rl requires M or O; it has no teacher coordinate}"
  case "$OPD_MATH_STUDENT_SOURCE" in M|O) ;; *) echo "invalid OPD_MATH_STUDENT_SOURCE" >&2; exit 2 ;; esac
  STUDENT_SOURCE="$OPD_MATH_STUDENT_SOURCE"
  RUN_KEY="baseline_$STUDENT_SOURCE"
else
  : "${OPD_MATH_PAIR:?task_rl_k1_gap requires one of M_M, M_O, O_M, O_O}"
  case "$OPD_MATH_PAIR" in M_M|M_O|O_M|O_O) ;; *) echo "invalid OPD_MATH_PAIR" >&2; exit 2 ;; esac
  TEACHER_SOURCE="${OPD_MATH_PAIR%%_*}"
  STUDENT_SOURCE="${OPD_MATH_PAIR##*_}"
  RUN_KEY="$OPD_MATH_PAIR"
fi

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
TRAIN_ENV="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
SERVE_ENV="${OPD_MATH_SERVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_serve}"
DATA_ROOT="$OPD_MATH_DATA_ROOT"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
STUDENT="Qwen/Qwen3-1.7B"
STUDENT_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"
TRAIN_FREEZE="$FREEZE_ROOT/train.freeze.txt"
SERVE_FREEZE="$FREEZE_ROOT/serve.freeze.txt"
OUT="$RUN_ROOT/students/$RUN_KEY/$MODE/run_${SLURM_JOB_ID}"
TASK="$DATA_ROOT/roles/$STUDENT_SOURCE/student_opd.jsonl"
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

test -x "$TRAIN_ENV/bin/python"
test -f "$TASK"
test -f "$DATA_ROOT/prepared_manifest.json"
test -f "$OPD_MATH_STUDENT_SUPPORT_MANIFEST"
test -f "$TRAIN_FREEZE"
for artifact in \
  "$OUT" \
  "$OUT.vllm.log" \
  "$OUT.server_models.json" \
  "$OUT.tokenizer_contract.json" \
  "$OUT.server_scoring_contract.json"; do
  if [[ -e "$artifact" || -L "$artifact" ]]; then
    echo "Refusing to reuse student-run artifact path: $artifact" >&2
    exit 2
  fi
done
mkdir -p "$(dirname "$OUT")"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TRAIN_ARGS=(
  --mode "$MODE"
  --task-file "$TASK"
  --task-limit "$OPD_MATH_TASK_LIMIT"
  --budget-mode "$OPD_MATH_BUDGET_MODE"
  --prepared-manifest "$DATA_ROOT/prepared_manifest.json"
  --student "$STUDENT"
  --student-revision "$STUDENT_REVISION"
  --student-support-manifest "$OPD_MATH_STUDENT_SUPPORT_MANIFEST"
  --train-environment-freeze "$TRAIN_FREEZE"
  --out-dir "$OUT"
  --steps "$OPD_MATH_STUDENT_STEPS"
  --group-size "${OPD_MATH_GROUP_SIZE:-4}"
  --micro-prompts "${OPD_MATH_MICRO_PROMPTS:-1}"
  --max-new-tokens "${OPD_MATH_STUDENT_MAX_NEW_TOKENS:-512}"
  --seed "${OPD_MATH_SEED:-0}"
  --top-k 0
  --min-informative-group-fraction "${OPD_MATH_MIN_INFORMATIVE_GROUP_FRACTION:-0.05}"
  --require-parameter-update
  --local-files-only
)

if [[ "$MODE" == task_rl ]]; then
  TRAIN_ARGS+=(--student-source "$STUDENT_SOURCE")
else
  TRAIN_ARGS+=(--pair-id "$OPD_MATH_PAIR")
fi

if [[ "$MODE" == task_rl_k1_gap ]]; then
  : "${OPD_MATH_TEACHER_CHECKPOINT:?Set the merged, teacher-gap-passing checkpoint}"
  : "${OPD_MATH_TEACHER_GAP_MANIFEST:?Set the passing teacher-gap manifest}"
  : "${OPD_MATH_TEACHER_PROVENANCE_MANIFEST:?Set merge_provenance.json for the checkpoint}"
  : "${OPD_MATH_TEACHER_BASE_MODEL:?Set the base model recorded by the teacher gate}"
  : "${OPD_MATH_TEACHER_BASE_REVISION:?Set the base revision recorded by the teacher gate}"
  test -x "$SERVE_ENV/bin/vllm"
  test -d "$OPD_MATH_TEACHER_CHECKPOINT"
  test -f "$OPD_MATH_TEACHER_GAP_MANIFEST"
  test -f "$OPD_MATH_TEACHER_PROVENANCE_MANIFEST"
  test -f "$SERVE_FREEZE"
  PORT="$("$TRAIN_ENV/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
  URL="http://127.0.0.1:$PORT"
  "$SERVE_ENV/bin/vllm" serve "$OPD_MATH_TEACHER_CHECKPOINT" \
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
  curl -sf --max-time 2 "$URL/health" >/dev/null
  curl -sf --max-time 10 "$URL/v1/models" >"$OUT.server_models.json"
  LIVE_TOKENIZER_CONTRACT="$OUT.tokenizer_contract.json"
  LIVE_SCORING_CONTRACT="$OUT.server_scoring_contract.json"
  "$TRAIN_ENV/bin/python" "$REPO/scripts/opd_math/tokenizer_contract.py" \
    --teacher "$OPD_MATH_TEACHER_CHECKPOINT" \
    --student "$STUDENT" \
    --student-revision "$STUDENT_REVISION" \
    --server-url "$URL" \
    --server-model opd-math-teacher \
    --output "$LIVE_TOKENIZER_CONTRACT" \
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
    --output "$LIVE_SCORING_CONTRACT" \
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
    --tokenizer-contract "$LIVE_TOKENIZER_CONTRACT"
    --server-scoring-contract "$LIVE_SCORING_CONTRACT"
    --serve-environment-freeze "$SERVE_FREEZE"
  )
fi

"$TRAIN_ENV/bin/python" "$REPO/scripts/opd/opd_train.py" "${TRAIN_ARGS[@]}"
echo "Student run completed; task metrics still require held-out evaluation: $OUT"
