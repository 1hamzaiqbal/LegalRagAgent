#!/bin/bash
#SBATCH --job-name=opd_math_student
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_student_%j.out

set -euo pipefail
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
TRAIN_ENV="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
SERVE_ENV="${OPD_MATH_SERVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_serve}"
DATA_ROOT="${OPD_MATH_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
PAIR="${OPD_MATH_SMOKE_PAIR:-M_M}"
case "$PAIR" in M_M|M_O|O_M|O_O) ;; *) echo "invalid OPD_MATH_SMOKE_PAIR" >&2; exit 2 ;; esac
SOURCE="${PAIR##*_}"
TEACHER="${OPD_MATH_TEACHER_CHECKPOINT:-Qwen/Qwen3-8B}"
TEACHER_REVISION="${OPD_MATH_TEACHER_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
STUDENT="Qwen/Qwen3-1.7B"
STUDENT_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PORT=""
URL=""
OUT="$RUN_ROOT/smoke/student_${SOURCE}_${SLURM_JOB_ID}"
SERVER_LOG="$OUT/vllm.log"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

test -x "$TRAIN_ENV/bin/python"
test -x "$SERVE_ENV/bin/vllm"
test -f "$DATA_ROOT/roles/$SOURCE/student_opd.jsonl"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo "Refusing to reuse student smoke path: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
PORT="$("$TRAIN_ENV/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
URL="http://127.0.0.1:$PORT"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SERVE_ARGS=(serve "$TEACHER" --served-model-name opd-math-teacher --port "$PORT" --max-model-len 4096 --gpu-memory-utilization 0.55)
if [[ "$TEACHER" == "Qwen/Qwen3-8B" ]]; then
  SERVE_ARGS+=(--revision "$TEACHER_REVISION")
fi
"$SERVE_ENV/bin/vllm" "${SERVE_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
READY=0
for _ in $(seq 1 600); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -100 "$SERVER_LOG"; exit 1; fi
  if curl -sf --max-time 2 "$URL/health" >/dev/null 2>&1; then READY=1; break; fi
  sleep 2
done
test "$READY" = 1
kill -0 "$SERVER_PID"
curl -sf --max-time 2 "$URL/health" >/dev/null
curl -sf --max-time 10 "$URL/v1/models" >"$OUT/server_models.json"

"$TRAIN_ENV/bin/python" "$REPO/scripts/opd_math/tokenizer_contract.py" \
  --teacher "$TEACHER" \
  --teacher-revision "$TEACHER_REVISION" \
  --student "$STUDENT" \
  --student-revision "$STUDENT_REVISION" \
  --server-url "$URL" \
  --server-model opd-math-teacher \
  --output "$OUT/tokenizer_contract.json" \
  --local-files-only
"$TRAIN_ENV/bin/python" "$REPO/scripts/opd_math/server_scoring_probe.py" \
  --tokenizer "$STUDENT" \
  --tokenizer-revision "$STUDENT_REVISION" \
  --server-url "$URL" \
  --server-model opd-math-teacher \
  --output "$OUT/server_scoring_contract.json" \
  --local-files-only

"$TRAIN_ENV/bin/python" "$REPO/scripts/opd/opd_train.py" \
  --mode task_rl_k1_gap \
  --pair-id "$PAIR" \
  --task-file "$DATA_ROOT/roles/$SOURCE/student_opd.jsonl" \
  --task-limit 1 \
  --budget-mode dose_response \
  --student "$STUDENT" \
  --student-revision "$STUDENT_REVISION" \
  --teacher-url "$URL" \
  --teacher-model opd-math-teacher \
  --teacher-checkpoint "$TEACHER" \
  --teacher-server-max-model-len 4096 \
  --prepared-manifest "$DATA_ROOT/prepared_manifest.json" \
  --tokenizer-contract "$OUT/tokenizer_contract.json" \
  --server-scoring-contract "$OUT/server_scoring_contract.json" \
  --out-dir "$OUT/student" \
  --steps 1 \
  --group-size 2 \
  --micro-prompts 1 \
  --max-new-tokens 64 \
  --top-k 0 \
  --allow-ungated-smoke \
  --require-parameter-update \
  --local-files-only
test -f "$OUT/student/final/adapter_config.json"
test -f "$OUT/student/traces/steps.jsonl"
echo "PASS task-reward plus sampled score-function OPD plumbing only: $OUT"
