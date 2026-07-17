#!/usr/bin/env bash
# Requires the judge_lane venv plus `pip install vllm`. This script does not
# install packages. It starts a local teacher server, runs three OPD steps, and
# verifies finite loss plus checkpoint creation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$ROOT/scripts/opd/_smoke_tmp"
TASK="$TMP/tasks.jsonl"
OUT="$TMP/out"
LOG="$TMP/opd_train.log"
SERVER_LOG="$TMP/vllm.log"
PORT="${OPD_SMOKE_PORT:-8000}"
MODE="${OPD_SMOKE_MODE:-opd_gated}"
URL="http://127.0.0.1:$PORT"
SERVER_PID=""

fail() {
  echo "FAIL $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "$TMP"
rm -rf "$OUT"
mkdir -p "$OUT"

python - "$ROOT" "$TASK" <<'PY'
import json, os, sys
root, out = sys.argv[1], sys.argv[2]
src = os.path.join(root, "scripts/judge_pilot/data/alloc_eval_pairs.jsonl")
prompts = []
if os.path.exists(src):
    with open(src) as f:
        for line in f:
            if len(prompts) >= 8:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("prompt_text") or row.get("question") or row.get("facts")
            if text:
                prompts.append(str(text)[:2000])
if not prompts:
    prompts = [
        "Decide whether retrieval is needed before answering: What rule controls eviction notice timing?",
        "Choose a retrieval effort for a bar-exam negligence question involving duty and breach.",
        "Should the agent retrieve statutes for a housing deposit dispute? Explain briefly.",
        "Pick whether k=0, k=3, or k=5 retrieval is appropriate for a contract formation question.",
        "Assess if conflicting retrieved passages require verification before final answer.",
        "Plan retrieval for a question about implied warranty in a residential lease.",
        "Decide whether a generated legal query should be rewritten before search.",
        "Select a cost-aware retrieval action for a multiple-choice evidence question.",
    ]
with open(out, "w") as f:
    for p in prompts[:8]:
        f.write(json.dumps({"prompt_text": p}) + "\n")
print(f"wrote {out} prompts={min(len(prompts), 8)}", flush=True)
PY

echo "starting vLLM teacher on $URL"
vllm serve Qwen/Qwen3-8B --port "$PORT" --max-model-len 2048 \
  --gpu-memory-utilization "${OPD_TEACHER_GPU_FRAC:-0.55}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 "${OPD_READY_TRIES:-600}"); do
  if curl -sf --max-time 2 "$URL/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2 || true
    fail "vLLM server exited before readiness"
  fi
  sleep 2
done
[[ "$READY" == "1" ]] || { tail -100 "$SERVER_LOG" >&2 || true; fail "vLLM readiness timed out"; }
echo "PASS teacher_ready"

if ! python "$ROOT/scripts/opd/opd_train.py" \
  --mode "$MODE" \
  --task-file "$TASK" \
  --student Qwen/Qwen3-1.7B \
  --teacher-url "$URL" \
  --teacher-model Qwen/Qwen3-8B \
  --out-dir "$OUT" \
  --steps 3 \
  --group-size 2 \
  --max-new-tokens 16 \
  --save-every 1 2>&1 | tee "$LOG"; then
  fail "opd_train.py failed"
fi

if grep -Eiq 'loss=(nan|inf|-inf)|non-finite' "$LOG"; then
  fail "non-finite loss detected"
fi
grep -q ' loss=' "$LOG" || fail "no loss lines found"
if [[ "$MODE" == "opd_gated" ]]; then
  grep -q 'gap_gate_mean=' "$LOG" || fail "gated mode did not log gap_gate_mean"
fi
find "$OUT" -maxdepth 1 -type d -name 'step_*' | grep -q . || fail "no checkpoint dir found"

echo "PASS finite_loss"
echo "PASS checkpoint_exists"
echo "PASS smoke_test"
