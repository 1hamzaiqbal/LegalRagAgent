#!/bin/bash
#SBATCH --job-name=opsd_pc_baseeval
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-sxm4:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opsd_pc_baseeval_%j.out

set -euo pipefail
REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
UPSTREAM="${OPD_UPSTREAM_REPO:-/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/OPSD}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
DATA_ROOT="${OPD_IDENT_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
HF_HOME="${OPD_IDENT_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
EXPECTED_COMMIT="${OPD_IDENT_EXPECTED_COMMIT:?set OPD_IDENT_EXPECTED_COMMIT at submission}"
PREFLIGHT_JOB_ID="${OPD_IDENT_PREFLIGHT_JOB_ID:?set OPD_IDENT_PREFLIGHT_JOB_ID at submission}"

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "${SLURM_JOB_NUM_NODES:?missing Slurm node count}" = "1"
PREFLIGHT="$RUN_ROOT/preflight/job_${PREFLIGHT_JOB_ID}/receipt.json"
test -f "$PREFLIGHT"
"$ENV_DIR/bin/python" - "$PREFLIGHT" "$EXPECTED_COMMIT" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload["status"] == "passed"
assert payload["repository_commit"] == sys.argv[2]
PY
"$ENV_DIR/bin/python" - <<'PY'
import torch
if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected exactly four visible CUDA devices, got {torch.cuda.device_count()}")
PY

OUT="$RUN_ROOT/base_eval/job_${SLURM_JOB_ID}"
test ! -e "$OUT"
mkdir -p "$OUT"
"$ENV_DIR/bin/python" "$REPO/scripts/opd/prepare_opsd_execution_tree.py" \
  --upstream "$UPSTREAM" \
  --target "$OUT/execution_tree" \
  --repository-commit "$EXPECTED_COMMIT"

MODEL="$($ENV_DIR/bin/python - "$HF_HOME" <<'PY'
import os, sys
os.environ["HF_HOME"] = sys.argv[1]
os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", local_files_only=True))
PY
)"

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export LEGALRAG_OPSD_TRAIN_PARQUET="$DATA_ROOT/opsd_train/**/*.parquet"
export LEGALRAG_OPSD_AIME24_PARQUET="$DATA_ROOT/aime24/**/*.parquet"
export NCCL_P2P_DISABLE=1

cd "$OUT/execution_tree/eval"
"$ENV_DIR/bin/python" evaluate_math.py \
  --base_model "$MODEL" \
  --dataset aime24 \
  --val_n 12 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k -1 \
  --min_p 0.0 \
  --presence_penalty 0.0 \
  --max_new_tokens 38912 \
  --tensor_parallel_size 4 \
  --output_file "$OUT/base_aime24_avg12.json"

"$ENV_DIR/bin/python" "$REPO/scripts/opd/positive_control_gate.py" \
  --eval-json "$OUT/base_aime24_avg12.json" \
  --config "$REPO/configs/opd_math/identifiability_v1.json" \
  --output "$OUT/base_gate.json" \
  --repository-commit "$EXPECTED_COMMIT"

echo "PASS upstream base-model reproduction gate: $OUT/base_gate.json"
