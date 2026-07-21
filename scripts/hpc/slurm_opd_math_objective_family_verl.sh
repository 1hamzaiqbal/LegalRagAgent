#!/bin/bash
#SBATCH --job-name=opd_obj_verl
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_objective_verl_%j.out

set -euo pipefail
: "${OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root}"
: "${OPD_MATH_STUDENT_SOURCE:?Set M or O}"
: "${OPD_MATH_SEED:?Set registered seed 0, 1, or 2}"
: "${OPD_MATH_CAMPAIGN_KIND:?Set diagnostic or scientific}"
: "${OPD_MATH_OBJECTIVE_OUT:?Set the fresh preregistered output directory}"
: "${OPD_MATH_STUDENT_SUPPORT_MANIFEST:?Set the same-commit support gate}"
: "${OPD_MATH_OBJECTIVE_PROMPT_PLAN:?Set the exact source/seed prompt plan}"
: "${OPD_MATH_OBJECTIVE_INITIALIZATION_MANIFEST:?Set the exact seed adapter manifest}"
: "${OPD_MATH_VERL_DATA_FILE:?Set the sealed materialized veRL JSONL}"
: "${OPD_MATH_VERL_DATA_MANIFEST:?Set its deterministic manifest}"
: "${OPD_MATH_TEACHER_CHECKPOINT:?Set the passing merged O teacher checkpoint}"
: "${OPD_MATH_TEACHER_GAP_MANIFEST:?Set the passing O teacher gap}"
: "${OPD_MATH_TEACHER_PROVENANCE_MANIFEST:?Set canonical O merge provenance}"

case "$OPD_MATH_STUDENT_SOURCE" in M|O) ;; *) echo "invalid student source" >&2; exit 2 ;; esac
case "$OPD_MATH_SEED" in 0|1|2) ;; *) echo "invalid seed" >&2; exit 2 ;; esac
case "$OPD_MATH_CAMPAIGN_KIND" in diagnostic|scientific) ;; *) echo "invalid campaign kind" >&2; exit 2 ;; esac
[[ "$OPD_MATH_OBJECTIVE_OUT" = /* ]] || { echo "objective output must be absolute" >&2; exit 2; }

SOURCE="$OPD_MATH_STUDENT_SOURCE"
SEED="$OPD_MATH_SEED"
OBJECTIVE_ID="k1_verl_upstream_clip10"
RUN_KEY="${OBJECTIVE_ID}__${SOURCE}__seed${SEED}"
RUN_ID="${OPD_MATH_OBJECTIVE_RUN_ID:-$RUN_KEY}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe run ID" >&2; exit 2; }
if [[ "$OPD_MATH_CAMPAIGN_KIND" == diagnostic ]]; then
  test "$SEED" = 0 || { echo "veRL diagnostic uses seed 0 only" >&2; exit 2; }
  STEPS=1
else
  STEPS=100
  : "${OPD_MATH_OBJECTIVE_PREREGISTRATION:?Scientific run requires preregistration}"
  : "${OPD_MATH_OBJECTIVE_LAUNCH_PLAN:?Scientific run requires launch plan}"
fi

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
VERL="${OPD_MATH_VERL_CHECKOUT:-/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/verl}"
VERL_ENV="${OPD_MATH_VERL_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_verl}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
OUT="$OPD_MATH_OBJECTIVE_OUT"
PRELAUNCH_RECEIPT="$OUT.prelaunch.json"
PREFLIGHT="$OUT.preflight.json"
RUN_RECEIPT="$OUT.receipt.json"
LAUNCHER="$REPO/scripts/hpc/slurm_opd_math_objective_family_verl.sh"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FREEZE="$RUN_ROOT/environment_freezes/$COMMIT/upstream_verl.freeze.txt"
VERIFY_ENVIRONMENT="$REPO/scripts/opd_math/verify_environment.py"
TASK="$OPD_MATH_DATA_ROOT/roles/$SOURCE/student_opd.jsonl"
PREPARED="$OPD_MATH_DATA_ROOT/prepared_manifest.json"
JOB_TMP="/engrfs/tmp/jacobsn/hiqbal_legalrag/verl_${SLURM_JOB_ID}"
RUN_LOG="$OUT/run.log"
CHECKPOINT_ROOT="$OUT/checkpoints"

for artifact in "$OUT" "$PRELAUNCH_RECEIPT" "$PREFLIGHT" "$RUN_RECEIPT"; do
  if [[ -e "$artifact" || -L "$artifact" ]]; then
    echo "Refusing to reuse veRL artifact path: $artifact" >&2
    exit 2
  fi
done
test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$VERL" rev-parse HEAD)" = "6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"
test -z "$(git -C "$VERL" status --porcelain=v1 --untracked-files=no)"
test -x "$VERL_ENV/bin/python"
test -f "$FREEZE"
INIT_ADAPTER="$($VERL_ENV/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter_path"])' "$OPD_MATH_OBJECTIVE_INITIALIZATION_MANIFEST")"
test -f "$TASK"
test -f "$PREPARED"
test -f "$OPD_MATH_VERL_DATA_FILE"
test -f "$OPD_MATH_VERL_DATA_MANIFEST"
test -f "$OPD_MATH_STUDENT_SUPPORT_MANIFEST"
test -f "$OPD_MATH_OBJECTIVE_PROMPT_PLAN"
test -f "$OPD_MATH_OBJECTIVE_INITIALIZATION_MANIFEST"
test -d "$INIT_ADAPTER"
test -d "$OPD_MATH_TEACHER_CHECKPOINT"
test -f "$OPD_MATH_TEACHER_GAP_MANIFEST"
test "$OPD_MATH_TEACHER_PROVENANCE_MANIFEST" = "$OPD_MATH_TEACHER_CHECKPOINT/merge_provenance.json"

"$VERL_ENV/bin/python" "$VERIFY_ENVIRONMENT" \
  --environment-root "$VERL_ENV" \
  --commit-freeze "$FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind upstream_verl

if [[ "$OPD_MATH_CAMPAIGN_KIND" == scientific ]]; then
  "$VERL_ENV/bin/python" "$REPO/scripts/opd/objective_family_preregistration.py" prelaunch \
    --preregistration "$OPD_MATH_OBJECTIVE_PREREGISTRATION" \
    --launch-plan "$OPD_MATH_OBJECTIVE_LAUNCH_PLAN" \
    --run-key "$RUN_KEY" \
    --run-id "$RUN_ID" \
    --scheduler-job-id "$SLURM_JOB_ID" \
    --output "$PRELAUNCH_RECEIPT"
fi

PREFLIGHT_ARGS=(
  preflight
  --campaign-kind "$OPD_MATH_CAMPAIGN_KIND"
  --source "$SOURCE"
  --seed "$SEED"
  --run-id "$RUN_ID"
  --scheduler-job-id "$SLURM_JOB_ID"
  --task-file "$TASK"
  --prepared-manifest "$PREPARED"
  --prompt-plan "$OPD_MATH_OBJECTIVE_PROMPT_PLAN"
  --initialization-manifest "$OPD_MATH_OBJECTIVE_INITIALIZATION_MANIFEST"
  --data-file "$OPD_MATH_VERL_DATA_FILE"
  --data-manifest "$OPD_MATH_VERL_DATA_MANIFEST"
  --student-support-manifest "$OPD_MATH_STUDENT_SUPPORT_MANIFEST"
  --teacher-checkpoint "$OPD_MATH_TEACHER_CHECKPOINT"
  --teacher-gap-manifest "$OPD_MATH_TEACHER_GAP_MANIFEST"
  --teacher-provenance-manifest "$OPD_MATH_TEACHER_PROVENANCE_MANIFEST"
  --verl-checkout "$VERL"
  --environment-root "$VERL_ENV"
  --environment-freeze "$FREEZE"
  --launcher "$LAUNCHER"
  --output-root "$OUT"
  --output "$PREFLIGHT"
)
if [[ "$OPD_MATH_CAMPAIGN_KIND" == scientific ]]; then
  PREFLIGHT_ARGS+=(--prelaunch-receipt "$PRELAUNCH_RECEIPT")
fi
"$VERL_ENV/bin/python" "$REPO/scripts/opd/verl_run_custody.py" "${PREFLIGHT_ARGS[@]}"

mkdir -p "$OUT" "$JOB_TMP" "$OUT/rollouts"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR="$JOB_TMP/ray"
export TMPDIR="$JOB_TMP/tmp"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

STUDENT_SNAPSHOT="$($VERL_ENV/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', local_files_only=True))
PY
)"

cd "$VERL"
COMMAND=(
  "$VERL_ENV/bin/python" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="['$OPD_MATH_VERL_DATA_FILE']"
  data.val_files="['$OPD_MATH_VERL_DATA_FILE']"
  data.train_batch_size=1
  data.max_prompt_length=1536
  data.max_response_length=512
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=False
  data.seed="$SEED"
  +data.apply_chat_template_kwargs.enable_thinking=False
  actor_rollout_ref.model.path="$STUDENT_SNAPSHOT"
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=64
  actor_rollout_ref.model.target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
  actor_rollout_ref.model.lora_adapter_path="$INIT_ADAPTER"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-5
  actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.optim.betas="[0.9,0.999]"
  actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.ppo_mini_batch_size=1
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.shuffle=False
  actor_rollout_ref.actor.data_loader_seed="$SEED"
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.clip_ratio=0.2
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.2
  actor_rollout_ref.actor.clip_ratio_c=3.0
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.use_torch_compile=True
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.n=4
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.seed="$SEED"
  actor_rollout_ref.rollout.max_model_len=2049
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  trainer.balance_batch=False
  trainer.logger="['console']"
  trainer.project_name=legalrag_opd_math
  trainer.experiment_name="$RUN_ID"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.val_before_train=False
  trainer.save_freq="$STEPS"
  trainer.test_freq=-1
  trainer.total_epochs=1
  trainer.total_training_steps="$STEPS"
  trainer.default_local_dir="$CHECKPOINT_ROOT"
  trainer.rollout_data_dir="$OUT/rollouts"
  reward.custom_reward_function.path="$REPO/scripts/opd/verl_zero_reward.py"
  reward.custom_reward_function.name=compute_score
  distillation.enabled=True
  distillation.n_gpus_per_node=1
  distillation.nnodes=1
  distillation.teacher_models.teacher_model.model_path="$OPD_MATH_TEACHER_CHECKPOINT"
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1
  distillation.teacher_models.teacher_model.inference.name=vllm
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.55
  distillation.teacher_models.teacher_model.inference.max_model_len=2049
  distillation.distillation_loss.loss_mode=k1
  distillation.distillation_loss.use_task_rewards=False
  distillation.distillation_loss.use_policy_gradient=True
  distillation.distillation_loss.policy_loss_mode=vanilla
  distillation.distillation_loss.loss_max_clamp=10.0
  distillation.distillation_loss.log_prob_min_clamp=-10.0
)

set +e
"${COMMAND[@]}" >"$RUN_LOG" 2>&1
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  tail -200 "$RUN_LOG"
  exit "$STATUS"
fi

ACTOR_CHECKPOINT="$CHECKPOINT_ROOT/global_step_${STEPS}/actor"
test -d "$ACTOR_CHECKPOINT"
"$VERL_ENV/bin/python" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUT/final_candidate" >>"$RUN_LOG" 2>&1
test -d "$OUT/final_candidate/lora_adapter"
mv "$OUT/final_candidate/lora_adapter" "$OUT/final"

"$VERL_ENV/bin/python" "$VERIFY_ENVIRONMENT" \
  --environment-root "$VERL_ENV" \
  --commit-freeze "$FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind upstream_verl

"$VERL_ENV/bin/python" "$REPO/scripts/opd/verl_run_custody.py" audit \
  --preflight "$PREFLIGHT" \
  --run-log "$RUN_LOG" \
  --actor-checkpoint "$ACTOR_CHECKPOINT" \
  --rollout-dir "$OUT/rollouts" \
  --final-adapter "$OUT/final" \
  --output "$RUN_RECEIPT"

echo "Pinned veRL objective-family run completed; held-out release remains campaign-wide: $OUT"
