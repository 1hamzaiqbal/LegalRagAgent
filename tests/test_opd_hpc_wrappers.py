from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EVALUATE = ROOT / "scripts" / "hpc" / "slurm_opd_math_evaluate.sh"
MERGE = ROOT / "scripts" / "hpc" / "slurm_opd_math_merge_evaluation.sh"
QUALITY_GATE = ROOT / "scripts" / "hpc" / "slurm_opd_math_quality_gate.sh"
TEACHER_TRAIN = ROOT / "scripts" / "hpc" / "slurm_opd_math_teacher_train.sh"
TEACHER_SMOKE = ROOT / "scripts" / "hpc" / "slurm_opd_math_teacher_smoke.sh"
STUDENT_RESULTS = ROOT / "scripts" / "hpc" / "slurm_opd_math_student_results.sh"
STUDENT_TRAIN = ROOT / "scripts" / "hpc" / "slurm_opd_math_student_train.sh"
OBJECTIVE_FAMILY_TRAIN = (
    ROOT / "scripts" / "hpc" / "slurm_opd_math_objective_family_train.sh"
)
OBJECTIVE_FAMILY_VERL = (
    ROOT / "scripts" / "hpc" / "slurm_opd_math_objective_family_verl.sh"
)
VERL_SETUP = ROOT / "scripts" / "hpc" / "setup_opd_math_verl_env.sh"
VERL_PREFLIGHT = ROOT / "scripts" / "hpc" / "slurm_opd_math_verl_preflight.sh"
DEEPMATH_DOWNLOAD = ROOT / "scripts" / "hpc" / "slurm_opd_math_deepmath_download.sh"
DEEPMATH_INVENTORY = ROOT / "scripts" / "hpc" / "slurm_opd_math_deepmath_inventory.sh"
DEEPMATH_AUDIT = ROOT / "scripts" / "hpc" / "slurm_opd_math_deepmath_audit.sh"
DEEPMATH_FINALIZE = ROOT / "scripts" / "hpc" / "slurm_opd_math_deepmath_finalize.sh"


def test_touched_evaluation_and_teacher_wrappers_have_valid_bash_syntax():
    subprocess.run(
        [
            "bash",
            "-n",
            str(EVALUATE),
            str(MERGE),
            str(QUALITY_GATE),
            str(TEACHER_TRAIN),
            str(TEACHER_SMOKE),
            str(STUDENT_RESULTS),
            str(STUDENT_TRAIN),
            str(OBJECTIVE_FAMILY_TRAIN),
            str(OBJECTIVE_FAMILY_VERL),
            str(VERL_SETUP),
            str(VERL_PREFLIGHT),
            str(DEEPMATH_DOWNLOAD),
            str(DEEPMATH_INVENTORY),
            str(DEEPMATH_AUDIT),
            str(DEEPMATH_FINALIZE),
        ],
        check=True,
    )


def test_deepmath_download_is_revision_pinned_restartable_and_non_authorizing():
    script = DEEPMATH_DOWNLOAD.read_text()
    assert "#SBATCH --partition=general-cpu" in script
    assert "deepmath_qualification_plan.json" in script
    assert "5cf055d1fe3d7a2eb19719ac020211469736ae44" in script
    assert 'test -z "$(git -C "$REPO" status --porcelain=v1)"' in script
    assert "--continue-at -" in script
    assert 'sha256sum "$partial"' in script
    assert 'mv "$partial" "$target"' in script
    assert "verify-raw" in script
    assert '"teacher_training_authorized": False' in script
    assert '"scientific_use_allowed": False' in script


def test_deepmath_inventory_uses_persistent_cache_and_never_authorizes_training():
    script = DEEPMATH_INVENTORY.read_text()
    assert "#SBATCH --partition=general-cpu" in script
    assert "deepmath_inventory_plan.json" in script
    assert "materialize_deepmath_inventory.py" in script
    assert 'test -z "$(git -C "$REPO" status --porcelain=v1)"' in script
    assert "/engrfs/project/jacobsn/hiqbal/cache/" in script
    assert "HF_HUB_CACHE" in script
    assert "HF_DATASETS_CACHE" in script
    assert "OPD_INVENTORY_LAUNCHER_PATH" in script
    assert "collision and training authorization remain closed" in script


def test_deepmath_audit_is_offline_high_memory_and_requires_new_output_root():
    script = DEEPMATH_AUDIT.read_text()
    assert "#SBATCH --mem=192G" in script
    assert "#SBATCH --time=24:00:00" in script
    assert "audit_deepmath_inventory.py" in script
    assert "OPD_DEEPMATH_AUDIT_ROOT:?Set a new immutable" in script
    assert 'test ! -e "$OUTPUT_DIR"' in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "TRANSFORMERS_OFFLINE=1" in script
    assert "OPD_DEEPMATH_AUDIT_LAUNCHER_PATH" in script
    assert "teacher training remains unauthorized" in script


def test_deepmath_finalize_requires_bound_scan_decisions_and_new_root():
    script = DEEPMATH_FINALIZE.read_text()
    assert "#SBATCH --mem=192G" in script
    assert "finalize_deepmath_audit.py" in script
    assert "OPD_DEEPMATH_INVENTORY_ROOT:?Bind the immutable" in script
    assert "OPD_DEEPMATH_AUDIT_ROOT:?Bind the immutable" in script
    assert "OPD_DEEPMATH_REVIEW_DECISIONS:?Bind the complete" in script
    assert "OPD_DEEPMATH_FINAL_ROOT:?Set a new immutable" in script
    assert 'test ! -e "$OUTPUT_DIR"' in script
    assert 'test -z "$(git -C "$REPO" status --porcelain=v1)"' in script
    assert "teacher training remains unauthorized" in script


def test_array_wrapper_keeps_global_budget_and_stable_shard_identity():
    script = EVALUATE.read_text()
    assert 'SHARD_COUNT="${OPD_MATH_EVAL_SHARDS:-1}"' in script
    assert 'SLURM_ARRAY_TASK_ID:-${OPD_MATH_EVAL_SHARD_INDEX:-0}' in script
    assert "sharded evaluation requires SLURM_ARRAY_TASK_ID" in script
    assert "explicit shard index conflicts with SLURM_ARRAY_TASK_ID" in script
    assert '--max-records "$OPD_MATH_EVAL_MAX_RECORDS"' in script
    assert '--shard-count "$SHARD_COUNT"' in script
    assert '--shard-index "$SHARD_INDEX"' in script
    assert 'shards/$SHARD_NAME' in script
    assert 'OPD_MATH_EVAL_RUN_ID' in script
    assert 'run_${SLURM_JOB_ID}' not in script
    assert 'OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root' in script


def test_merge_wrapper_uses_cpu_and_exact_shard_count():
    script = MERGE.read_text()
    assert "#SBATCH --partition=general-cpu" in script
    assert '--shard-root "$SHARD_ROOT"' in script
    assert '--shard-count "$OPD_MATH_EVAL_SHARDS"' in script
    assert '--task-file "$TASK"' in script
    assert '--output-dir "$OUT"' in script
    assert 'OPD_MATH_EVAL_MAX_RECORDS:?Set the exact shard-job record budget' in script


def test_full_o_base_and_trained_evaluations_consume_the_fixed_plan():
    evaluate = EVALUATE.read_text()
    merge = MERGE.read_text()
    for script, phase in ((evaluate, "shard"), (merge, "merge")):
        assert "plan_evaluation_shards.py" in script
        assert "OPD_MATH_EVAL_SHARD_PLAN" in script
        assert "OPD_MATH_EVAL_PLAN_ARM" in script
        assert "OPD_MATH_EVAL_ARRAY_SPEC" in script
        assert 'TASK_REL" == roles/O/teacher_gap_dev.jsonl' in script
        assert "OPD_MATH_EVAL_MAX_RECORDS >= TASK_RECORDS" in script
        assert "must be a canonical nonnegative integer" in script
        assert 'validate-launch' in script
        assert f'--phase {phase}' in script
        assert '--source O' in script
        assert '--role teacher_gap_dev' in script
        assert '--shard-count' in script
        assert '--git-commit "$COMMIT"' in script
        assert '--train-freeze "$TRAIN_FREEZE"' in script
        assert '--array-spec "$OPD_MATH_EVAL_ARRAY_SPEC"' in script
        assert '--samples-per-problem "$EVAL_SAMPLES_PER_PROBLEM"' in script
        assert '--temperature "$EVAL_TEMPERATURE"' in script
        assert '--top-p "$EVAL_TOP_P"' in script
        assert '--top-k "$EVAL_TOP_K"' in script
        assert '--max-new-tokens "$EVAL_MAX_NEW_TOKENS"' in script
        assert '--seed "$EVAL_SEED"' in script
    assert '--array-task-count "$SLURM_ARRAY_TASK_COUNT"' in evaluate
    assert '--array-task-min "$SLURM_ARRAY_TASK_MIN"' in evaluate
    assert '--array-task-max "$SLURM_ARRAY_TASK_MAX"' in evaluate


def test_evaluation_wrappers_bind_exact_environment_and_require_commit_companion():
    evaluate = EVALUATE.read_text()
    merge = MERGE.read_text()
    for script in (evaluate, merge):
        assert 'FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"' in script
        assert 'TRAIN_FREEZE="$FREEZE_ROOT/train.freeze.txt"' in script
        assert 'test -z "$(git -C "$REPO" status --porcelain=v1)"' in script
        assert 'test ! -L "$VERIFY_ENVIRONMENT"' in script
        assert 'sha256sum "$VERIFY_ENVIRONMENT"' in script
        assert script.count('"$ENV_DIR/bin/python" "$VERIFY_ENVIRONMENT"') == 1
        assert '--train-environment-root "$ENV_DIR"' in script
        assert '--train-environment-freeze "$TRAIN_FREEZE"' in script
        assert 'test -f "$OUT.custody.json"' in script
    assert '"$ENV_DIR/bin/python" "$REPO/scripts/opd_math/evaluate_math.py"' in evaluate
    assert '\npython "$REPO/scripts/opd_math/evaluate_math.py"' not in evaluate


def test_scientific_wrappers_require_an_explicit_canonical_data_root():
    names = (
        "slurm_opd_math_evaluate.sh",
        "slurm_opd_math_merge_evaluation.sh",
        "slurm_opd_math_quality_gate.sh",
        "slurm_opd_math_teacher_train.sh",
        "slurm_opd_math_student_train.sh",
        "slurm_opd_math_student_results.sh",
        "slurm_opd_math_objective_family_train.sh",
    )
    for name in names:
        script = (ROOT / "scripts" / "hpc" / name).read_text()
        assert "${OPD_MATH_DATA_ROOT:?" in script
        assert "data/legalrag/opd_math/v1}" not in script


def test_objective_family_wrapper_is_fixed_to_o_teacher_and_exact_inputs():
    script = OBJECTIVE_FAMILY_TRAIN.read_text()
    assert "OPD_MATH_CAMPAIGN_KIND:?Set diagnostic or scientific" in script
    assert "one-step fidelity diagnostics use seed 0 only" in script
    assert "k1_verl_upstream_clip10" in script
    assert "upstream veRL has a separate pinned launcher" in script
    assert "--objective-family-prompt-plan" in script
    assert "--objective-family-initialization-manifest" in script
    assert "--objective-family-launcher" in script
    assert 'STEPS=1' in script and 'STEPS=100' in script
    assert "Teacher-scored objective requires passing O checkpoint" in script
    assert "OPD_MATH_TEACHER_SOURCE" not in script
    assert "M_M" not in script and "M_O" not in script
    assert 'objective_family_preregistration.py" prelaunch' not in script
    assert '"$PREREG_TOOL" prelaunch' in script


def test_upstream_verl_wrapper_binds_and_reverifies_exact_environment():
    script = OBJECTIVE_FAMILY_VERL.read_text()
    marker = '"${COMMAND[@]}" >"$RUN_LOG" 2>&1'
    assert 'FREEZE="$RUN_ROOT/environment_freezes/$COMMIT/upstream_verl.freeze.txt"' in script
    assert script.count('"$VERL_ENV/bin/python" "$VERIFY_ENVIRONMENT"') == 2
    assert script.count("--freeze-kind upstream_verl") == 2
    assert script.index("--freeze-kind upstream_verl") < script.index(marker)
    assert script.rindex("--freeze-kind upstream_verl") > script.index(marker)


def test_upstream_verl_environment_setup_is_fresh_pinned_and_exactly_frozen():
    script = VERL_SETUP.read_text()
    assert 'PINNED_VERL_COMMIT="6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"' in script
    assert '"vllm==0.12.0"' in script
    assert '"$VERL"' in script
    assert "Refusing to alter existing pinned-veRL environment" in script
    assert 'upstream_verl.freeze.txt' in script
    assert "importlib.metadata" in script
    assert "--freeze-kind upstream_verl" in script


def test_upstream_verl_gpu_preflight_requires_two_gpus_and_cached_student():
    script = VERL_PREFLIGHT.read_text()
    assert "#SBATCH --gpus=a100-sxm4:2" in script
    assert "torch.cuda.device_count() != 2" in script
    assert 'revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"' in script
    assert "local_files_only=True" in script
    assert "--freeze-kind upstream_verl" in script


def test_student_result_wrapper_exposes_o_teacher_four_arm_readout():
    script = STUDENT_RESULTS.read_text()
    assert 'OPD_MATH_RESULT_KIND" == o_teacher' in script
    assert 'student_results.py" o-teacher-readout' in script
    assert '--baseline-m "$OPD_MATH_RESULT_BASELINE_M"' in script
    assert '--o-m "$OPD_MATH_RESULT_O_M"' in script
    assert '--baseline-o "$OPD_MATH_RESULT_BASELINE_O"' in script
    assert '--o-o "$OPD_MATH_RESULT_O_O"' in script
    assert '--preregistration "$OPD_MATH_RESULT_PREREGISTRATION"' in script
    assert '--launch-ledger "$OPD_MATH_RESULT_LAUNCH_LEDGER"' in script
    assert "OPD_MATH_RESULT_OUTPUT_MANIFEST" in script
    assert '--output-manifest "$OPD_MATH_RESULT_OUTPUT_MANIFEST"' in script
    o_teacher_branch = script.split(
        'elif [[ "$OPD_MATH_RESULT_KIND" == o_teacher ]]', 1
    )[1].split('elif [[ "$OPD_MATH_RESULT_KIND" == matrix ]]', 1)[0]
    assert "OPD_MATH_RESULT_M_M" not in o_teacher_branch
    assert "OPD_MATH_RESULT_M_O" not in o_teacher_branch


def test_primary_student_runs_use_preregisterable_stable_paths():
    script = STUDENT_TRAIN.read_text()
    assert "Primary matched runs require a preregistered stable run ID" in script
    assert 'RUN_ID="$OPD_MATH_STUDENT_RUN_ID"' in script
    assert 'OUT="$RUN_ROOT/students/$RUN_KEY/$MODE/run_$RUN_ID"' in script
    assert '--campaign-run-id "$RUN_ID"' in script
    assert '--scheduler-job-id "$SLURM_JOB_ID"' in script
    assert "Primary matched runs require the sealed preregistration" in script
    assert "Primary matched runs require the sealed launch ledger" in script
    assert 'PRELAUNCH_RECEIPT="$OUT.prelaunch.json"' in script
    assert 'student_results.py" "${PRELAUNCH_ARGS[@]}"' in script
    assert '--prelaunch-receipt "$PRELAUNCH_RECEIPT"' in script
    assert '--preregistration "$OPD_MATH_STUDENT_PREREGISTRATION"' in script
    assert '--launch-ledger "$OPD_MATH_STUDENT_LAUNCH_LEDGER"' in script
    assert '[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]' in script
    primary_branch = script.split(
        'if [[ "$OPD_MATH_BUDGET_MODE" == primary_matched ]]', 1
    )[1].split("else", 1)[0]
    assert "SLURM_JOB_ID" not in primary_branch
    assert 'slurm_${SLURM_JOB_ID}' in script


def test_teacher_wrapper_uses_the_registered_full_pool_prompt_bound():
    script = TEACHER_TRAIN.read_text()
    assert 'OPD_MATH_MAX_PROMPT_TOKENS:-2304' in script
    assert 'OPD_MATH_MAX_PROMPT_TOKENS:-1536' not in script
    smoke = TEACHER_SMOKE.read_text()
    assert "--max-prompt-tokens 2304" in smoke
    assert "--max-prompt-tokens 1536" not in smoke
    assert 'OPD_MATH_TEACHER_SMOKE_LIMIT:-16' in smoke


def test_teacher_wrappers_bind_and_reverify_the_commit_specific_train_environment():
    for path in (TEACHER_TRAIN, TEACHER_SMOKE):
        script = path.read_text()
        assert 'FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"' in script
        assert 'TRAIN_FREEZE="$FREEZE_ROOT/train.freeze.txt"' in script
        assert script.count('"$ENV_DIR/bin/python" "$VERIFY_ENVIRONMENT"') == 2
        assert '--train-environment-root "$ENV_DIR"' in script
        assert '--train-environment-freeze "$TRAIN_FREEZE"' in script
        before = script.index("Verifying live train environment before")
        trainer = script.index('python "$REPO/scripts/opd_math/train_teacher_grpo.py"')
        after = script.index("Re-verifying live train environment after")
        assert before < trainer < after


def test_quality_gate_has_headroom_for_full_o_reward_recomputation():
    script = QUALITY_GATE.read_text()
    assert "#SBATCH --time=04:00:00" in script


def test_quality_wrapper_exposes_only_non_authorizing_cross_source_report():
    script = QUALITY_GATE.read_text()
    assert '[[ "$OPD_MATH_GATE_KIND" == teacher_target_report ]]' in script
    assert "quality_gates.py\" teacher-target-report" in script
    assert '--teacher-source "$OPD_MATH_GATE_TEACHER_SOURCE"' in script
    assert '--target-source "$OPD_MATH_GATE_SOURCE"' in script
    assert "teacher_target_report has no smoke mode" in script
    assert "teacher_target_report requires distinct teacher and target sources" in script
    assert "PASS non-authorizing teacher-target report completed" in script
    # Preserve the established gate handoff string for teacher/support jobs.
    assert (
        "PASS gate computation completed; inspect passed/strength before use"
        in script
    )


def test_student_checkpointing_is_explicit_not_ambient():
    trainer = (ROOT / "scripts" / "opd" / "opd_train.py").read_text()
    wrapper = (ROOT / "scripts" / "hpc" / "slurm_opd_math_student_train.sh").read_text()
    assert "OPD_GRAD_CKPT" not in trainer
    assert 'if args.gradient_checkpointing:' in trainer
    assert '"gradient_checkpointing": args.gradient_checkpointing' in trainer
    assert "--gradient-checkpointing" in wrapper
