from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EVALUATE = ROOT / "scripts" / "hpc" / "slurm_opd_math_evaluate.sh"
MERGE = ROOT / "scripts" / "hpc" / "slurm_opd_math_merge_evaluation.sh"


def test_evaluation_wrappers_have_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(EVALUATE), str(MERGE)], check=True)


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


def test_scientific_wrappers_require_an_explicit_canonical_data_root():
    names = (
        "slurm_opd_math_evaluate.sh",
        "slurm_opd_math_merge_evaluation.sh",
        "slurm_opd_math_quality_gate.sh",
        "slurm_opd_math_teacher_train.sh",
        "slurm_opd_math_student_train.sh",
        "slurm_opd_math_student_results.sh",
    )
    for name in names:
        script = (ROOT / "scripts" / "hpc" / name).read_text()
        assert "${OPD_MATH_DATA_ROOT:?" in script
        assert "data/legalrag/opd_math/v1}" not in script


def test_quality_gate_has_headroom_for_full_o_reward_recomputation():
    script = (ROOT / "scripts" / "hpc" / "slurm_opd_math_quality_gate.sh").read_text()
    assert "#SBATCH --time=04:00:00" in script


def test_student_checkpointing_is_explicit_not_ambient():
    trainer = (ROOT / "scripts" / "opd" / "opd_train.py").read_text()
    wrapper = (ROOT / "scripts" / "hpc" / "slurm_opd_math_student_train.sh").read_text()
    assert "OPD_GRAD_CKPT" not in trainer
    assert 'if args.gradient_checkpointing:' in trainer
    assert '"gradient_checkpointing": args.gradient_checkpointing' in trainer
    assert "--gradient-checkpointing" in wrapper
