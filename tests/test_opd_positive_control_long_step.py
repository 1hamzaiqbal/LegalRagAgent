from __future__ import annotations

import json
from pathlib import Path

from scripts.opd import positive_control_long_step as long_step
from scripts.opd import positive_control_one_step as one_step
from scripts.opd import prepare_opsd_execution_tree as execution_tree


def test_long_step_is_length_qualified_and_keeps_effective_batch() -> None:
    root = Path(__file__).resolve().parents[1]
    short = json.loads(
        (root / "configs/opd_math/identifiability_v1_one_step_retry4.json").read_text()
    )
    long = json.loads(
        (
            root
            / "configs/opd_math/identifiability_v1_one_step_long4096.json"
        ).read_text()
    )
    assert long["status"] == (
        "preregistered_diagnostic_only_100_step_training_blocked"
    )
    assert long["release_prerequisite"]["calibrated_student_cap"] == 4096
    assert long["release_prerequisite"]["calibration_samples_at_4096"] == 128
    assert long["release_prerequisite"]["calibration_at_cap_at_4096"] == 0
    for key in ("upstream", "training_data", "runtime_cache_policy"):
        assert long[key] == short[key]
    changed = {
        key
        for key in short["recipe"]
        if short["recipe"][key] != long["recipe"][key]
    }
    assert changed == long_step.ALLOWED_RECIPE_CHANGES
    assert long["recipe"]["max_completion_tokens"] == 4096
    assert long["recipe"]["per_device_train_batch_size"] == 1
    assert long["recipe"]["gradient_accumulation_steps"] == 8
    assert long["recipe"]["effective_batch_size"] == 32
    assert long["hardware"]["gpu_type"] == "a100-sxm4"
    assert long["hardware"]["minimum_vram_mib_per_gpu"] == 81000
    assert long["pass_gate"]["expected_trajectory_count"] == 32
    assert long["pass_gate"]["maximum_at_cap_trajectory_count"] == 1


def test_long_step_command_uses_4096_without_changing_objective(tmp_path: Path) -> None:
    command = one_step.training_command(
        tmp_path / "env",
        tmp_path / "execution",
        tmp_path / "model",
        tmp_path / "output",
        12345,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        run_config=long_step.RUN_CONFIG,
        max_completion_tokens=4096,
    )
    assert command[command.index("--max_completion_length") + 1] == "4096"
    assert command[command.index("--per_device_train_batch_size") + 1] == "1"
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--gradient_accumulation_steps"
    ] == ["8", "8"]
    assert command[command.index("--beta") + 1] == "0"
    assert command[command.index("--jsd_token_clip") + 1] == "0.05"
    assert "--fixed_teacher" in command
    assert "--top_k_loss" not in command


def test_execution_patch_uses_dynamic_padding_and_rank_safe_trajectory_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opsd_trainer.py"
    path.write_text(
        "\n".join(
            (
                execution_tree.DYNAMIC_PAD_OLD,
                execution_tree.BUFFER_INIT_OLD,
                execution_tree.SAVE_PATH_OLD,
                execution_tree.BUFFER_APPEND_OLD,
            )
        )
    )
    record = execution_tree.replace_many(
        path,
        [
            (execution_tree.DYNAMIC_PAD_OLD, execution_tree.DYNAMIC_PAD_NEW),
            (execution_tree.BUFFER_INIT_OLD, execution_tree.BUFFER_INIT_NEW),
            (execution_tree.SAVE_PATH_OLD, execution_tree.SAVE_PATH_NEW),
            (execution_tree.BUFFER_APPEND_OLD, execution_tree.BUFFER_APPEND_NEW),
        ],
    )
    source = path.read_text()
    assert record["replacement_count"] == 4
    assert "batch_completion_length = max(1, max(observed_completion_lengths))" in source
    assert "generation_ids" not in execution_tree.DYNAMIC_PAD_NEW
    assert 'open(output_file, "x"' in source
    assert "generations_step_{step}_rank_{rank}.json" in source
    assert '"completion_tokens"' in source
    assert '"at_cap"' in source


def test_long_step_trajectory_audit_reconstructs_every_rank(tmp_path: Path) -> None:
    generation_dir = tmp_path / "generations"
    generation_dir.mkdir()
    for rank in range(4):
        rows = []
        for sequence in range(8):
            length = 4096 if rank == 0 and sequence == 0 else 100 + sequence
            rows.append(
                {
                    "step": 0,
                    "rank": rank,
                    "local_sequence": sequence,
                    "prompt": f"prompt-{rank}-{sequence}",
                    "completion": f"completion-{rank}-{sequence}",
                    "completion_tokens": length,
                    "at_cap": length >= 4096,
                    "max_completion_length": 4096,
                }
            )
        (generation_dir / f"generations_step_1_final_rank_{rank}.json").write_text(
            json.dumps(
                {
                    "step": "1_final",
                    "rank": rank,
                    "num_samples": len(rows),
                    "generations": rows,
                }
            )
        )
    config = {
        "recipe": {"max_completion_tokens": 4096},
        "pass_gate": {
            "expected_trajectory_count": 32,
            "maximum_at_cap_trajectory_count": 1,
            "maximum_at_cap_fraction": 0.05,
        },
    }
    audit = long_step.audit_trajectories(tmp_path, config)
    assert audit["trajectory_count"] == 32
    assert audit["at_cap_trajectory_count"] == 1
    assert audit["at_cap_fraction"] == 1 / 32


def test_long_step_job_is_four_a100_80gb_and_does_not_autoqueue() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (
        root / "scripts/hpc/slurm_opd_positive_control_long_step.sh"
    ).read_text()
    submit = (
        root / "scripts/hpc/submit_opd_positive_control_long_step.sh"
    ).read_text()
    assert "#SBATCH --nodes=1" in wrapper
    assert "#SBATCH --ntasks=1" in wrapper
    assert "#SBATCH --gpus=a100-sxm4:4" in wrapper
    assert "A100-SXM4-80GB" in wrapper
    assert "memory_mib < 81000" in wrapper
    assert "identifiability_v1_one_step_long4096.json" in wrapper
    assert "positive_control_long_step.py" in wrapper
    assert "--dependency" not in submit
    assert '"dependent_jobs": []' in submit
    assert '"one_hundred_step_training_queued": False' in submit
    assert '"checkpoint_evaluations_queued": False' in submit
