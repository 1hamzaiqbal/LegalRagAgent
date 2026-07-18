import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.opd.opd_train import (
    EXPECTED_MERGE_PACKAGES,
    MERGED_TEACHER_SCHEMA,
    MERGER_FILE,
    _local_server_process_binding_state,
    _server_process_binding_gate_satisfied,
    _validate_server_scoring_contract,
    _validate_teacher_provenance,
    clean_stable_git_custody,
    sha256_file,
    validate_server_environment_process_binding,
)
from scripts.opd_math.quality_gates import sha256_tree
from scripts.opd_math.server_scoring_probe import (
    LOCAL_BINDING_SCOPE,
    build_local_process_binding,
    revalidate_local_process_binding,
    scientific_binding_inputs_complete,
)


PID = 4312
PORT = 18432
MODEL = "opd-math-teacher"


def canonical_server_tail(checkpoint):
    return [
        "serve",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--served-model-name",
        MODEL,
        "--port",
        str(PORT),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.55",
    ]


def test_local_server_process_binding_state_separates_requirement_from_validation():
    missing = {"live_local_server_process_binding_validated": False}
    validated = {"live_local_server_process_binding_validated": True}

    assert _local_server_process_binding_state("task_rl_k1_gap", False, missing) == (
        False,
        False,
    )
    assert _local_server_process_binding_state("task_rl_k1_gap", True, missing) == (
        True,
        False,
    )
    assert _local_server_process_binding_state("task_rl_k1_gap", True, validated) == (
        True,
        True,
    )
    assert _local_server_process_binding_state("task_rl", True, missing) == (False, False)


def test_server_process_binding_gate_accepts_modes_without_a_teacher_server():
    assert _server_process_binding_gate_satisfied(False, False)
    assert _server_process_binding_gate_satisfied(False, True)
    assert _server_process_binding_gate_satisfied(True, True)
    assert not _server_process_binding_gate_satisfied(True, False)


def write_checkpoint(tmp_path):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    tree_hash = sha256_tree(checkpoint)
    provenance = checkpoint / "merge_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "schema": "opd_math_merged_teacher_v2",
                "output_checkpoint": str(checkpoint.resolve()),
                "output_checkpoint_tree_sha256": tree_hash,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return checkpoint, provenance


def write_fake_proc(
    tmp_path,
    checkpoint,
    *,
    start_time=987654,
    served_checkpoint=None,
    launcher_root=None,
    process_executable=None,
    extra_argv=(),
    command_tail=None,
):
    proc_root = tmp_path / "proc"
    proc_dir = proc_root / str(PID)
    proc_dir.mkdir(parents=True)
    boot_id = proc_root / "sys" / "kernel" / "random" / "boot_id"
    boot_id.parent.mkdir(parents=True)
    boot_id.write_text("00000000-1111-2222-3333-444444444444\n")

    if launcher_root is not None:
        bin_dir = launcher_root / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        launcher_python = bin_dir / "python"
        launcher_vllm = bin_dir / "vllm"
        launcher_python.write_text("fake environment interpreter\n")
        launcher_vllm.write_text("fake vllm launcher\n")
        launcher_prefix = [str(launcher_python), str(launcher_vllm)]
        executable = Path(process_executable or launcher_python)
    else:
        executable = Path(process_executable or (tmp_path / "vllm"))
        launcher_prefix = [str(executable)]
    if not executable.exists():
        executable.write_text("fake executable\n")
    (proc_dir / "exe").symlink_to(executable)
    (proc_dir / "cwd").symlink_to(tmp_path)
    fields_3_to_22 = ["S"] + ["0"] * 18 + [str(start_time)]
    (proc_dir / "stat").write_text(
        f"{PID} (vllm worker with spaces) " + " ".join(fields_3_to_22) + "\n"
    )
    (proc_dir / "status").write_text(f"Name:\tvllm\nUid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n")
    tail = (
        list(command_tail)
        if command_tail is not None
        else canonical_server_tail(served_checkpoint or checkpoint)
    )
    argv = launcher_prefix + tail + list(extra_argv)
    (proc_dir / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
    return proc_root


def test_local_process_binding_reinspects_pid_start_cmdline_and_checkpoint(tmp_path):
    checkpoint, provenance = write_checkpoint(tmp_path)
    proc_root = write_fake_proc(tmp_path, checkpoint)
    kwargs = {
        "teacher_checkpoint": checkpoint,
        "teacher_provenance_manifest": provenance,
        "server_url": f"http://127.0.0.1:{PORT}",
        "server_model": MODEL,
        "server_max_model_len": 4096,
        "proc_root": proc_root,
    }
    binding = build_local_process_binding(server_pid=PID, **kwargs)
    assert binding["scope"] == LOCAL_BINDING_SCOPE
    assert binding["validated"] is True
    assert binding["teacher_checkpoint"] == str(checkpoint.resolve())
    assert binding["server_max_model_len"] == 4096
    assert binding["proc_start_time_ticks"] == 987654
    assert revalidate_local_process_binding(binding, **kwargs) == binding

    write_fake_stat = proc_root / str(PID) / "stat"
    fields = ["S"] + ["0"] * 18 + ["987655"]
    write_fake_stat.write_text(
        f"{PID} (vllm worker with spaces) " + " ".join(fields) + "\n"
    )
    with pytest.raises(RuntimeError, match="changed since the scoring probe"):
        revalidate_local_process_binding(binding, **kwargs)


def test_legacy_no_environment_binding_keeps_flexible_option_shape(tmp_path):
    checkpoint, provenance = write_checkpoint(tmp_path)
    proc_root = write_fake_proc(
        tmp_path,
        checkpoint,
        command_tail=[
            "serve",
            str(checkpoint),
            "--max-model-len=4096",
            "--gpu-memory-utilization",
            "0.73",
            f"--port={PORT}",
            f"--served-model-name={MODEL}",
            "--disable-log-requests",
        ],
    )
    binding = build_local_process_binding(
        server_pid=PID,
        teacher_checkpoint=checkpoint,
        teacher_provenance_manifest=provenance,
        server_url=f"http://127.0.0.1:{PORT}",
        server_model=MODEL,
        server_max_model_len=4096,
        proc_root=proc_root,
    )
    assert "serve_environment_launcher" not in binding
    assert "canonical_scientific_argv" not in binding


def test_binding_rejects_declared_server_context_mismatch(tmp_path):
    checkpoint, provenance = write_checkpoint(tmp_path)
    proc_root = write_fake_proc(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="max model length mismatch"):
        build_local_process_binding(
            server_pid=PID,
            teacher_checkpoint=checkpoint,
            teacher_provenance_manifest=provenance,
            server_url=f"http://127.0.0.1:{PORT}",
            server_model=MODEL,
            server_max_model_len=8192,
            proc_root=proc_root,
        )


@pytest.mark.parametrize(
    "duplicate",
    [
        ("--served-model-name", "spoofed-model"),
        ("--port=9999",),
        ("--max-model-len", "8192"),
    ],
)
def test_binding_rejects_duplicate_bound_server_options(tmp_path, duplicate):
    checkpoint, provenance = write_checkpoint(tmp_path)
    proc_root = write_fake_proc(tmp_path, checkpoint, extra_argv=duplicate)
    with pytest.raises(ValueError, match="must contain exactly one"):
        build_local_process_binding(
            server_pid=PID,
            teacher_checkpoint=checkpoint,
            teacher_provenance_manifest=provenance,
            server_url=f"http://127.0.0.1:{PORT}",
            server_model=MODEL,
            server_max_model_len=4096,
            proc_root=proc_root,
        )


def test_binding_rejects_wrong_cmdline_checkpoint_or_provenance_location(tmp_path):
    checkpoint, provenance = write_checkpoint(tmp_path)
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    proc_root = write_fake_proc(tmp_path, checkpoint, served_checkpoint=wrong)
    with pytest.raises(ValueError, match="not serving the gated"):
        build_local_process_binding(
            server_pid=PID,
            teacher_checkpoint=checkpoint,
            teacher_provenance_manifest=provenance,
            server_url=f"http://127.0.0.1:{PORT}",
            server_model=MODEL,
            server_max_model_len=4096,
            proc_root=proc_root,
        )

    copied = tmp_path / "copied-provenance.json"
    copied.write_bytes(provenance.read_bytes())
    with pytest.raises(ValueError, match="inside the served checkpoint"):
        build_local_process_binding(
            server_pid=PID,
            teacher_checkpoint=checkpoint,
            teacher_provenance_manifest=copied,
            server_url=f"http://127.0.0.1:{PORT}",
            server_model=MODEL,
            server_max_model_len=4096,
            proc_root=proc_root,
        )


def test_scientific_process_inputs_are_all_or_none():
    assert not scientific_binding_inputs_complete(None, None, None, None, None)
    assert scientific_binding_inputs_complete(
        PID, Path("checkpoint"), Path("provenance"), 4096, Path("serve-env")
    )
    with pytest.raises(ValueError, match="must be supplied together"):
        scientific_binding_inputs_complete(
            PID, Path("checkpoint"), None, 4096, Path("serve-env")
        )


def test_process_launcher_must_match_selected_serve_environment(tmp_path):
    checkpoint, provenance = write_checkpoint(tmp_path)
    serve_b = tmp_path / "serve_b"
    serve_c = tmp_path / "serve_c"
    serve_b.mkdir()
    serve_c.mkdir()
    serve_b_bin = serve_b / "bin"
    serve_b_bin.mkdir()
    (serve_b_bin / "python").write_text("other environment interpreter\n")
    (serve_b_bin / "vllm").write_text("other vllm launcher\n")
    proc_root = write_fake_proc(tmp_path, checkpoint, launcher_root=serve_c)
    kwargs = {
        "teacher_checkpoint": checkpoint,
        "teacher_provenance_manifest": provenance,
        "server_url": f"http://127.0.0.1:{PORT}",
        "server_model": MODEL,
        "server_max_model_len": 4096,
        "proc_root": proc_root,
    }
    binding = build_local_process_binding(
        server_pid=PID, serve_environment_root=serve_c, **kwargs
    )
    assert binding["serve_environment_launcher"]["argv_prefix"] == [
        str(serve_c / "bin" / "python"),
        str(serve_c / "bin" / "vllm"),
    ]
    assert binding["serve_environment_launcher"]["resolved_python_executable"] == str(
        (serve_c / "bin" / "python").resolve()
    )
    assert binding["executable"] == str((serve_c / "bin" / "python").resolve())
    assert binding["argv_count"] == 14
    assert len(binding["canonical_scientific_argv"]) == 14
    assert binding["canonical_scientific_argv"][4:6] == ["--host", "127.0.0.1"]
    serve_c_contract = {
        "serve_verification": {
            "environment_root": str(serve_c),
            "live_python": str(serve_c / "bin" / "python"),
            "expected_executable": {"path": str(serve_c / "bin" / "vllm")},
        }
    }
    assert (
        validate_server_environment_process_binding(binding, serve_c_contract)
        == binding["serve_environment_launcher"]
    )
    assert (
        revalidate_local_process_binding(
            binding, serve_environment_root=serve_c, **kwargs
        )
        == binding
    )
    with pytest.raises(ValueError, match="exact canonical scientific launcher argv"):
        build_local_process_binding(
            server_pid=PID, serve_environment_root=serve_b, **kwargs
        )
    serve_b_contract = {
        "serve_verification": {
            "environment_root": str(serve_b),
            "live_python": str(serve_b / "bin" / "python"),
            "expected_executable": {"path": str(serve_b / "bin" / "vllm")},
        }
    }
    with pytest.raises(ValueError, match="was not launched by the verified"):
        validate_server_environment_process_binding(binding, serve_b_contract)


def test_process_launcher_rejects_spoofed_argv_with_wrong_proc_executable(tmp_path):
    checkpoint, provenance = write_checkpoint(tmp_path)
    serve_environment = tmp_path / "serve_environment"
    spoofed_executable = tmp_path / "spoofed_python"
    spoofed_executable.write_text("not the selected environment interpreter\n")
    proc_root = write_fake_proc(
        tmp_path,
        checkpoint,
        launcher_root=serve_environment,
        process_executable=spoofed_executable,
    )

    with pytest.raises(ValueError, match="executable does not resolve"):
        build_local_process_binding(
            server_pid=PID,
            teacher_checkpoint=checkpoint,
            teacher_provenance_manifest=provenance,
            server_url=f"http://127.0.0.1:{PORT}",
            server_model=MODEL,
            server_max_model_len=4096,
            serve_environment_root=serve_environment,
            proc_root=proc_root,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "underscore_served_model",
        "underscore_max_model_len",
        "missing_host",
        "wildcard_host",
        "alternate_host",
        "abbreviated_host",
        "abbreviated_served_model",
        "abbreviated_port",
        "abbreviated_max_model_len",
        "config",
        "extra_option",
        "reordered_host",
        "reordered_options",
    ],
)
def test_scientific_launcher_rejects_every_noncanonical_argv_form(tmp_path, mutation):
    checkpoint, provenance = write_checkpoint(tmp_path)
    serve_environment = tmp_path / "serve_environment"
    tail = canonical_server_tail(checkpoint)
    if mutation == "underscore_served_model":
        tail[4] = "--served_model_name"
    elif mutation == "underscore_max_model_len":
        tail[8] = "--max_model_len"
    elif mutation == "missing_host":
        del tail[2:4]
    elif mutation == "wildcard_host":
        tail[3] = "0.0.0.0"
    elif mutation == "alternate_host":
        tail[3] = "127.0.0.2"
    elif mutation == "abbreviated_host":
        tail[2] = "--ho"
    elif mutation == "abbreviated_served_model":
        tail[4] = "--served-model-n"
    elif mutation == "abbreviated_port":
        tail[6] = "--por"
    elif mutation == "abbreviated_max_model_len":
        tail[8] = "--max-model-l"
    elif mutation == "config":
        tail.extend(["--config", str(tmp_path / "vllm-config.json")])
    elif mutation == "extra_option":
        tail.append("--disable-log-requests")
    elif mutation == "reordered_host":
        tail[2:6] = tail[4:6] + tail[2:4]
    elif mutation == "reordered_options":
        tail[6:10] = tail[8:10] + tail[6:8]
    proc_root = write_fake_proc(
        tmp_path,
        checkpoint,
        launcher_root=serve_environment,
        command_tail=tail,
    )

    with pytest.raises(ValueError, match="exact canonical scientific launcher argv"):
        build_local_process_binding(
            server_pid=PID,
            teacher_checkpoint=checkpoint,
            teacher_provenance_manifest=provenance,
            server_url=f"http://127.0.0.1:{PORT}",
            server_model=MODEL,
            server_max_model_len=4096,
            serve_environment_root=serve_environment,
            proc_root=proc_root,
        )


def test_student_git_custody_requires_same_clean_40_hex_commit():
    start = {"commit": "a" * 40, "dirty": False}
    assert clean_stable_git_custody(start, dict(start))
    assert not clean_stable_git_custody(start, {"commit": "b" * 40, "dirty": False})
    assert not clean_stable_git_custody(start, {"commit": "a" * 40, "dirty": True})
    assert not clean_stable_git_custody({"commit": None, "dirty": False}, dict(start))


def test_student_requires_trusted_merge_code_custody(tmp_path, monkeypatch):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    state = {"commit": "a" * 40, "dirty": False}
    gate = {
        "manifest_sha256": "b" * 64,
        "trained_adapter": str((tmp_path / "adapter").resolve()),
        "trained_adapter_tree_sha256": "c" * 64,
    }
    args = Namespace(
        teacher_base_model="Qwen/Qwen3-8B",
        teacher_base_revision="d" * 40,
        teacher_checkpoint=str(checkpoint),
    )
    payload = {
        "schema_version": 1,
        "schema": MERGED_TEACHER_SCHEMA,
        "status": "completed",
        "teacher_gap_manifest_sha256": gate["manifest_sha256"],
        "base_model": args.teacher_base_model,
        "base_revision": args.teacher_base_revision,
        "adapter": gate["trained_adapter"],
        "adapter_tree_sha256": gate["trained_adapter_tree_sha256"],
        "output_checkpoint": str(checkpoint.resolve()),
        "output_checkpoint_tree_sha256": sha256_tree(checkpoint),
        "merge_code": {
            "git_state_start": state,
            "git_state_after_merge": state,
            "git_state_before_promotion": state,
            "git_state_end": state,
            "clean_stable_code": True,
            "merger_file_sha256": sha256_file(MERGER_FILE),
            "packages": EXPECTED_MERGE_PACKAGES,
        },
    }
    provenance = checkpoint / "merge_provenance.json"
    provenance.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr("scripts.opd.opd_train.git_state", lambda: dict(state))
    validated = _validate_teacher_provenance(str(provenance), gate, args)
    assert validated["merge_code"]["clean_stable_code"] is True

    payload.pop("merge_code")
    provenance.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="merge-code custody"):
        _validate_teacher_provenance(str(provenance), gate, args)


def test_scientific_student_revalidates_probe_process_binding(tmp_path, monkeypatch):
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()
    provenance = checkpoint / "merge_provenance.json"
    provenance.write_text("{}\n")
    binding = {
        "schema_version": 1,
        "scope": LOCAL_BINDING_SCOPE,
        "validated": True,
        "pid": PID,
    }
    probe = {
        "schema_version": 2,
        "probe": "exact_token_teacher_scoring_v1",
        "passed": True,
        "tokenizer": "Qwen/Qwen3-1.7B",
        "tokenizer_revision": "a" * 40,
        "server_url": f"http://127.0.0.1:{PORT}",
        "server_model": MODEL,
        "local_process_binding_validated": True,
        "local_process_binding": binding,
    }
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe) + "\n")
    args = Namespace(
        allow_ungated_smoke=False,
        student="Qwen/Qwen3-1.7B",
        student_revision="a" * 40,
        teacher_url=f"http://127.0.0.1:{PORT}",
        teacher_model=MODEL,
        teacher_checkpoint=str(checkpoint),
        teacher_provenance_manifest=str(provenance),
        teacher_server_max_model_len=4096,
        serve_environment_root=str(tmp_path / "serve_environment"),
    )
    calls = []

    def fake_revalidate(candidate, **kwargs):
        calls.append((candidate, kwargs))
        return candidate

    monkeypatch.setattr(
        "scripts.opd.opd_train.revalidate_local_process_binding", fake_revalidate
    )
    result = _validate_server_scoring_contract(str(probe_path), args=args)
    assert result["local_process_binding"] == binding
    assert len(calls) == 1
    assert calls[0][1]["teacher_checkpoint"] == checkpoint
    assert calls[0][1]["teacher_provenance_manifest"] == provenance
    assert calls[0][1]["server_max_model_len"] == 4096

    probe["local_process_binding_validated"] = False
    probe_path.write_text(json.dumps(probe) + "\n")
    with pytest.raises(ValueError, match="local_process_binding_validated mismatch"):
        _validate_server_scoring_contract(str(probe_path), args=args)

    # The distinct smoke lane may still prove exact-token scoring without
    # claiming local checkpoint/process custody.
    probe["local_process_binding"] = None
    probe_path.write_text(json.dumps(probe) + "\n")
    args.allow_ungated_smoke = True
    smoke = _validate_server_scoring_contract(str(probe_path), args=args)
    assert smoke["local_process_binding"] is None
