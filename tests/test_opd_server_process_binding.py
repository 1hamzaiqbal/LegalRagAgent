import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.opd.opd_train import (
    EXPECTED_MERGE_PACKAGES,
    MERGED_TEACHER_SCHEMA,
    MERGER_FILE,
    _validate_server_scoring_contract,
    _validate_teacher_provenance,
    clean_stable_git_custody,
    sha256_file,
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


def write_fake_proc(tmp_path, checkpoint, *, start_time=987654, served_checkpoint=None):
    proc_root = tmp_path / "proc"
    proc_dir = proc_root / str(PID)
    proc_dir.mkdir(parents=True)
    boot_id = proc_root / "sys" / "kernel" / "random" / "boot_id"
    boot_id.parent.mkdir(parents=True)
    boot_id.write_text("00000000-1111-2222-3333-444444444444\n")

    executable = tmp_path / "vllm"
    executable.write_text("fake executable\n")
    (proc_dir / "exe").symlink_to(executable)
    (proc_dir / "cwd").symlink_to(tmp_path)
    fields_3_to_22 = ["S"] + ["0"] * 18 + [str(start_time)]
    (proc_dir / "stat").write_text(
        f"{PID} (vllm worker with spaces) " + " ".join(fields_3_to_22) + "\n"
    )
    (proc_dir / "status").write_text(f"Name:\tvllm\nUid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n")
    argv = [
        str(executable),
        "serve",
        str(served_checkpoint or checkpoint),
        "--served-model-name",
        MODEL,
        "--port",
        str(PORT),
        "--max-model-len",
        "4096",
    ]
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
    assert not scientific_binding_inputs_complete(None, None, None, None)
    assert scientific_binding_inputs_complete(
        PID, Path("checkpoint"), Path("provenance"), 4096
    )
    with pytest.raises(ValueError, match="must be supplied together"):
        scientific_binding_inputs_complete(PID, Path("checkpoint"), None, 4096)


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
