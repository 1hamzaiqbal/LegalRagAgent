#!/usr/bin/env python3
"""Fail-fast probe for exact-token prompt-logprob scoring on a live vLLM server."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.opd import teacher_client
from scripts.opd_math.quality_gates import sha256_file, sha256_tree


LOCAL_BINDING_SCOPE = "local_linux_proc_process_binding_not_remote_cryptographic_attestation"
PROVENANCE_FILENAME = "merge_provenance.json"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _proc_start_time_ticks(stat_text: str, pid: int) -> int:
    """Parse Linux /proc/PID/stat field 22 without splitting the comm field."""

    close = stat_text.rfind(")")
    if close < 0 or not stat_text[:close].startswith(f"{pid} ("):
        raise ValueError(f"malformed /proc/{pid}/stat")
    fields_3_onward = stat_text[close + 1 :].strip().split()
    if len(fields_3_onward) < 20:
        raise ValueError(f"truncated /proc/{pid}/stat")
    try:
        start_time = int(fields_3_onward[19])
    except ValueError as exc:
        raise ValueError(f"invalid /proc/{pid}/stat start time") from exc
    if start_time <= 0:
        raise ValueError(f"invalid /proc/{pid}/stat start time")
    return start_time


def _status_uid(status_text: str, pid: int) -> int:
    for line in status_text.splitlines():
        if line.startswith("Uid:"):
            fields = line.split()
            if len(fields) < 2:
                break
            try:
                return int(fields[1])
            except ValueError as exc:
                raise ValueError(f"invalid /proc/{pid}/status Uid") from exc
    raise ValueError(f"/proc/{pid}/status lacks Uid")


def _option_value(argv: list[str], option: str) -> str | None:
    for index, value in enumerate(argv):
        if value == option:
            if index + 1 >= len(argv):
                raise ValueError(f"server command line ends after {option}")
            return argv[index + 1]
        prefix = option + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _served_checkpoint_arg(argv: list[str]) -> str:
    indices = [index for index, value in enumerate(argv) if value == "serve"]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError("server command line must contain exactly one `serve CHECKPOINT` pair")
    return argv[indices[0] + 1]


def _read_proc_identity(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    if pid <= 0:
        raise ValueError("--server-pid must be positive")
    proc_dir = proc_root / str(pid)
    stat_path = proc_dir / "stat"
    cmdline_path = proc_dir / "cmdline"
    status_path = proc_dir / "status"
    try:
        first_stat = stat_path.read_text()
        cmdline_raw = cmdline_path.read_bytes()
        status_text = status_path.read_text()
        second_stat = stat_path.read_text()
    except FileNotFoundError as exc:
        raise RuntimeError(f"local server process {pid} is not live under {proc_root}") from exc
    first_start = _proc_start_time_ticks(first_stat, pid)
    second_start = _proc_start_time_ticks(second_stat, pid)
    if first_start != second_start:
        raise RuntimeError(f"local server PID {pid} changed identity during /proc inspection")
    argv_bytes = [value for value in cmdline_raw.split(b"\0") if value]
    if not argv_bytes:
        raise RuntimeError(f"local server process {pid} has an empty command line")
    argv = [value.decode("utf-8", errors="surrogateescape") for value in argv_bytes]
    uid = _status_uid(status_text, pid)
    if uid != os.getuid():
        raise RuntimeError(
            f"local server process {pid} is owned by uid={uid}, expected uid={os.getuid()}"
        )
    boot_id_path = proc_root / "sys" / "kernel" / "random" / "boot_id"
    boot_id = boot_id_path.read_text().strip() if boot_id_path.is_file() else None
    if not boot_id:
        raise RuntimeError(f"Linux boot ID is unavailable under {proc_root}")
    try:
        executable = str((proc_dir / "exe").resolve(strict=True))
        working_directory = str((proc_dir / "cwd").resolve(strict=True))
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"cannot resolve executable or working directory for local server PID {pid}"
        ) from exc
    return {
        "pid": pid,
        "proc_start_time_ticks": first_start,
        "linux_boot_id": boot_id,
        "process_uid": uid,
        "executable": executable,
        "working_directory": working_directory,
        "cmdline_sha256": hashlib.sha256(cmdline_raw).hexdigest(),
        "argv_count": len(argv),
        "argv": argv,
    }


def build_local_process_binding(
    *,
    server_pid: int,
    teacher_checkpoint: Path,
    teacher_provenance_manifest: Path,
    server_url: str,
    server_model: str,
    server_max_model_len: int,
    proc_root: Path = Path("/proc"),
    verify_checkpoint_tree: bool = True,
) -> dict[str, Any]:
    """Bind one live, local Linux process to a provenance-gated checkpoint.

    This is deliberately a same-host /proc custody check. It is not remote or
    cryptographic attestation of arbitrary server hardware.
    """

    parsed = urlparse(server_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("scientific local process binding requires a loopback http server URL")
    if parsed.port is None:
        raise ValueError("scientific local process binding requires an explicit server port")
    if server_max_model_len <= 0:
        raise ValueError("scientific local process binding requires a positive server max model length")

    checkpoint = Path(teacher_checkpoint).resolve()
    provenance_path = Path(teacher_provenance_manifest).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"teacher checkpoint is not a local directory: {checkpoint}")
    expected_provenance = (checkpoint / PROVENANCE_FILENAME).resolve()
    if provenance_path != expected_provenance:
        raise ValueError(
            "teacher provenance must be the merge_provenance.json inside the served checkpoint"
        )
    provenance = _json_object(provenance_path, "teacher provenance")
    if provenance.get("schema") != "opd_math_merged_teacher_v2":
        raise ValueError("teacher provenance has the wrong schema")
    if provenance.get("output_checkpoint") != str(checkpoint):
        raise ValueError("teacher provenance output_checkpoint differs from the served checkpoint")
    recorded_checkpoint_hash = provenance.get("output_checkpoint_tree_sha256")
    if not isinstance(recorded_checkpoint_hash, str) or len(recorded_checkpoint_hash) != 64:
        raise ValueError("teacher provenance lacks the served checkpoint tree hash")
    checkpoint_hash = recorded_checkpoint_hash
    if verify_checkpoint_tree:
        checkpoint_hash = sha256_tree(
            checkpoint, exclude_relative_paths=(PROVENANCE_FILENAME,)
        )
        if recorded_checkpoint_hash != checkpoint_hash:
            raise ValueError("served checkpoint tree differs from teacher provenance")

    proc = _read_proc_identity(server_pid, proc_root=proc_root)
    argv = proc.pop("argv")
    checkpoint_arg = _served_checkpoint_arg(argv)
    checkpoint_arg_path = Path(checkpoint_arg)
    if not checkpoint_arg_path.is_absolute():
        checkpoint_arg_path = Path(proc["working_directory"]) / checkpoint_arg_path
    if checkpoint_arg_path.resolve() != checkpoint:
        raise ValueError(
            "local vLLM process command line is not serving the gated teacher checkpoint"
        )
    served_model = _option_value(argv, "--served-model-name")
    if served_model != server_model:
        raise ValueError(
            f"local vLLM served-model alias mismatch: expected={server_model!r}, actual={served_model!r}"
        )
    port = _option_value(argv, "--port")
    if port != str(parsed.port):
        raise ValueError(
            f"local vLLM port mismatch: expected={parsed.port!r}, actual={port!r}"
        )
    max_model_len = _option_value(argv, "--max-model-len")
    try:
        parsed_max_model_len = int(max_model_len) if max_model_len is not None else None
    except ValueError as exc:
        raise ValueError(
            f"local vLLM max model length is not an integer: {max_model_len!r}"
        ) from exc
    if parsed_max_model_len != server_max_model_len:
        raise ValueError(
            "local vLLM max model length mismatch: "
            f"expected={server_max_model_len!r}, actual={parsed_max_model_len!r}"
        )
    return {
        "schema_version": 1,
        "scope": LOCAL_BINDING_SCOPE,
        "validated": True,
        **proc,
        "server_url": server_url.rstrip("/"),
        "server_model": server_model,
        "server_max_model_len": server_max_model_len,
        "served_checkpoint_argv": checkpoint_arg,
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_tree_sha256": checkpoint_hash,
        "teacher_provenance_manifest": str(provenance_path),
        "teacher_provenance_manifest_sha256": sha256_file(provenance_path),
    }


def revalidate_local_process_binding(
    binding: dict[str, Any],
    *,
    teacher_checkpoint: Path,
    teacher_provenance_manifest: Path,
    server_url: str,
    server_model: str,
    server_max_model_len: int,
    proc_root: Path = Path("/proc"),
    verify_checkpoint_tree: bool = False,
) -> dict[str, Any]:
    """Reinspect /proc and require the live identity to equal the recorded binding."""

    if not isinstance(binding, dict) or binding.get("scope") != LOCAL_BINDING_SCOPE:
        raise ValueError("server scoring contract lacks the local Linux process-binding scope")
    if binding.get("validated") is not True:
        raise ValueError("server scoring contract local process binding did not pass")
    pid = binding.get("pid")
    if not isinstance(pid, int):
        raise ValueError("server scoring contract local process binding lacks an integer PID")
    current = build_local_process_binding(
        server_pid=pid,
        teacher_checkpoint=teacher_checkpoint,
        teacher_provenance_manifest=teacher_provenance_manifest,
        server_url=server_url,
        server_model=server_model,
        server_max_model_len=server_max_model_len,
        proc_root=proc_root,
        verify_checkpoint_tree=verify_checkpoint_tree,
    )
    if current != binding:
        changed = sorted(
            key for key in set(binding) | set(current) if binding.get(key) != current.get(key)
        )
        raise RuntimeError(
            "local vLLM process binding changed since the scoring probe; "
            f"changed_fields={changed}"
        )
    return current


def scientific_binding_inputs_complete(
    server_pid: int | None,
    teacher_checkpoint: Path | None,
    teacher_provenance_manifest: Path | None,
    server_max_model_len: int | None,
) -> bool:
    supplied = [
        server_pid is not None,
        teacher_checkpoint is not None,
        teacher_provenance_manifest is not None,
        server_max_model_len is not None,
    ]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "--server-pid, --teacher-checkpoint, --teacher-provenance-manifest, and "
            "--teacher-server-max-model-len "
            "must be supplied together"
        )
    return all(supplied)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--server-model", required=True)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-provenance-manifest", type=Path)
    parser.add_argument("--teacher-server-max-model-len", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    has_scientific_binding_inputs = scientific_binding_inputs_complete(
        args.server_pid,
        args.teacher_checkpoint,
        args.teacher_provenance_manifest,
        args.teacher_server_max_model_len,
    )
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite scoring probe: {args.output}")
    local_process_binding = None
    if has_scientific_binding_inputs:
        local_process_binding = build_local_process_binding(
            server_pid=args.server_pid,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_provenance_manifest=args.teacher_provenance_manifest,
            server_url=args.server_url,
            server_model=args.server_model,
            server_max_model_len=args.teacher_server_max_model_len,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    messages = [{"role": "user", "content": "Return the number after 1: 2"}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    completion_ids = list(tokenizer.encode("2", add_special_tokens=False))
    values = teacher_client.score_completion_token_logprobs(
        args.server_url,
        args.server_model,
        prompt_ids,
        completion_ids,
    )
    if len(values) != len(completion_ids) or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid exact-token scoring response: {values}")
    if local_process_binding is not None:
        local_process_binding = revalidate_local_process_binding(
            local_process_binding,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_provenance_manifest=args.teacher_provenance_manifest,
            server_url=args.server_url,
            server_model=args.server_model,
            server_max_model_len=args.teacher_server_max_model_len,
        )
    result = {
        "schema_version": 2,
        "probe": "exact_token_teacher_scoring_v1",
        "passed": True,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "server_url": args.server_url.rstrip("/"),
        "server_model": args.server_model,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens_scored": len(values),
        "local_process_binding_validated": local_process_binding is not None,
        "local_process_binding": local_process_binding,
        "claim_boundary": (
            "When present, the binding proves same-host Linux /proc custody for the named "
            "PID, command line, checkpoint, and provenance at probe time. It is not "
            "cryptographic remote attestation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
