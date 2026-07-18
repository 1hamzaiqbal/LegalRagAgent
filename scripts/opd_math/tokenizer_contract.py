#!/usr/bin/env python3
"""Verify exact teacher/student tokenizer and non-thinking prompt alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROBES = (
    "2 + 2 =",
    r"Find $x$ if $\frac{x+1}{2}=4$.",
    "Whitespace boundary probe: answer",
    "Unicode: α ≤ β and ½.",
)
CHAT_PROBE = [{"role": "user", "content": r"Solve $x^2=4$ and answer in \boxed{}."}]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenizer_fingerprint(tokenizer) -> dict:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise ValueError("exact OPD tokenizer custody requires a serializable fast-tokenizer backend")
    backend_json = backend.to_str()
    vocab = sorted((str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items())
    special = {
        name: getattr(tokenizer, name, None)
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "sep_token_id",
            "cls_token_id",
            "mask_token_id",
        )
    }
    chat_template = tokenizer.chat_template or ""
    rendered_false = tokenizer.apply_chat_template(
        CHAT_PROBE,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rendered_true = tokenizer.apply_chat_template(
        CHAT_PROBE,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(vocab),
        "vocab_sha256": canonical_sha256(vocab),
        "backend_tokenizer_json_sha256": hashlib.sha256(
            backend_json.encode("utf-8")
        ).hexdigest(),
        "special_token_ids": special,
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "probe_token_ids": {
            probe: list(tokenizer.encode(probe, add_special_tokens=False)) for probe in PROBES
        },
        "nonthinking_render_sha256": hashlib.sha256(rendered_false.encode("utf-8")).hexdigest(),
        "thinking_render_sha256": hashlib.sha256(rendered_true.encode("utf-8")).hexdigest(),
        "nonthinking_probe_token_ids": list(tokenizer.encode(rendered_false, add_special_tokens=False)),
        "thinking_modes_differ": rendered_false != rendered_true,
    }


def comparable_contract(fingerprint: dict) -> dict:
    return {
        key: fingerprint[key]
        for key in (
            "vocab_size",
            "vocab_sha256",
            "backend_tokenizer_json_sha256",
            "special_token_ids",
            "chat_template_sha256",
            "probe_token_ids",
            "nonthinking_render_sha256",
            "nonthinking_probe_token_ids",
        )
    }


def server_tokenize(base_url: str, model: str, token_ids_expected: list[int], rendered: str) -> dict:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("server tokenizer probe requires requests") from exc
    response = requests.post(
        base_url.rstrip("/") + "/tokenize",
        json={"model": model, "prompt": rendered, "add_special_tokens": False},
        timeout=(10.0, 120.0),
    )
    response.raise_for_status()
    payload = response.json()
    token_ids = payload.get("tokens")
    if token_ids is None:
        token_ids = payload.get("token_ids")
    if token_ids is None:
        raise RuntimeError(f"server /tokenize response lacked token IDs: {payload}")
    token_ids = [int(x) for x in token_ids]
    return {"matches": token_ids == token_ids_expected, "token_ids": token_ids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-revision")
    parser.add_argument("--student", required=True)
    parser.add_argument("--student-revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url")
    parser.add_argument("--server-model")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite tokenizer contract: {args.output}")

    from transformers import AutoTokenizer

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher,
        revision=args.teacher_revision,
        local_files_only=args.local_files_only,
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student,
        revision=args.student_revision,
        local_files_only=args.local_files_only,
    )
    teacher = tokenizer_fingerprint(teacher_tokenizer)
    student = tokenizer_fingerprint(student_tokenizer)
    matched = comparable_contract(teacher) == comparable_contract(student)
    result = {
        "schema_version": 1,
        "gate": "tokenizer_contract_v1",
        "teacher": {"model": args.teacher, "revision": args.teacher_revision, **teacher},
        "student": {"model": args.student, "revision": args.student_revision, **student},
        "exact_contract_match": matched,
    }
    if not teacher["thinking_modes_differ"] or not student["thinking_modes_differ"]:
        result["error"] = "enable_thinking did not change the pinned Qwen chat rendering"
        matched = False

    if args.server_url:
        if not args.server_model:
            parser.error("--server-model is required with --server-url")
        rendered = student_tokenizer.apply_chat_template(
            CHAT_PROBE,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        expected = list(student_tokenizer.encode(rendered, add_special_tokens=False))
        result["server_probe"] = server_tokenize(args.server_url, args.server_model, expected, rendered)
        result["server"] = {
            "url": args.server_url.rstrip("/"),
            "model": args.server_model,
        }
        matched = matched and result["server_probe"]["matches"]

    result["passed"] = matched
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": matched}, sort_keys=True))
    return 0 if matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
