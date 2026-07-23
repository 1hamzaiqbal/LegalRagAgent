#!/usr/bin/env python3
"""Verify the isolated environment used for the pinned upstream OPSD control."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path


EXPECTED = {
    "accelerate": "1.11.0",
    "bitsandbytes": "0.48.2",
    "datasets": "3.6.0",
    "deepspeed": "0.18.2",
    "einops": "0.8.1",
    "flash-attn": "2.8.3",
    "math-verify": "0.8.0",
    "peft": "0.17.1",
    "safetensors": "0.5.3",
    "sentencepiece": "0.1.99",
    "tiktoken": "0.9.0",
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "triton": "3.4.0",
    "trl": "0.26.0",
    "vllm": "0.11.0",
    "wandb": "0.22.3",
    "xformers": "0.0.32.post1",
}

IMPORTS = [
    "accelerate",
    "bitsandbytes",
    "datasets",
    "deepspeed",
    "einops",
    "flash_attn",
    "math_verify",
    "peft",
    "safetensors",
    "torch",
    "transformers",
    "trl",
    "vllm",
    "wandb",
    "xformers",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-cuda-devices", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.environment_root.resolve()
    freeze = args.freeze.resolve()
    if not (root / "bin/python").is_file():
        raise RuntimeError(f"environment Python is missing under {root}")
    if not freeze.is_file():
        raise RuntimeError(f"environment freeze is missing: {freeze}")

    installed = {name: importlib.metadata.version(name) for name in EXPECTED}
    mismatches = {
        name: {"expected": EXPECTED[name], "observed": installed[name]}
        for name in EXPECTED
        if installed[name] != EXPECTED[name]
    }
    if mismatches:
        raise RuntimeError(f"package-version mismatch: {json.dumps(mismatches)}")
    for module in IMPORTS:
        importlib.import_module(module)

    import torch

    cuda_devices = torch.cuda.device_count()
    if args.expected_cuda_devices is not None and cuda_devices != args.expected_cuda_devices:
        raise RuntimeError(
            f"expected {args.expected_cuda_devices} CUDA devices, observed {cuda_devices}"
        )
    if args.expected_cuda_devices and not torch.cuda.is_bf16_supported():
        raise RuntimeError("visible CUDA hardware does not report bfloat16 support")

    payload = {
        "status": "passed",
        "environment_root": str(root),
        "freeze": str(freeze),
        "freeze_sha256": sha256(freeze),
        "packages": installed,
        "cuda_devices": cuda_devices,
        "cuda_names": [torch.cuda.get_device_name(index) for index in range(cuda_devices)],
        "bf16_supported": bool(cuda_devices and torch.cuda.is_bf16_supported()),
    }
    if args.output is not None:
        write_exclusive(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
