#!/usr/bin/env python3
"""Create an immutable OPSD execution tree with audited harness edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PINNED_UPSTREAM_COMMIT = "7448751f307a9cdbcc1246dd1565a1a605b443df"

TRAIN_OLD = 'dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")'
TRAIN_NEW = '''train_parquet = os.environ["LEGALRAG_OPSD_TRAIN_PARQUET"]
    dataset = load_dataset("parquet", data_files={"train": train_parquet})'''

EVAL_OLD = 'dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")'
EVAL_NEW = '''aime_parquet = __import__("os").environ["LEGALRAG_OPSD_AIME24_PARQUET"]
        dataset = load_dataset("parquet", data_files=aime_parquet, split="train")'''

DYNAMIC_PAD_OLD = '''        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)'''

DYNAMIC_PAD_NEW = '''        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        observed_completion_lengths = [len(ids) for ids in completion_ids]
        if not observed_completion_lengths:
            raise RuntimeError("vLLM returned no student completions")
        batch_completion_length = max(1, max(observed_completion_lengths))
        at_cap_count = sum(
            length >= max_completion_length for length in observed_completion_lengths
        )
        self._last_completion_token_counts = observed_completion_lengths
        self._last_completion_at_cap = [
            length >= max_completion_length for length in observed_completion_lengths
        ]
        self._completion_total_count += len(observed_completion_lengths)
        self._completion_token_total += sum(observed_completion_lengths)
        self._completion_at_cap_count += at_cap_count
        print(
            "LEGALRAG_COMPLETION_STATS "
            f"rank={self.accelerator.process_index} "
            f"count={len(observed_completion_lengths)} "
            f"tokens={sum(observed_completion_lengths)} "
            f"at_cap={at_cap_count} cap={max_completion_length} "
            f"lengths={','.join(str(value) for value in observed_completion_lengths)}"
        )

        # Pad only to this rank-local batch maximum. Every generated token and
        # every active-token loss term is unchanged; cap-sized padding never
        # enters the model solely to be masked out again.
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > batch_completion_length:
                padded_completion_ids_list.append(completion_tensor[:batch_completion_length])
            elif len(completion_tensor) < batch_completion_length:
                padding_needed = batch_completion_length - len(completion_tensor)'''

BUFFER_INIT_OLD = '''        # Track generation outputs for saving
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5  # Save every 5 steps'''

BUFFER_INIT_NEW = '''        # Track every local trajectory with rank-stable identities.
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5  # Save every 5 steps
        self._local_generation_sequence = 0
        self._completion_total_count = 0
        self._completion_token_total = 0
        self._completion_at_cap_count = 0
        self._last_completion_token_counts = []
        self._last_completion_at_cap = []'''

SAVE_PATH_OLD = '''        output_file = generations_dir / f"generations_step_{step}.json"

        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)'''

SAVE_PATH_NEW = '''        rank = int(self.accelerator.process_index)
        output_file = generations_dir / f"generations_step_{step}_rank_{rank}.json"

        output_data = {
            "step": step,
            "rank": rank,
            "num_samples": len(self._generation_outputs_buffer),
            "cumulative_local_samples": self._completion_total_count,
            "cumulative_local_tokens": self._completion_token_total,
            "cumulative_local_at_cap": self._completion_at_cap_count,
            "generations": self._generation_outputs_buffer,
        }

        with open(output_file, "x", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)'''

BUFFER_APPEND_OLD = '''        # Collect generation outputs for saving
        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )'''

BUFFER_APPEND_NEW = '''        # Collect generation outputs for saving with the exact token-length surface.
        if len(prompt_texts) != len(self._last_completion_token_counts):
            raise RuntimeError("trajectory text and token-count cardinalities disagree")
        for index, (prompt, completion) in enumerate(zip(prompt_texts, completion_texts)):
            self._generation_outputs_buffer.append(
                {
                    "step": self.state.global_step,
                    "rank": int(self.accelerator.process_index),
                    "local_sequence": self._local_generation_sequence,
                    "prompt": prompt,
                    "completion": completion,
                    "completion_tokens": self._last_completion_token_counts[index],
                    "at_cap": self._last_completion_at_cap[index],
                    "max_completion_length": self.generation_config.max_new_tokens,
                }
            )
            self._local_generation_sequence += 1'''

FINAL_FLUSH_OLD = '''    trainer.train()

    trainer.save_model(training_args.output_dir)'''

FINAL_FLUSH_NEW = '''    trainer.train()

    # The upstream periodic saver can leave the final partial buffer unwritten.
    # Flush it under a unique final label so all training trajectories survive.
    if trainer._generation_outputs_buffer:
        trainer._save_generation_outputs(f"{trainer.state.global_step}_final")

    trainer.save_model(training_args.output_dir)'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def replace_once(path: Path, old: str, new: str) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one pinned source match in {path}")
    before = sha256(path)
    path.write_text(source.replace(old, new), encoding="utf-8")
    return {"before_sha256": before, "after_sha256": sha256(path)}


def replace_many(path: Path, replacements: list[tuple[str, str]]) -> dict[str, str | int]:
    source = path.read_text(encoding="utf-8")
    before = sha256(path)
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"expected exactly one pinned source match in {path}")
        source = source.replace(old, new)
    path.write_text(source, encoding="utf-8")
    return {
        "before_sha256": before,
        "after_sha256": sha256(path),
        "replacement_count": len(replacements),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    target = args.target.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace execution tree: {target}")
    if git(upstream, "rev-parse", "HEAD") != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError("upstream OPSD checkout is not at the pinned commit")
    if git(upstream, "status", "--porcelain=v1", "--untracked-files=no"):
        raise RuntimeError("upstream OPSD checkout has tracked modifications")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(upstream, target, ignore=shutil.ignore_patterns(".git"))
    changes = {
        "opsd_train.py": replace_once(target / "opsd_train.py", TRAIN_OLD, TRAIN_NEW),
        "eval/evaluate_math.py": replace_once(
            target / "eval/evaluate_math.py", EVAL_OLD, EVAL_NEW
        ),
        "opsd_trainer.py": replace_many(
            target / "opsd_trainer.py",
            [
                (DYNAMIC_PAD_OLD, DYNAMIC_PAD_NEW),
                (BUFFER_INIT_OLD, BUFFER_INIT_NEW),
                (SAVE_PATH_OLD, SAVE_PATH_NEW),
                (BUFFER_APPEND_OLD, BUFFER_APPEND_NEW),
            ],
        ),
    }
    changes["opsd_train.py_final_flush"] = replace_once(
        target / "opsd_train.py", FINAL_FLUSH_OLD, FINAL_FLUSH_NEW
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "opsd_audited_harness_execution_tree",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
        "semantic_edits": [],
        "semantic_preserving_efficiency_edits": [
            "pad each rank only to its observed batch maximum while retaining every generated token and active-token loss term",
        ],
        "harness_edits": [
            "load pinned training parquet from LEGALRAG_OPSD_TRAIN_PARQUET",
            "load pinned AIME24 parquet from LEGALRAG_OPSD_AIME24_PARQUET",
            "save every trajectory with rank, local sequence, token count, cap flag, prompt, and completion",
            "use rank-specific exclusive generation files and flush the final partial buffer",
        ],
        "changed_files": changes,
    }
    manifest_path = target / "LEGALRAG_EXECUTION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(manifest_path, 0o444)
    print(json.dumps({"status": "passed", "target": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
