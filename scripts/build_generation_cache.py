#!/usr/bin/env python3
"""Build question-only HyDE or Snap-HyRE generation caches without retrieval.

The cache can be used by ``scripts/build_retrieval_cache.py`` for top-k
retrieval diagnostics and by ``eval/eval_harness.py --hyre-cache-path`` for
answer replay. This lets top-k selection happen before the expensive final
answer sweep while keeping generated queries fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import (  # noqa: E402
    _contains_answer_artifact,
    _extract_required_final_line_prediction,
    _fmt,
    _fmt_intermediate,
    _generate_hyde,
    _generate_snap_hyre_blocks,
    _get_call_trace,
    _get_metrics,
    _get_trace_events,
    _question_only_hyde_user,
    _reset_call_trace,
    _reset_llm_call_counter,
    _reset_trace_events,
    _row_label,
    _setup_provider,
    _llm_call,
)


def _load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    labels: set[str] = set()
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON in existing cache: {exc}") from exc
            label = row.get("label")
            if label:
                labels.add(str(label))
    return labels


def _row_idx(row: Any, fallback_i: int) -> str:
    value = row.get("idx", fallback_i)
    try:
        if value != value:
            return str(fallback_i)
    except Exception:
        pass
    return str(value)


def _no_silent_fallback_enabled() -> bool:
    return os.getenv("NO_SILENT_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _strict_generation_violations(record: dict[str, Any], mode: str) -> list[str]:
    violations: list[str] = []
    if record.get("error"):
        violations.append(f"error={str(record.get('error'))[:160]}")
    if not record.get("hyde_passage"):
        violations.append("missing_hyde_passage")
    if record.get("hyde_used_fallback") is True:
        violations.append("hyde_used_fallback=True")
    if record.get("hyde_contains_answer_artifact") is True:
        violations.append("hyde_contains_answer_artifact=True")
    if mode == "snap_hyre" and record.get("snap_hyre_parse_ok") is False:
        violations.append("snap_hyre_parse_ok=False")
    if mode == "snap_hyre" and not record.get("snap_letter"):
        violations.append("snap_letter missing required final answer line")
    return violations


def _build_rag_hyde(row, config: EvalConfig) -> dict[str, Any]:
    question_intermediate = _fmt_intermediate(row, config)
    hyde = _generate_hyde(
        config,
        "hyde",
        _question_only_hyde_user(question_intermediate),
        label="hyde/generate",
        fallback=question_intermediate,
    )
    return {
        "source_mode": "rag_hyde",
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "hyde_used_fallback": hyde.get("used_fallback", False),
        "hyde_parse_ok": bool(hyde["text"]),
    }


def _build_snap_hyre(row, config: EvalConfig) -> dict[str, Any]:
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    raw, snap_block, hyre_passage, parse_ok, retry_meta = _generate_snap_hyre_blocks(
        config,
        question=question,
        fallback_passage=question_intermediate,
        label="snap_hyre/snap_and_hyre",
    )
    return {
        "source_mode": "snap_hyre",
        "snap_answer": snap_block,
        "snap_letter": _extract_required_final_line_prediction(snap_block, config),
        "snap_and_hyre_raw": raw,
        "snap_hyre_parse_ok": parse_ok,
        "hyde_passage": hyre_passage,
        "hyde_passage_raw": raw,
        "hyde_contains_answer_artifact": _contains_answer_artifact(hyre_passage),
        **retry_meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["rag_hyde", "snap_hyre"])
    parser.add_argument("--provider", required=True)
    parser.add_argument("--dataset", required=True, choices=[
        "barexam", "housing", "legal_rag", "australian", "casehold",
        "musique", "legalbench_scalr",
    ])
    parser.add_argument("--questions", default="full", help="'full' or integer N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--source-filter", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Append missing labels if the output already exists")
    parser.add_argument("--trace-calls", action="store_true")
    parser.add_argument("--trace-events", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trace_calls:
        os.environ["EVAL_TRACE_CALLS"] = "1"
    if args.trace_events:
        os.environ["EVAL_TRACE_EVENTS"] = "1"

    config = EvalConfig(
        mode=args.mode,
        provider=args.provider,
        questions=args.questions,
        seed=args.seed,
        source_filter=args.source_filter,
        dataset=args.dataset,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        tag=args.tag,
    )
    _setup_provider(config)
    questions = load_questions(config)
    if args.sample_start or args.sample_end is not None:
        start = max(0, int(args.sample_start or 0))
        end = None if args.sample_end is None else max(start, int(args.sample_end))
        questions = questions.iloc[start:end].reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing(args.out) if args.resume else set()
    open_mode = "a" if args.resume else "w"
    failures = 0
    consecutive_errors = 0
    wrote = 0

    with args.out.open(open_mode) as f:
        for fallback_i, row in questions.iterrows():
            label = _row_label(row, config, fallback_i=fallback_i)
            if label in done:
                continue
            _reset_llm_call_counter()
            _reset_call_trace()
            _reset_trace_events()
            start_time = time.time()
            error = ""
            payload: dict[str, Any] = {}
            try:
                if args.mode == "rag_hyde":
                    payload = _build_rag_hyde(row, config)
                else:
                    payload = _build_snap_hyre(row, config)
            except Exception as exc:
                failures += 1
                consecutive_errors += 1
                error = str(exc)
                payload = {"source_mode": args.mode}
            else:
                consecutive_errors = 0

            metrics = _get_metrics()
            record: dict[str, Any] = {
                "label": label,
                "idx": _row_idx(row, fallback_i),
                "dataset": args.dataset,
                "mode": args.mode,
                "provider": args.provider,
                "tag": args.tag,
                "elapsed_sec": round(time.time() - start_time, 1),
                "error": error,
                "llm_calls": metrics["count"],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                **payload,
            }
            if args.trace_calls:
                record["call_trace"] = _get_call_trace()
            if args.trace_events:
                record["trace_events"] = _get_trace_events()
                record["trace_schema_version"] = 1
            if _no_silent_fallback_enabled():
                violations = _strict_generation_violations(record, args.mode)
                if violations:
                    raise SystemExit(
                        f"NO_SILENT_FALLBACK blocked generation row {label}: "
                        + "; ".join(violations)
                    )
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            wrote += 1

            status = "ERROR" if error else "OK"
            print(
                f"[{wrote}] {label:<35} {status:<5} "
                f"({record['elapsed_sec']:.1f}s, {metrics['count']} calls)",
                flush=True,
            )
            if consecutive_errors >= 5:
                raise SystemExit(f"Aborting after {consecutive_errors} consecutive generation errors: {error[:200]}")

    print(f"wrote={wrote} failures={failures} out={args.out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
