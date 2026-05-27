#!/usr/bin/env python3
"""Build question-only HyDE or Snap-HyRE generation caches without retrieval.

The cache can be used by ``scripts/build_retrieval_cache.py`` for top-k
retrieval diagnostics and by ``eval/eval_harness.py --hyre-cache-path`` for
answer replay. This lets top-k selection happen before the expensive final
answer sweep while keeping generated queries fixed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import BEIR_DATASETS, EvalConfig, load_questions  # noqa: E402
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
    _provider_route_metadata,
    _passage_style_signal_metadata,
    _passage_style_signal_variant,
    _reset_call_trace,
    _reset_llm_call_counter,
    _reset_trace_events,
    _sanitize_intermediate_text,
    _system_prompt,
    _row_label,
    _setup_provider,
    _llm_call,
)


class StrictGenerationViolation(RuntimeError):
    def __init__(self, label: str, record: dict[str, Any], violations: list[str]):
        self.label = label
        self.record = record
        self.violations = violations
        super().__init__(f"{label}: " + "; ".join(violations))


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


STRICT_GENERATION_ANSWER_DATASETS = {
    "barexam",
    "housing",
    "casehold",
    "legalbench_scalr",
    "mas_legal_bench",
    "legal_link_eu",
    "medqa",
}


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
    near_cap = int(os.getenv("EVAL_GENERATION_NEAR_CAP_TOKENS", "1900"))
    if int(record.get("output_tokens") or 0) >= near_cap and not _generation_retry_resolved_near_cap(
        record,
        mode,
        near_cap,
    ):
        violations.append(f"generation_output_tokens_near_cap>={near_cap}")
    max_hyde_chars = int(os.getenv("EVAL_HYDE_MAX_CHARS", "2500"))
    if len(str(record.get("hyde_passage") or "")) > max_hyde_chars:
        violations.append(f"hyde_passage_chars>{max_hyde_chars}")
    if mode in {"snap_hyre", "snap_hyre_exemplar"} and record.get("snap_hyre_parse_ok") is False:
        violations.append("snap_hyre_parse_ok=False")
    if (
        mode in {"snap_hyre", "snap_hyre_exemplar"}
        and str(record.get("dataset") or "") in STRICT_GENERATION_ANSWER_DATASETS
        and not record.get("snap_letter")
    ):
        violations.append("snap_letter missing required final answer line")
    return violations


def _generation_retry_resolved_near_cap(record: dict[str, Any], mode: str, near_cap: int) -> bool:
    """Accept only explicit same-model format repairs that replace a near-cap generation."""
    if mode in {"snap_hyre", "snap_hyre_exemplar"}:
        return _snap_hyre_retry_resolved_near_cap(record, near_cap)
    if mode in {"rag_hyde", "rag_hyde_exemplar"}:
        return _hyde_retry_resolved_near_cap(record)
    return False


def _hyde_retry_resolved_near_cap(record: dict[str, Any]) -> bool:
    if record.get("hyde_format_retry") is not True:
        return False
    if record.get("hyde_format_retry_valid") is not True:
        return False
    if not record.get("hyde_passage"):
        return False
    if record.get("hyde_contains_answer_artifact") is True:
        return False
    if record.get("hyde_used_fallback") is True:
        return False
    return True


def _snap_hyre_retry_resolved_near_cap(record: dict[str, Any], near_cap: int) -> bool:
    """Accept only a logged same-model retry that produced the usable Snap-HyRE text."""
    if record.get("snap_hyre_format_retry") is not True:
        return False
    if record.get("snap_hyre_parse_ok") is not True:
        return False
    requires_snap_letter = str(record.get("dataset") or "") in STRICT_GENERATION_ANSWER_DATASETS
    if (requires_snap_letter and not record.get("snap_letter")) or not record.get("hyde_passage"):
        return False
    if record.get("hyde_contains_answer_artifact") is True:
        return False
    retry_tokens = record.get("snap_hyre_format_retry_output_tokens")
    if retry_tokens is None:
        return False
    try:
        return int(retry_tokens) < near_cap
    except (TypeError, ValueError):
        return False


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _max_hyde_chars() -> int:
    return int(os.getenv("EVAL_HYDE_MAX_CHARS", "2500"))


def _retry_excerpt(text: str, limit: int = 4000) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    head = max(0, limit - 700)
    return text[:head] + "\n\n[... previous output truncated for retry prompt ...]\n\n" + text[-500:]


def _repair_hyde_payload(row: Any, config: EvalConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Run a logged same-model format repair for malformed/overlong HyDE text."""
    reasons: list[str] = []
    hyde_passage = str(payload.get("hyde_passage") or "")
    if not hyde_passage:
        reasons.append("missing_hyde_passage")
    if payload.get("hyde_used_fallback") is True:
        reasons.append("hyde_used_fallback")
    if payload.get("hyde_contains_answer_artifact") is True:
        reasons.append("passage_contains_answer_artifact")
    max_chars = _max_hyde_chars()
    if max_chars > 0 and len(hyde_passage) > max_chars:
        reasons.append(f"hyde_passage_chars>{max_chars}")
    if not reasons or not _env_truthy("EVAL_GENERATION_FORMAT_RETRY"):
        return payload

    question_intermediate = _fmt_intermediate(row, config)
    previous = str(payload.get("hyde_passage_raw") or payload.get("hyde_passage") or "")
    repair_user = (
        "## Original Scenario\n"
        f"{question_intermediate}\n\n"
        "## Previous Malformed Passage\n"
        f"{_retry_excerpt(previous)}\n\n"
        "## Repair Instruction\n"
        "Rewrite the previous passage as a valid HyDE retrieval passage. Preserve the same legal "
        "issue and doctrine, but return only 2-3 concise sentences in neutral legal reference "
        "style. Maximum 120 words. Do not repeat phrases or sentences. Do not include answer "
        "labels, choice letters, markdown, bullets, or an `Answer:` line."
    )
    retry_raw = _llm_call(
        _system_prompt(config, "hyde"),
        repair_user,
        label="hyde/generate/format_retry",
    )
    retry_text = _sanitize_intermediate_text(retry_raw, fallback="")
    retry_contains_answer = _contains_answer_artifact(retry_text)
    repaired = {
        **payload,
        "hyde_passage": retry_text,
        "hyde_passage_raw": retry_raw,
        "hyde_contains_answer_artifact": retry_contains_answer,
        "hyde_used_fallback": not bool(retry_text),
        "hyde_parse_ok": bool(retry_text),
        "hyde_format_retry": True,
        "hyde_format_retry_reason": ",".join(reasons),
        "hyde_format_retry_reasons": reasons,
        "hyde_format_retry_valid": (
            bool(retry_text)
            and not retry_contains_answer
            and (max_chars <= 0 or len(retry_text) <= max_chars)
        ),
        "hyde_passage_before_format_retry": payload.get("hyde_passage", ""),
        "hyde_raw_before_format_retry": payload.get("hyde_passage_raw", ""),
    }
    return repaired


def _build_rag_hyde(row, config: EvalConfig) -> dict[str, Any]:
    question_intermediate = _fmt_intermediate(row, config)
    use_style_signal = config.mode == "rag_hyde_exemplar"
    hyde = _generate_hyde(
        config,
        "hyde",
        _question_only_hyde_user(question_intermediate, config=config, use_style_signal=use_style_signal),
        label="hyde_exemplar/generate" if use_style_signal else "hyde/generate",
        fallback=question_intermediate,
    )
    payload = {
        "source_mode": config.mode,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "hyde_used_fallback": hyde.get("used_fallback", False),
        "hyde_parse_ok": bool(hyde["text"]),
        "passage_style_signal_used": use_style_signal,
        **(_passage_style_signal_metadata(config) if use_style_signal else {}),
    }
    return _repair_hyde_payload(row, config, payload)


def _build_snap_hyre(row, config: EvalConfig) -> dict[str, Any]:
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    use_style_signal = config.mode == "snap_hyre_exemplar"
    raw, snap_block, hyre_passage, parse_ok, retry_meta = _generate_snap_hyre_blocks(
        config,
        question=question,
        fallback_passage=question_intermediate,
        label="snap_hyre_exemplar/snap_and_hyre" if use_style_signal else "snap_hyre/snap_and_hyre",
        use_style_signal=use_style_signal,
    )
    return {
        "source_mode": config.mode,
        "snap_answer": snap_block,
        "snap_letter": _extract_required_final_line_prediction(snap_block, config),
        "snap_and_hyre_raw": raw,
        "snap_hyre_parse_ok": parse_ok,
        "hyde_passage": hyre_passage,
        "hyde_passage_raw": raw,
        "hyde_contains_answer_artifact": _contains_answer_artifact(hyre_passage),
        "passage_style_signal_used": use_style_signal,
        **(_passage_style_signal_metadata(config) if use_style_signal else {}),
        **retry_meta,
    }


def _is_openrouter_provider(provider: str) -> bool:
    return str(provider or "").strip().lower().startswith("or-")


def _resolved_concurrency(config: EvalConfig, provider: str) -> int:
    configured = int(getattr(config, "concurrency", 0) or 0)
    if configured <= 0:
        raw = os.getenv("EVAL_CONCURRENCY", "").strip()
        if raw:
            try:
                configured = int(raw)
            except ValueError as exc:
                raise SystemExit(f"EVAL_CONCURRENCY must be an integer, got {raw!r}") from exc
    if configured <= 0:
        configured = 8 if _is_openrouter_provider(provider) else 1
    return max(1, configured)


def _build_one_generation_record(
    fallback_i: int,
    row: Any,
    *,
    config: EvalConfig,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any], bool]:
    label = _row_label(row, config, fallback_i=fallback_i)
    _reset_llm_call_counter()
    _reset_call_trace()
    _reset_trace_events()
    start_time = time.time()
    error = ""
    payload: dict[str, Any] = {}
    failed = False
    try:
        if args.mode in {"rag_hyde", "rag_hyde_exemplar"}:
            payload = _build_rag_hyde(row, config)
        else:
            payload = _build_snap_hyre(row, config)
    except Exception as exc:
        failed = True
        error = str(exc)
        payload = {"source_mode": args.mode}

    metrics = _get_metrics()
    record: dict[str, Any] = {
        "label": label,
        "idx": _row_idx(row, fallback_i),
        "dataset": args.dataset,
        "mode": args.mode,
        "provider": args.provider,
        "provider_route": _provider_route_metadata(),
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
            raise StrictGenerationViolation(label, record, violations)
    return fallback_i, record, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=[
        "rag_hyde", "snap_hyre", "rag_hyde_exemplar", "snap_hyre_exemplar",
    ])
    parser.add_argument("--provider", required=True)
    parser.add_argument("--dataset", required=True, choices=[
        "barexam", "housing", "legal_rag", "legal_rag_bench", "mas_legal_bench", "legal_link_eu", "australian", "casehold",
        "musique", "hotpotqa", "legalbench_scalr", "medqa", *BEIR_DATASETS.keys(),
    ])
    parser.add_argument("--questions", default="full", help="'full' or integer N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--source-filter", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--passage-style-variant", default="",
                        help="Probe-only exemplar style variant: single or multi3")
    parser.add_argument("--exclude-gold-ids", default="",
                        help="Comma/whitespace-separated gold ids to exclude from question loading")
    parser.add_argument("--exclude-gold-ids-path", default="",
                        help="JSON/TXT file of gold ids to exclude from question loading")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Append missing labels if the output already exists")
    parser.add_argument("--trace-calls", action="store_true")
    parser.add_argument("--trace-events", action="store_true")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="OpenRouter worker count; 0 uses EVAL_CONCURRENCY or the provider default")
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
        passage_style_variant=args.passage_style_variant,
        exclude_gold_ids=args.exclude_gold_ids,
        exclude_gold_ids_path=args.exclude_gold_ids_path,
        concurrency=args.concurrency,
    )
    if config.mode in {"rag_hyde_exemplar", "snap_hyre_exemplar"} and args.passage_style_variant:
        os.environ["EVAL_PASSAGE_STYLE_VARIANT"] = _passage_style_signal_variant(config)
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
    wrote = 0

    pending_rows = [
        (fallback_i, row)
        for fallback_i, row in questions.iterrows()
        if _row_label(row, config, fallback_i=fallback_i) not in done
    ]
    concurrency = _resolved_concurrency(config, args.provider)
    use_parallel = _is_openrouter_provider(args.provider) and concurrency > 1
    if use_parallel:
        os.environ["EVAL_CONCURRENCY"] = str(concurrency)
    print(f"[concurrency] provider={args.provider} workers={concurrency if use_parallel else 1}")

    records_by_order: dict[int, tuple[dict[str, Any], bool]] = {}
    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _build_one_generation_record,
                    fallback_i,
                    row,
                    config=config,
                    args=args,
                ): fallback_i
                for fallback_i, row in pending_rows
            }
            error_count = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    order_i, record, failed = future.result()
                except StrictGenerationViolation as exc:
                    for pending in futures:
                        pending.cancel()
                    raise SystemExit(
                        f"NO_SILENT_FALLBACK blocked generation row {exc.label}: "
                        + "; ".join(exc.violations)
                    ) from exc
                records_by_order[order_i] = (record, failed)
                if failed:
                    error_count += 1
                    if error_count >= 5:
                        for pending in futures:
                            pending.cancel()
                        raise SystemExit(
                            "Aborting after 5 generation errors in parallel run: "
                            f"{str(record.get('error'))[:200]}"
                        )
    else:
        consecutive_errors = 0
        for fallback_i, row in pending_rows:
            try:
                order_i, record, failed = _build_one_generation_record(
                    fallback_i,
                    row,
                    config=config,
                    args=args,
                )
            except StrictGenerationViolation as exc:
                raise SystemExit(
                    f"NO_SILENT_FALLBACK blocked generation row {exc.label}: "
                    + "; ".join(exc.violations)
                ) from exc
            records_by_order[order_i] = (record, failed)
            if failed:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    raise SystemExit(
                        f"Aborting after {consecutive_errors} consecutive generation errors: "
                        f"{str(record.get('error'))[:200]}"
                    )
            else:
                consecutive_errors = 0

    with args.out.open(open_mode) as f:
        for order_i in sorted(records_by_order):
            record, failed = records_by_order[order_i]
            if failed:
                failures += 1
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            wrote += 1
            status = "ERROR" if record.get("error") else "OK"
            print(
                f"[{wrote}] {record['label']:<35} {status:<5} "
                f"({record['elapsed_sec']:.1f}s, {record['llm_calls']} calls)",
                flush=True,
            )

    print(f"wrote={wrote} failures={failures} out={args.out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
