#!/usr/bin/env python3
"""Build a Speculative-RAG-aligned metrics report from eval detail logs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

import eval_metrics  # type: ignore  # noqa: E402


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_log_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = resolve_path(raw)
        return path.stem, path
    label, path_raw = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid --log label in {raw!r}")
    return label, resolve_path(path_raw)


def logs_from_manifest(path: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    with path.open() as f:
        manifest = json.load(f)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: expected top-level entries list")
    result: list[tuple[str, Path, dict[str, Any]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        detail_log = str(entry.get("detail_log") or "").strip()
        if not label or not detail_log:
            raise SystemExit(f"{path}: manifest entries require label and detail_log")
        result.append((label, resolve_path(detail_log), entry))
    return result


def load_runs(args: argparse.Namespace) -> list[tuple[str, Path, dict[str, Any], dict[str, Any]]]:
    requested: list[tuple[str, Path, dict[str, Any]]] = []
    if args.manifest:
        requested.extend(logs_from_manifest(resolve_path(args.manifest)))
    for raw in args.log or []:
        label, path = parse_log_arg(raw)
        requested.append((label, path, {}))
    if not requested:
        raise SystemExit("Pass at least one --manifest or --log")

    runs: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    for label, path, metadata in requested:
        rows = eval_metrics.load_jsonl(path)
        runs.append((label, path, metadata, eval_metrics.summarize_records(rows, label=label, path=path)))
    return runs


def markdown(title: str, runs: list[tuple[str, Path, dict[str, Any], dict[str, Any]]]) -> str:
    lines = [
        f"# {title}",
        "",
        "Generated from detail JSONL logs. Metrics are offline and do not call an LLM.",
        "",
        "## Speculative-RAG Metric Mapping",
        "",
        "| Speculative RAG metric family | What this report computes now | Gap / caveat |",
        "|---|---|---|",
        "| Answer quality | closed-set accuracy, MuSiQue EM/F1, and free-form gold-answer containment when aliases are logged | Containment is only an automatic proxy; legal open-ended rows still need judge/rubric scoring. |",
        "| Efficiency | average, p50, and p95 latency; LLM calls; input/output token use | Local timings mix API latency and harness overhead, so compare only like-for-like runs. |",
        "| Rationale/context compression | generated pseudo-context tokens versus retrieved evidence tokens | This approximates Speculative RAG rationale-vs-document compression; our logs do not yet separate verifier rationale from HyDE/snap artifacts. |",
        "| Drafting | draft count and speculative-score row coverage | Current modes do not log answer drafts or verifier probabilities, so rhoDraft/rhoSelf-contain/rhoSelf-reflect are not computable yet. |",
        "| Retrieval diagnostics | gold-hit rate, retrieval row rate, empty retrieval, evidence docs/tokens | CaseHOLD gold-hit instrumentation is known untrustworthy in current logs. |",
        "",
        "## Run Matrix",
        "",
        "| Label | Dataset | Mode | N | Acc | EM | F1 | Contains gold | Gold hit | Evid docs/q | Evid tok/q | Gen ctx tok/q | Gen/Evid | Calls/q | Lat avg/p95 | In tok/q | Out tok/q | Drafts/q | Spec score rows |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _path, _metadata, item in runs:
        lines.append(
            "| {label} | {dataset} | {mode} | {n} | {acc} | {em} | {f1} | {contains} | {gold} | {docs} | {evidence_tok} | {gen_tok} | {ratio} | {calls} | {lat}/{p95} | {in_tok} | {out_tok} | {drafts} | {scores} |".format(
                label=label,
                dataset=item["dataset"],
                mode=item["mode"],
                n=item["n"],
                acc=pct(item["accuracy"]),
                em=pct(item["em_rate"]),
                f1=pct(item["avg_f1"]),
                contains=pct(item["contains_gold_rate"]),
                gold=pct(item["gold_hit_rate"]),
                docs=num(item["avg_evidence_docs"], 1),
                evidence_tok=num(item["avg_evidence_tokens"], 0),
                gen_tok=num(item["avg_generated_context_tokens"], 0),
                ratio=num(item["generated_to_evidence_token_ratio"], 2),
                calls=num(item["avg_llm_calls"], 2),
                lat=num(item["avg_latency_sec"], 2),
                p95=num(item["p95_latency_sec"], 2),
                in_tok=num(item["avg_input_tokens"], 0),
                out_tok=num(item["avg_output_tokens"], 0),
                drafts=num(item["avg_draft_count"], 1),
                scores=pct(item["spec_score_row_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Log Provenance",
            "",
            "| Label | Detail log | Hypothesis | Caveat |",
            "|---|---|---|---|",
        ]
    )
    for label, path, metadata, _item in runs:
        rel_path = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        hypothesis = str(metadata.get("hypothesis") or "-")
        caveat = str(metadata.get("known_caveat") or "-")
        lines.append(f"| {label} | `{rel_path}` | {hypothesis} | {caveat} |")

    lines.extend(
        [
            "",
            "## Immediate Wiring Gaps",
            "",
            "- Add explicit `answer_drafts` and `draft_rationales` arrays if we implement a Speculative-RAG arm.",
            "- Store verifier logprob-derived scores only when the backend exposes token logprobs; otherwise log a separate `llm_verifier_vote` field and keep it labeled as a proxy.",
            "- Split generated-context logging into `query_pseudo_context`, `reasoning_trace`, and `verifier_rationale` so compression is not overloaded.",
            "- Repair CaseHOLD gold-option retrieval mapping before interpreting gold-hit or recall numbers.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="JSON manifest with entries containing label and detail_log")
    parser.add_argument("--log", action="append", help="Detail log path, optionally label=path")
    parser.add_argument("--out", required=True, help="Markdown output path")
    parser.add_argument("--title", default="Speculative-RAG-Aligned Metrics Report")
    args = parser.parse_args()

    runs = load_runs(args)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown(args.title, runs))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
