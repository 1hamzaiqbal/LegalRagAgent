#!/usr/bin/env python3
"""Focused completion audit for the HousingQA state-filter goal."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNOFF = REPO_ROOT / "docs" / "signoff_log.md"
LOGS = REPO_ROOT / "logs"
EXPECTED_ROWS = 6853
REQUIRED_PROVIDERS = ("groq-llama8b", "or-gemma4-26b", "groq-llama70b")
REQUIRED_MODES = ("rag_simple", "rag_hyde", "snap_hyre")
HYRE_MODES = {"rag_hyde", "snap_hyre"}

@dataclass
class CellStatus:
    provider: str
    mode: str
    signed: bool = False
    signed_log: str = ""
    rows: int = 0
    errors: int = 0
    missing_predictions: int = 0
    empty_retrieval: int = 0
    missing_state_filter: int = 0
    cache_misses: int = 0
    doc_cache_misses: int = 0
    hyre_cache_misses: int = 0
    missing_final: int = 0
    fallback_rows: int = 0
    think_rows: int = 0

    @property
    def complete(self) -> bool:
        return self.signed and self.rows == EXPECTED_ROWS and not any(
            [
                self.errors,
                self.missing_predictions,
                self.empty_retrieval,
                self.missing_state_filter,
                self.cache_misses,
                self.doc_cache_misses,
                self.hyre_cache_misses,
                self.missing_final,
                self.fallback_rows,
                self.think_rows,
            ]
        )


def truthy_fallback(row: dict) -> bool:
    falsey_strings = {"", "0", "false", "no", "none", "null", "[]", "{}"}
    for key, value in row.items():
        if "fallback" not in str(key).lower():
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip().lower() not in falsey_strings:
                return True
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
            continue
        if value:
            return True
    return False


def expected_final_line(row: dict) -> str | None:
    predicted = str(row.get("predicted_answer") or row.get("prediction") or "").strip().lower()
    if predicted == "yes":
        return "Answer: Yes"
    if predicted == "no":
        return "Answer: No"
    return None


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def signoff_map() -> dict[tuple[str, str], str]:
    text = SIGNOFF.read_text(errors="replace") if SIGNOFF.exists() else ""
    result: dict[tuple[str, str], str] = {}
    pattern = re.compile(
        r"^\| HousingQA state-filtered \| `(?P<provider>[^`]+)` \| `(?P<mode>[^`]+)` \| `(?P<log>[^`]+)` \|",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        result[(match.group("provider"), match.group("mode"))] = match.group("log")
    return result


def candidate_logs(provider: str, mode: str) -> list[Path]:
    patterns = [
        LOGS / f"eval_{mode}_{provider}_*_housing_*nfull-k5*_detail.jsonl",
    ]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(pattern.parent.glob(pattern.name))
    return sorted(paths)


def summarize_rows(rows: list[dict], provider: str, mode: str) -> CellStatus:
    status = CellStatus(provider=provider, mode=mode, rows=len(rows))
    seen_labels: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        label = str(row.get("label") or row.get("question_id") or len(seen_labels))
        if label in seen_labels:
            continue
        seen_labels.add(label)
        deduped.append(row)
    status.rows = len(deduped)
    for row in deduped:
        status.errors += int(bool(row.get("error")))
        status.missing_predictions += int(not str(row.get("predicted_answer") or "").strip())
        retrieved = row.get("retrieved_ids") or row.get("retrieved_doc_ids") or []
        evidence = row.get("evidence_store") or []
        status.empty_retrieval += int(not retrieved and not evidence)
        where = row.get("retrieval_where") or row.get("where") or {}
        state_filtered = row.get("housing_state_filter") is True or (
            isinstance(where, dict) and bool(str(where.get("state", "")).strip())
        )
        status.missing_state_filter += int(not state_filtered)
        if "retrieval_cache_hit" in row:
            status.cache_misses += int(row.get("retrieval_cache_hit") is not True)
        if "retrieval_doc_cache_hit" in row:
            status.doc_cache_misses += int(row.get("retrieval_doc_cache_hit") is not True)
        if mode in HYRE_MODES and "hyre_cache_hit" in row:
            status.hyre_cache_misses += int(row.get("hyre_cache_hit") is not True)
        final_lines = [
            line.strip()
            for line in str(row.get("final_answer") or "").splitlines()
            if line.strip()
        ]
        status.missing_final += int(not final_lines or final_lines[-1] != expected_final_line(row))
        status.fallback_rows += int(truthy_fallback(row))
        text = "\n".join(str(row.get(key, "")) for key in ("final_answer", "hyde_passage", "snap_answer")).lower()
        status.think_rows += int("<think" in text or "</think" in text)
    return status


def best_partial(provider: str, mode: str) -> CellStatus:
    best = CellStatus(provider=provider, mode=mode)
    best_paths: list[str] = []
    for path in candidate_logs(provider, mode):
        try:
            rows = load_rows(path)
        except Exception:
            continue
        if not rows:
            continue
        if not any(row.get("dataset") == "housing" for row in rows):
            continue
        if not any(row.get("mode") == mode and row.get("provider") == provider for row in rows):
            continue
        if not any(row.get("housing_state_filter") is True for row in rows):
            continue
        best_paths.append(str(path.relative_to(REPO_ROOT)))
    if best_paths:
        combined: dict[str, dict] = {}
        for rel in best_paths:
            for row in load_rows(REPO_ROOT / rel):
                label = str(row.get("label") or row.get("question_id") or len(combined))
                combined[label] = row
        best = summarize_rows(list(combined.values()), provider, mode)
        best.signed_log = f"{len(best_paths)} candidate state-filter logs"
    return best


def audit() -> list[CellStatus]:
    signed = signoff_map()
    cells: list[CellStatus] = []
    for provider in REQUIRED_PROVIDERS:
        for mode in REQUIRED_MODES:
            cell = CellStatus(provider=provider, mode=mode)
            rel_log = signed.get((provider, mode))
            if rel_log:
                path = REPO_ROOT / rel_log
                cell.signed = True
                cell.signed_log = rel_log
                if path.exists():
                    cell = summarize_rows(load_rows(path), provider, mode)
                    cell.signed = True
                    cell.signed_log = rel_log
            else:
                cell = best_partial(provider, mode)
            cells.append(cell)
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    cells = audit()
    print("| Provider | Mode | Signed | Rows | Health | Log |")
    print("|---|---|---:|---:|---|---|")
    complete_count = 0
    for cell in cells:
        complete_count += int(cell.complete)
        health_parts = [
            f"errors={cell.errors}",
            f"missing_pred={cell.missing_predictions}",
            f"empty_ret={cell.empty_retrieval}",
            f"missing_state={cell.missing_state_filter}",
            f"ret_miss={cell.cache_misses}",
            f"doc_miss={cell.doc_cache_misses}",
            f"hyre_miss={cell.hyre_cache_misses}",
            f"missing_final={cell.missing_final}",
            f"fallback={cell.fallback_rows}",
            f"think={cell.think_rows}",
        ]
        print(
            "| {provider} | {mode} | {signed} | {rows}/{expected} | {health} | `{log}` |".format(
                provider=cell.provider,
                mode=cell.mode,
                signed="yes" if cell.signed else "no",
                rows=cell.rows,
                expected=EXPECTED_ROWS,
                health=", ".join(health_parts),
                log=cell.signed_log or "--",
            )
        )
    print(f"\ncomplete={complete_count}/{len(cells)}")
    if complete_count != len(cells) and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
