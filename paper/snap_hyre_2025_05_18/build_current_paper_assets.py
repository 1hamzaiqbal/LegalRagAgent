#!/usr/bin/env python3
"""Build paper-local tables and figures from signed Snap-HyRE rows.

The script intentionally treats docs/signoff_log.md as the paper-facing result
gate. Signed accuracy and retrieval numbers in the generated artifacts below
come from signoff lines.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PAPER = Path(__file__).resolve().parent
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "docs" / "signoff_log.md").exists():
            return candidate
    raise FileNotFoundError("Could not find docs/signoff_log.md above paper folder")


ROOT = find_repo_root(PAPER)
SIGNOFF = ROOT / "docs" / "signoff_log.md"
EXPERIMENTS = ROOT / "logs" / "experiments.jsonl"

DATASETS = ["BarExamQA", "HousingQA"]
ARCHIVED_DATASETS = ["Legal-Link-EU", "MASLegalBench"]
PROVIDERS = ["groq-llama8b", "or-gemma4-26b", "groq-llama70b"]
MODES = [
    "llm_only",
    "rag_simple",
    "rag_hyde",
    "snap_hyre",
    "golden_passage",
    "golden_plus_neighbors",
    "rag_rewrite",
]
MAIN_RETRIEVAL_MODES = ["rag_simple", "rag_hyde", "snap_hyre"]
RETRIEVAL_MODES = ["rag_simple", "rag_hyde", "snap_hyre", "rag_rewrite"]
HOUSING_STATE_FILTER_REQUIRED_MODES = {"rag_simple", "rag_hyde", "snap_hyre", "rag_rewrite", "golden_plus_neighbors"}
UNSIGNED_STATE_FILTER_ROWS = set()
UNSIGNED_STATE_FILTER_ACCURACY = {}
UNSIGNED_STATE_FILTER_RETRIEVAL = {}

DATASET_LABEL = {
    "BarExamQA": "BarExamQA",
    "HousingQA": "HousingQA",
    "Legal-Link-EU": "Legal-Link-EU",
    "MASLegalBench": "MASLegalBench",
}
PROVIDER_LABEL = {
    "groq-llama8b": "Llama 3.1 8B",
    "or-gemma4-26b": "Gemma 4 26B",
    "groq-llama70b": "Llama 3.3 70B",
}
MODE_LABEL = {
    "llm_only": "LLM",
    "rag_simple": "Raw question RAG",
    "rag_hyde": "HyDE",
    "snap_hyre": "Snap-HyRE (ours)",
    "golden_passage": "Gold",
    "golden_plus_neighbors": "Gold + neighbors",
    "rag_rewrite": "Rewrite",
}
CALLS = {
    "llm_only": 1,
    "rag_simple": 1,
    "rag_hyde": 2,
    "snap_hyre": 2,
    "golden_passage": 1,
    "golden_plus_neighbors": 1,
    "rag_rewrite": 2,
}
COLORS = {
    "llm_only": "#5f6f7f",
    "rag_simple": "#2f6f9f",
    "rag_hyde": "#48a9a6",
    "snap_hyre": "#d4942f",
    "golden_passage": "#7f5aa2",
    "golden_plus_neighbors": "#9c7c38",
    "rag_rewrite": "#c65f5f",
}

EXEMPLAR_PROBES = [
    {
        "dataset": "BarExamQA",
        "metric": "Hit@5/MRR@5",
        "kind": "gold",
        "raw": "caches/retrieval/full/barexam_q20_seed42_raw_question_k10.jsonl",
        "snap_hyre": "caches/retrieval/probes/barexam_q20_seed42_or-gemma4-26b_snap_hyre_k10.jsonl",
        "exemplar": "caches/retrieval/probes/barexam_q20_seed42_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl",
    },
    {
        "dataset": "HousingQA",
        "metric": "Hit@5/MRR@5",
        "kind": "gold",
        "raw": "caches/retrieval/full/housing_q20_seed42_raw_question_k10.jsonl",
        "snap_hyre": "caches/retrieval/probes/housing_q20_seed42_or-gemma4-26b_snap_hyre_k10.jsonl",
        "exemplar": "caches/retrieval/probes/housing_q20_seed42_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl",
    },
]

TOPK_KS = [1, 3, 5, 10]
TOPK_RETRIEVAL_SPECS = [
    {
        "dataset": "BarExamQA",
        "provider": "shared",
        "model": "Shared",
        "mode": "rag_simple",
        "path": "caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl",
    },
    {
        "dataset": "BarExamQA",
        "provider": "groq-llama8b",
        "model": PROVIDER_LABEL["groq-llama8b"],
        "mode": "rag_hyde",
        "path": "caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl",
    },
    {
        "dataset": "BarExamQA",
        "provider": "groq-llama8b",
        "model": PROVIDER_LABEL["groq-llama8b"],
        "mode": "snap_hyre",
        "path": "caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl",
    },
    {
        "dataset": "BarExamQA",
        "provider": "or-gemma4-26b",
        "model": PROVIDER_LABEL["or-gemma4-26b"],
        "mode": "rag_hyde",
        "path": "caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl",
    },
    {
        "dataset": "BarExamQA",
        "provider": "or-gemma4-26b",
        "model": PROVIDER_LABEL["or-gemma4-26b"],
        "mode": "snap_hyre",
        "path": "caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl",
    },
    {
        "dataset": "BarExamQA",
        "provider": "groq-llama70b",
        "model": PROVIDER_LABEL["groq-llama70b"],
        "mode": "rag_hyde",
        "path": "caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl",
    },
    {
        "dataset": "BarExamQA",
        "provider": "groq-llama70b",
        "model": PROVIDER_LABEL["groq-llama70b"],
        "mode": "snap_hyre",
        "path": "caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl",
    },
    {
        "dataset": "HousingQA",
        "provider": "shared",
        "model": "State-filtered shared",
        "mode": "rag_simple",
        "path": "caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl",
    },
    {
        "dataset": "HousingQA",
        "provider": "groq-llama8b",
        "model": "State-filtered " + PROVIDER_LABEL["groq-llama8b"],
        "mode": "rag_hyde",
        "path": "caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_rag_hyde_k10.jsonl",
    },
    {
        "dataset": "HousingQA",
        "provider": "groq-llama8b",
        "model": "State-filtered " + PROVIDER_LABEL["groq-llama8b"],
        "mode": "snap_hyre",
        "path": "caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl",
    },
    {
        "dataset": "HousingQA",
        "provider": "groq-llama70b",
        "model": "State-filtered " + PROVIDER_LABEL["groq-llama70b"],
        "mode": "rag_hyde",
        "path": "caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_rag_hyde_k10.jsonl",
    },
    {
        "dataset": "HousingQA",
        "provider": "groq-llama70b",
        "model": "State-filtered " + PROVIDER_LABEL["groq-llama70b"],
        "mode": "snap_hyre",
        "path": "caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl",
    },
]

HOUSING_METADATA_FILTER_SPECS = [
    {
        "scope": "National corpus",
        "method": "Raw question RAG",
        "path": "caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl",
    },
    {
        "scope": "Jurisdiction metadata filter",
        "method": "Raw question RAG",
        "path": "caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl",
    },
]


@dataclass
class SignedRow:
    dataset: str
    provider: str
    mode: str
    correct: int | None = None
    total: int | None = None
    accuracy: float | None = None
    hit5: float | None = None
    mrr5: float | None = None
    signoff: str = ""
    health: str = ""
    detail_log: str = ""
    source_rows: int = 0


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.{digits}f}"


def pct_cell(value: float | None, bold: bool = False, digits: int = 1) -> str:
    text = pct(value, digits)
    if bold and value is not None:
        return rf"\textbf{{{text}}}"
    return text


def pct_from_percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def pp_delta(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value:+.{digits}f}"


def pp_delta_with_n(values: list[float], digits: int = 1) -> str:
    value = mean(values)
    if value is None:
        return "--"
    return f"{value:+.{digits}f} (n={len(values)})"


def num(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def mean(values: list[float]) -> float | None:
    vals = [value for value in values if value is not None and not math.isnan(value)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def is_expected(dataset: str, provider: str, mode: str) -> bool:
    if dataset == "MASLegalBench" and mode in {"golden_passage", "golden_plus_neighbors"}:
        return False
    return dataset in DATASETS and provider in PROVIDERS and mode in MODES


def is_unfiltered_housing_provenance(dataset: str, mode: str) -> bool:
    return dataset == "HousingQA" and mode in HOUSING_STATE_FILTER_REQUIRED_MODES


def is_state_filtered_housing_row(row: SignedRow) -> bool:
    if row.dataset != "HousingQA" or row.mode not in HOUSING_STATE_FILTER_REQUIRED_MODES:
        return False
    text = f"{row.signoff} {row.health}".lower()
    return (
        "| housingqa state-filtered |" in text
        or "housing_state_filter=true" in text
        or 'retrieval_where={"state"' in text
        or "comprehensive-clean-statefilter" in text
        or "comprehensive-cite-statefilter" in text
    )


def is_main_signed_row(row: SignedRow) -> bool:
    if row.dataset == "HousingQA" and row.mode in HOUSING_STATE_FILTER_REQUIRED_MODES:
        return is_state_filtered_housing_row(row)
    return is_main_signed_mode(row.dataset, row.mode)


def is_main_signed_mode(dataset: str, mode: str) -> bool:
    return not is_unfiltered_housing_provenance(dataset, mode)


def star_if_provenance(text: str, dataset: str, mode: str) -> str:
    if text in {"--", "n/a"}:
        return text
    if is_unfiltered_housing_provenance(dataset, mode):
        return text + r"$^{*}$"
    return text


def is_unsigned_state_filter_row(dataset: str, provider: str, mode: str) -> bool:
    return (dataset, provider, mode) in UNSIGNED_STATE_FILTER_ROWS


def mark_answer_cell(text: str, dataset: str, provider: str, mode: str) -> str:
    if text == "n/a":
        return text
    if text == "--":
        return r"unsigned$^{\dagger}$" if is_unsigned_state_filter_row(dataset, provider, mode) else text
    marks = []
    if is_unfiltered_housing_provenance(dataset, mode):
        marks.append("*")
    if is_unsigned_state_filter_row(dataset, provider, mode):
        marks.append(r"\dagger")
    if marks:
        return text + "$^{" + ",".join(marks) + "}$"
    return text


def main_accuracy_value(
    rows: dict[tuple[str, str, str], "SignedRow"], dataset: str, provider: str, mode: str
) -> tuple[float | None, str]:
    """Return main-paper accuracy and status for a dataset/method cell.

    Status is "signed", "unsigned", or "missing". Unfiltered Housing retrieval
    rows are excluded from the main display unless a state-filtered value is
    explicitly tracked as unsigned context.
    """
    key = (dataset, provider, mode)
    if key in UNSIGNED_STATE_FILTER_ACCURACY:
        return UNSIGNED_STATE_FILTER_ACCURACY[key], "unsigned"
    row = rows.get(key)
    if row and row.accuracy is not None:
        if is_unfiltered_housing_provenance(dataset, mode) and not is_state_filtered_housing_row(row):
            return None, "missing"
        return row.accuracy, "signed"
    if is_unfiltered_housing_provenance(dataset, mode):
        return None, "missing"
    return None, "missing"


def main_retrieval_value(
    retrieval: dict[tuple[str, str, str], "SignedRow"], dataset: str, provider: str, mode: str
) -> tuple[float | None, float | None, str]:
    key = (dataset, provider, mode)
    if key in UNSIGNED_STATE_FILTER_RETRIEVAL:
        hit, mrr = UNSIGNED_STATE_FILTER_RETRIEVAL[key]
        return hit, mrr, "unsigned"
    row = retrieval.get(key)
    if row and row.hit5 is not None:
        if is_unfiltered_housing_provenance(dataset, mode) and not is_state_filtered_housing_row(row):
            return None, None, "missing"
        return row.hit5, row.mrr5, "signed"
    if is_unfiltered_housing_provenance(dataset, mode):
        return None, None, "missing"
    return None, None, "missing"


def fmt_main_percent(value: float | None, status: str = "signed", bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{100 * value:.1f}"
    if bold and status == "signed":
        text = rf"\textbf{{{text}}}"
    if status == "unsigned":
        text += r"$^{\dagger}$"
    return text


def fmt_main_metric(value: float | None, status: str = "signed") -> str:
    if value is None:
        return "--"
    text = f"{100 * value:.1f}"
    if status == "unsigned":
        text += r"$^{\dagger}$"
    return text


def mean_with_status(items: list[tuple[float | None, str]]) -> tuple[float | None, str]:
    vals = [(value, status) for value, status in items if value is not None]
    if not vals:
        return None, "missing"
    status = "unsigned" if any(status == "unsigned" for _, status in vals) else "signed"
    return sum(value for value, _ in vals) / len(vals), status


def snap_raw_answer_delta_cell(rows: dict[tuple[str, str, str], "SignedRow"], dataset: str, provider: str) -> str:
    raw = accuracy_value(rows, dataset, provider, "rag_simple")
    snap = accuracy_value(rows, dataset, provider, "snap_hyre")
    if raw is None or snap is None:
        return "--"
    text = f"{100 * (snap - raw):+.1f}"
    return star_if_provenance(text, dataset, "snap_hyre")


def extract_hit_mrr(text: str) -> tuple[float | None, float | None]:
    hit = None
    mrr = None
    hit_match = re.search(r"(?:Hit@5|same-source@5)\s+([0-9]*\.?[0-9]+)", text)
    if hit_match:
        hit = float(hit_match.group(1))
    mrr_match = re.search(r"MRR@5\s+([0-9]*\.?[0-9]+)", text)
    if mrr_match:
        mrr = float(mrr_match.group(1))
    return hit, mrr


def markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def clean_signoff_label(label: str) -> str:
    label = re.sub(r"^[^\w-]+", "", label).strip()
    return label or "SIGNED"


def paper_caveat_categories(row: SignedRow) -> list[str]:
    text = row.signoff.upper()
    categories = []
    if "SOURCE-PROXY" in text:
        categories.append("Source-document retrieval proxy")
    if "SAME-MODEL-ROUTE" in text or "SAME-ROUTE" in text or "ROUTE-RECOVERY" in text or "TRANSIENT" in text:
        categories.append("Endpoint-identity note")
    if "REPAIR" in text or "MERGED" in text or "RERUN" in text or "TPM" in text:
        categories.append("Segmented-evaluation note")
    if "RETRY" in text:
        categories.append("Format-validation note")
    if "NEAR-CAP" in text:
        categories.append("Output-length note")
    if "CE-MAX" in text or "POSTRUN" in text or "LOGGING" in text or "TRACE" in text:
        categories.append("Auxiliary-metadata note")
    if not categories:
        categories.append("Standard row")
    return categories


def parse_signoff() -> tuple[dict[tuple[str, str, str], SignedRow], dict[tuple[str, str, str], SignedRow]]:
    rows: dict[tuple[str, str, str], SignedRow] = {}
    retrieval: dict[tuple[str, str, str], SignedRow] = {}
    dataset_re = "|".join(re.escape(d) for d in DATASETS)
    provider_re = r"(groq-llama8b|or-gemma4-26b|groq-llama70b)"
    mode_re = r"(llm_only|rag_simple|rag_hyde|snap_hyre|golden_passage|golden_plus_neighbors|rag_rewrite)"

    for line in SIGNOFF.read_text().splitlines():
        if not line.startswith("| "):
            continue
        if line.startswith("| Dataset") or line.startswith("|---"):
            continue
        if not re.search(dataset_re, line):
            continue

        dataset_match = re.search(dataset_re, line)
        provider_match = re.search(rf"`{provider_re}`", line)
        mode_match = re.search(rf"`{mode_re}`", line)
        if not (dataset_match and provider_match and mode_match):
            continue
        dataset = dataset_match.group(0)
        provider = provider_match.group(1)
        mode = mode_match.group(1)
        if not is_expected(dataset, provider, mode):
            continue

        cells = markdown_cells(line)
        signoff_cell = clean_signoff_label(cells[-1]) if cells else "SIGNED"
        detail_match = re.search(r"`(logs/[^`]+detail\.jsonl)`", line)
        detail_log = detail_match.group(1) if detail_match else ""
        hit, mrr = extract_hit_mrr(line)
        key = (dataset, provider, mode)
        rows_match = re.search(r"(?:rows|generation rows)\s+(\d+)(?:/(\d+))?", line)
        source_rows = int(rows_match.group(2) or rows_match.group(1)) if rows_match else 0
        if hit is not None or mrr is not None:
            old = retrieval.get(key)
            if old is None or source_rows >= old.source_rows:
                retrieval[key] = SignedRow(
                    dataset=dataset,
                    provider=provider,
                    mode=mode,
                    hit5=hit if hit is not None else (old.hit5 if old else None),
                    mrr5=mrr if mrr is not None else (old.mrr5 if old else None),
                    health=line,
                    detail_log=detail_log,
                    source_rows=source_rows,
                )

        acc_match = re.search(r"(\d+)/(\d+)\s*=\s*([0-9]+(?:\.[0-9]+)?)%", line)
        if not acc_match:
            continue
        total = int(acc_match.group(2))
        if key in rows and rows[key].total and total < rows[key].total:
            continue
        row = SignedRow(
            dataset=dataset,
            provider=provider,
            mode=mode,
            correct=int(acc_match.group(1)),
            total=total,
            accuracy=float(acc_match.group(3)) / 100.0,
            hit5=hit,
            mrr5=mrr,
            signoff=signoff_cell,
            health=line,
            detail_log=detail_log,
            source_rows=total,
        )
        rows[key] = row
        old = retrieval.get(key)
        if old is None or row.source_rows >= old.source_rows:
            retrieval[key] = SignedRow(
                dataset=dataset,
                provider=provider,
                mode=mode,
                hit5=hit if hit is not None else (old.hit5 if old else None),
                mrr5=mrr if mrr is not None else (old.mrr5 if old else None),
                health=line,
                detail_log=detail_log,
                source_rows=row.source_rows,
            )
    return rows, retrieval


def write_csv(rows: dict[tuple[str, str, str], SignedRow], retrieval: dict[tuple[str, str, str], SignedRow]) -> None:
    out = PAPER / "current_audited_rows.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "provider", "mode", "correct", "total", "accuracy", "hit5", "mrr5", "detail_log", "signoff"])
        for key in sorted(set(rows) | set(retrieval)):
            row = rows.get(key)
            ret = retrieval.get(key)
            writer.writerow(
                [
                    key[0],
                    key[1],
                    key[2],
                    "" if row is None or row.correct is None else row.correct,
                    "" if row is None or row.total is None else row.total,
                    "" if row is None or row.accuracy is None else f"{row.accuracy:.6f}",
                    "" if ret is None or ret.hit5 is None else f"{ret.hit5:.6f}",
                    "" if ret is None or ret.mrr5 is None else f"{ret.mrr5:.6f}",
                    "" if row is None else row.detail_log,
                    "" if row is None else row.signoff,
                ]
            )


def table_answer_matrix(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main answer accuracy (\%). Bold marks the best non-oracle row within each dataset/model slice; averages are descriptive over available cells.}",
        r"\label{tab:answer_matrix}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & BarExam 8B & 26B & 70B & Avg. & Housing 8B & 26B & 70B & Avg. \\",
        r"\midrule",
    ]
    display_modes = ["llm_only", "rag_simple", "rag_hyde", "snap_hyre", "golden_passage"]
    for mode in display_modes:
        cells = []
        for dataset in ["BarExamQA", "HousingQA"]:
            best_by_provider: dict[str, float] = {}
            for provider in PROVIDERS:
                signed_values = []
                for candidate in ["llm_only", "rag_simple", "rag_hyde", "snap_hyre"]:
                    value, status = main_accuracy_value(rows, dataset, provider, candidate)
                    if value is not None and status == "signed":
                        signed_values.append(value)
                if signed_values:
                    best_by_provider[provider] = max(signed_values)
            dataset_items = []
            for provider in PROVIDERS:
                value, status = main_accuracy_value(rows, dataset, provider, mode)
                bold = (
                    mode in {"llm_only", "rag_simple", "rag_hyde", "snap_hyre"}
                    and status == "signed"
                    and value is not None
                    and provider in best_by_provider
                    and abs(value - best_by_provider[provider]) < 1e-12
                )
                cells.append(fmt_main_percent(value, status, bold=bold))
                dataset_items.append((value, status))
            avg, avg_status = mean_with_status(dataset_items)
            cells.append(fmt_main_percent(avg, avg_status, bold=False))
        lines.append(f"{tex_escape(MODE_LABEL[mode])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TABLES / "current_answer_matrix.tex").write_text("\n".join(lines) + "\n")


def table_retrieval_matrix(retrieval: dict[tuple[str, str, str], SignedRow]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evidence exposure at $k=5$ (\%). Generated-query rows average available model-specific generators; HousingQA uses state-filtered retrieval.}",
        r"\label{tab:retrieval_matrix}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & BarExam Hit@5 & BarExam MRR@5 & Housing Hit@5 & Housing MRR@5 \\",
        r"\midrule",
    ]
    for mode in MAIN_RETRIEVAL_MODES:
        row_cells = []
        for dataset in ["BarExamQA", "HousingQA"]:
            hit_items = []
            mrr_items = []
            for provider in PROVIDERS:
                hit, mrr, status = main_retrieval_value(retrieval, dataset, provider, mode)
                if hit is not None:
                    hit_items.append((hit, status))
                if mrr is not None:
                    mrr_items.append((mrr, status))
            hit_avg, hit_status = mean_with_status(hit_items)
            mrr_avg, mrr_status = mean_with_status(mrr_items)
            row_cells.append(fmt_main_metric(hit_avg, hit_status))
            row_cells.append(fmt_main_metric(mrr_avg, mrr_status))
        lines.append(f"{tex_escape(MODE_LABEL[mode])} & " + " & ".join(row_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TABLES / "current_retrieval_matrix.tex").write_text("\n".join(lines) + "\n")


def table_snap_deltas(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Snap-HyRE answer deltas against raw question RAG for the main datasets. Positive values mean the fixed Snap-HyRE row improves final answer accuracy over \method{rag\_simple}.}",
        r"\label{tab:snap_deltas}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Model & Raw question RAG & Snap-HyRE & $\Delta$ pp \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for provider in PROVIDERS:
            raw_value, raw_status = main_accuracy_value(rows, dataset, provider, "rag_simple")
            snap_value, snap_status = main_accuracy_value(rows, dataset, provider, "snap_hyre")
            if raw_value is not None and snap_value is not None:
                delta = 100 * (snap_value - raw_value)
                status = "unsigned" if raw_status == "unsigned" or snap_status == "unsigned" else "signed"
                suffix = r"$^{\dagger}$" if status == "unsigned" else ""
                lines.append(
                    f"{tex_escape(dataset)} & {tex_escape(PROVIDER_LABEL[provider])} & "
                    f"{fmt_main_percent(raw_value, raw_status)} & "
                    f"{fmt_main_percent(snap_value, snap_status)} & "
                    f"{delta:+.1f}{suffix} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "current_snap_deltas.tex").write_text("\n".join(lines) + "\n")


def accuracy_value(rows: dict[tuple[str, str, str], SignedRow], dataset: str, provider: str, mode: str) -> float | None:
    row = rows.get((dataset, provider, mode))
    return row.accuracy if row and row.accuracy is not None else None


def signed_delta_values(
    rows: dict[tuple[str, str, str], SignedRow], dataset: str, mode: str, base_mode: str
) -> list[float]:
    deltas = []
    for provider in PROVIDERS:
        value = accuracy_value(rows, dataset, provider, mode)
        base = accuracy_value(rows, dataset, provider, base_mode)
        if value is not None and base is not None:
            deltas.append(100 * (value - base))
    return deltas


def best_non_oracle(rows: dict[tuple[str, str, str], SignedRow], dataset: str) -> str:
    best = None
    for provider in PROVIDERS:
        for mode in ["llm_only", "rag_simple", "rag_hyde", "snap_hyre", "rag_rewrite"]:
            value = accuracy_value(rows, dataset, provider, mode)
            if value is None:
                continue
            if best is None or value > best[2]:
                best = (provider, mode, value)
    if best is None:
        return "--"
    provider, mode, value = best
    return f"{tex_escape(MODE_LABEL[mode])}, {tex_escape(PROVIDER_LABEL[provider])} ({100 * value:.1f})"


def table_result_anatomy(
    rows: dict[tuple[str, str, str], SignedRow], retrieval: dict[tuple[str, str, str], SignedRow]
) -> None:
    raw_story = {
        "BarExamQA": "Very low raw evidence exposure; generated legal text supplies missing vocabulary.",
        "HousingQA": "Jurisdiction must be fixed before retrieval methods are compared; available state-filtered rows favor raw questions.",
    }
    interpretation = {
        "BarExamQA": "Snap-HyRE clearest here.",
        "HousingQA": "Jurisdiction scope changes the comparison.",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Dataset-level result anatomy. Deltas are mean percentage-point differences over included row pairs where both methods exist.}",
        r"\label{tab:result_anatomy}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{lYcccY}",
        r"\toprule",
        r"Dataset & Raw evidence regime & Snap vs. raw pp & HyDE vs. Snap pp & Gold vs. raw pp & Interpretation \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        def deltas(mode: str, base_mode: str) -> list[tuple[float, str]]:
            out = []
            for provider in PROVIDERS:
                value, value_status = main_accuracy_value(rows, dataset, provider, mode)
                base, base_status = main_accuracy_value(rows, dataset, provider, base_mode)
                if value is not None and base is not None:
                    status = "unsigned" if "unsigned" in {value_status, base_status} else "signed"
                    out.append((100 * (value - base), status))
            return out

        def fmt_delta_mean(items: list[tuple[float, str]]) -> str:
            if not items:
                return "--"
            value = sum(v for v, _ in items) / len(items)
            status = "unsigned" if any(s == "unsigned" for _, s in items) else "signed"
            return f"{value:+.1f} (n={len(items)})" + (r"$^{\dagger}$" if status == "unsigned" else "")

        snap_raw_values = deltas("snap_hyre", "rag_simple")
        hyde_snap_values = deltas("rag_hyde", "snap_hyre")
        oracle_gap_values = deltas("golden_passage", "rag_simple")
        lines.append(
            f"{tex_escape(dataset)} & {tex_escape(raw_story[dataset])} & "
            f"{fmt_delta_mean(snap_raw_values)} & "
            f"{fmt_delta_mean(hyde_snap_values)} & {fmt_delta_mean(oracle_gap_values)} & "
            f"{tex_escape(interpretation[dataset])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    (TABLES / "current_result_anatomy.tex").write_text("\n".join(lines) + "\n")


def table_snap_vs_controls(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Snap-HyRE compared with non-oracle controls. Cells are accuracy percentages; bold marks the strongest non-oracle row for that dataset/model slice. Rewrite is supplemental.}",
        r"\label{tab:snap_vs_controls}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrl}",
        r"\toprule",
        r"Dataset & Model & LLM & Raw question RAG & HyDE & Snap-HyRE & Rewrite & Snap vs. raw pp & Best main non-oracle \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for provider in PROVIDERS:
            values = {}
            statuses = {}
            for mode in ["llm_only", "rag_simple", "rag_hyde", "snap_hyre", "rag_rewrite"]:
                value, status = main_accuracy_value(rows, dataset, provider, mode)
                values[mode] = value
                statuses[mode] = status
            if values["snap_hyre"] is None and not any(value is not None for value in values.values()):
                continue
            best_mode = None
            best_value = None
            for mode, value in values.items():
                if statuses[mode] != "signed":
                    continue
                if value is None:
                    continue
                if best_value is None or value > best_value:
                    best_mode = mode
                    best_value = value
            delta = None
            if values["snap_hyre"] is not None and values["rag_simple"] is not None:
                delta = 100 * (values["snap_hyre"] - values["rag_simple"])
            cells = [
                fmt_main_percent(values[mode], statuses[mode], bold=(best_mode == mode))
                for mode in ["llm_only", "rag_simple", "rag_hyde", "snap_hyre", "rag_rewrite"]
            ]
            best_label = "--" if best_mode is None else f"{MODE_LABEL[best_mode]} ({100 * best_value:.1f})"
            delta_text = pp_delta(delta)
            if delta is not None and ("unsigned" in {statuses["snap_hyre"], statuses["rag_simple"]}):
                delta_text += r"$^{\dagger}$"
            lines.append(
                f"{tex_escape(dataset)} & {tex_escape(PROVIDER_LABEL[provider])} & "
                + " & ".join(cells)
                + f" & {delta_text} & {tex_escape(best_label)} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TABLES / "current_snap_vs_controls.tex").write_text("\n".join(lines) + "\n")


def table_oracle_summary(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Gold-passage oracle controls for the main datasets. Accuracy values are percentages; deltas are percentage points over included row pairs. Neighbor-augmented oracle rows are left to the appendix figure because they diagnose context dilution rather than the main retrieval claim.}",
        r"\label{tab:oracle_summary}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Model & Raw question RAG & Snap-HyRE & Gold & Gold vs. raw pp \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for provider in PROVIDERS:
            raw, raw_status = main_accuracy_value(rows, dataset, provider, "rag_simple")
            snap, snap_status = main_accuracy_value(rows, dataset, provider, "snap_hyre")
            gold, gold_status = main_accuracy_value(rows, dataset, provider, "golden_passage")
            if raw is None and gold is None:
                continue
            gold_raw = 100 * (gold - raw) if gold is not None and raw is not None else None
            gold_raw_text = pp_delta(gold_raw)
            if gold_raw is not None and ("unsigned" in {raw_status, gold_status}):
                gold_raw_text += r"$^{\dagger}$"
            lines.append(
                f"{tex_escape(dataset)} & {tex_escape(PROVIDER_LABEL[provider])} & "
                f"{fmt_main_percent(raw, raw_status)} & "
                f"{fmt_main_percent(snap, snap_status)} & "
                f"{fmt_main_percent(gold, gold_status)} & "
                f"{gold_raw_text} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TABLES / "current_oracle_summary.tex").write_text("\n".join(lines) + "\n")


def table_neighbor_dilution(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Supplemental neighbor-dilution diagnostic. Values are answer accuracy percentages for oracle rows only; positive deltas mean adding nearby retrieved legal text helped relative to the gold passage alone.}",
        r"\label{tab:neighbor_dilution}",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Model & Gold & Gold + neighbors & $\Delta$ pp \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for provider in PROVIDERS:
            gold, gold_status = main_accuracy_value(rows, dataset, provider, "golden_passage")
            gpn, gpn_status = main_accuracy_value(rows, dataset, provider, "golden_plus_neighbors")
            if gold is None or gpn is None:
                continue
            delta = 100 * (gpn - gold)
            suffix = r"$^{\dagger}$" if "unsigned" in {gold_status, gpn_status} else ""
            lines.append(
                f"{tex_escape(dataset)} & {tex_escape(PROVIDER_LABEL[provider])} & "
                f"{fmt_main_percent(gold, gold_status)} & "
                f"{fmt_main_percent(gpn, gpn_status)} & {delta:+.1f}{suffix} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TABLES / "neighbor_dilution.tex").write_text("\n".join(lines) + "\n")


def table_caveats(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    caveat_counts = defaultdict(int)
    for row in rows.values():
        for category in paper_caveat_categories(row):
            caveat_counts[category] += 1
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Tagged row notes. These tags mark validation conditions to carry with row-level claims; they do not by themselves remove a row from the included set or imply severity. Counts are non-exclusive because a row may carry more than one note.}",
        r"\label{tab:row_notes}",
        r"\scriptsize",
        r"\begin{tabularx}{\columnwidth}{rY}",
        r"\toprule",
        r"Tagged rows & Note \\",
        r"\midrule",
    ]
    for label, count in sorted(caveat_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"{count} & {tex_escape(label)} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    (TABLES / "current_audit_notes.tex").write_text("\n".join(lines) + "\n")


def table_completion(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Coverage in the BarExamQA/HousingQA matrix. Included rows support paper claims; archived rows are older or unfiltered runs kept outside the main comparison.}",
        r"\label{tab:coverage}",
        r"\footnotesize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Included & Archived & Pending & Expected \\",
        r"\midrule",
    ]
    total_signed = total_provenance = total_unsigned = total_expected = 0
    for dataset in DATASETS:
        expected = sum(1 for provider in PROVIDERS for mode in MODES if is_expected(dataset, provider, mode))
        signed = sum(
            1
            for provider in PROVIDERS
            for mode in MODES
            if (dataset, provider, mode) in rows
            and is_expected(dataset, provider, mode)
            and is_main_signed_row(rows[(dataset, provider, mode)])
        )
        provenance = sum(
            1
            for provider in PROVIDERS
            for mode in MODES
            if (dataset, provider, mode) in rows
            and is_expected(dataset, provider, mode)
            and is_unfiltered_housing_provenance(dataset, mode)
            and not is_main_signed_row(rows[(dataset, provider, mode)])
        )
        unsigned = sum(
            1
            for provider in PROVIDERS
            for mode in MODES
            if is_unsigned_state_filter_row(dataset, provider, mode)
        )
        total_signed += signed
        total_provenance += provenance
        total_unsigned += unsigned
        total_expected += expected
        lines.append(f"{tex_escape(dataset)} & {signed} & {provenance} & {unsigned} & {expected} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Total & {total_signed} & {total_provenance} & {total_unsigned} & {total_expected} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TABLES / "current_coverage.tex").write_text("\n".join(lines) + "\n")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.unicode_minus": False,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


DATASET_SHORT = {
    "BarExamQA": "Bar",
    "HousingQA": "Hous*",
    "Legal-Link-EU": "Link",
    "MASLegalBench": "MAS",
}


def figure_dataset_label(dataset: str) -> str:
    if dataset == "HousingQA":
        return "HousingQA*"
    return DATASET_LABEL[dataset]
PROVIDER_SHORT = {
    "groq-llama8b": "8B",
    "or-gemma4-26b": "26B",
    "groq-llama70b": "70B",
}


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if value is not None and not math.isnan(value)]


def nice_upper(values: list[float], floor: float = 5.0, ceiling: float | None = None) -> float:
    vals = finite_values(values)
    if not vals:
        return floor
    upper = max(floor, math.ceil(max(vals) * 1.18 / 5.0) * 5.0)
    if ceiling is not None:
        upper = min(ceiling, upper)
    return upper


def set_symmetric_pp_axis(ax, values: list[float], floor: float = 5.0) -> float:
    upper = nice_upper([abs(value) for value in values], floor=floor)
    ax.set_ylim(-upper, upper)
    return upper


def annotate_delta_bars(ax, values: list[float], upper: float) -> None:
    offset = max(0.25, upper * 0.035)
    for i, val in enumerate(values):
        ax.text(
            i,
            val + (offset if val >= 0 else -offset),
            f"{val:+.1f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )


def fig_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge="#2f3a45", fontsize=9.5, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.022",
            linewidth=1.15,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#1d2630", weight=weight)
        return patch

    def arrow(start, end, color="#4b5663", style="-|>", linestyle="-", rad=0.0, label=None, label_xy=None):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=15,
            linewidth=1.25,
            linestyle=linestyle,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(patch)
        if label and label_xy:
            ax.text(label_xy[0], label_xy[1], label, ha="center", va="center", fontsize=8, color=color)

    ax.text(
        0.50,
        0.96,
        "Snap-HyRE changes the search text, not the evidence shown to the final answerer",
        ha="center",
        va="top",
        fontsize=13,
        weight="bold",
        color="#1d2630",
    )

    for x, label in [
        (0.18, "Input"),
        (0.38, "Retrieval query"),
        (0.57, "Retrieve"),
        (0.74, "Evidence"),
        (0.90, "Answer"),
    ]:
        ax.text(x, 0.865, label, ha="center", va="center", fontsize=9, weight="bold", color="#4b5663")

    rows = [
        ("Raw question RAG", 0.70, "#eaf2f8", "#2f6f9f"),
        ("HyDE", 0.47, "#e4f4f2", "#48a9a6"),
        ("Snap-HyRE", 0.24, "#fff3d6", "#d4942f"),
    ]

    for label, y, face, edge in rows:
        ax.text(0.035, y, label, ha="left", va="center", fontsize=10.5, weight="bold", color=edge)
        ax.plot([0.13, 0.96], [y - 0.12, y - 0.12], color="#d8dee6", linewidth=0.8)
        box(0.15, y - 0.055, 0.105, 0.11, "question", "#ffffff", "#8290a3", weight="bold")
        box(0.52, y - 0.055, 0.105, 0.11, "retriever", "#f6f8fb", "#5f6f7f")
        box(0.68, y - 0.055, 0.12, 0.11, "top-5\nlegal text", "#eaf2f8", "#2f6f9f")
        box(0.86, y - 0.055, 0.11, 0.11, "final\nanswer", "#f4e8f7", "#7f5aa2", weight="bold")

    box(0.32, 0.645, 0.13, 0.11, "question text", "#eaf2f8", "#2f6f9f", weight="bold")
    arrow((0.255, 0.70), (0.32, 0.70), color="#2f6f9f")
    arrow((0.45, 0.70), (0.52, 0.70), color="#2f6f9f")
    arrow((0.625, 0.70), (0.68, 0.70), color="#2f6f9f")
    arrow((0.80, 0.70), (0.86, 0.70), color="#7f5aa2")

    box(0.31, 0.415, 0.16, 0.11, "generated\nlegal passage", "#e4f4f2", "#48a9a6", weight="bold")
    arrow((0.255, 0.47), (0.31, 0.47), color="#48a9a6")
    arrow((0.47, 0.47), (0.52, 0.47), color="#48a9a6")
    arrow((0.625, 0.47), (0.68, 0.47), color="#48a9a6")
    arrow((0.80, 0.47), (0.86, 0.47), color="#7f5aa2")

    box(0.30, 0.265, 0.15, 0.085, "private\ndraft answer", "#fff7e8", "#a15c38", fontsize=8.5, weight="bold")
    box(0.30, 0.145, 0.15, 0.095, "draft-guided\nsearch passage", "#e7f6ef", "#2f855a", fontsize=8.5, weight="bold")
    arrow((0.255, 0.24), (0.30, 0.305), color="#d4942f", rad=0.15)
    arrow((0.375, 0.265), (0.375, 0.24), color="#a15c38")
    arrow((0.45, 0.192), (0.52, 0.24), color="#2f855a", rad=0.08)
    arrow((0.625, 0.24), (0.68, 0.24), color="#2f6f9f")
    arrow((0.80, 0.24), (0.86, 0.24), color="#7f5aa2")
    arrow((0.45, 0.308), (0.86, 0.295), color="#a15c38", linestyle="--", rad=-0.05)
    ax.text(0.64, 0.335, "kept private", ha="center", va="bottom", fontsize=8, color="#a15c38")
    ax.text(0.83, 0.305, "X", ha="center", va="center", fontsize=12, weight="bold", color="#a15c38")
    ax.text(
        0.50,
        0.035,
        "The final call always receives the original question plus retrieved legal text. The draft answer is logged for analysis, not supplied as evidence.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4b5663",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "20_snap_hyre_pipeline_art.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_canonical_method_ladder() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [
        ("Method row", 0.02, 0.16),
        ("Retrieval query", 0.20, 0.23),
        ("Context shown to answer call", 0.46, 0.25),
        ("Interpretation", 0.74, 0.23),
    ]
    rows = [
        (
            "LLM-only\nllm_only",
            "none",
            "original question only",
            "parametric baseline",
            "#f2f4f7",
            "#5f6f7f",
        ),
        (
            "Raw RAG\nrag_simple",
            "raw question",
            "question + top-k retrieved evidence",
            "retrieval baseline",
            "#eaf2f8",
            "#2f6f9f",
        ),
        (
            "HyDE\nrag_hyde",
            "hypothetical legal passage\n(no preliminary answer)",
            "question + evidence retrieved by\nthe generated passage",
            "generated prose without\nsnap conditioning",
            "#e4f4f2",
            "#48a9a6",
        ),
        (
            "Snap-HyRE\nsnap_hyre",
            "HyRE legal-reference passage\nconditioned by a private snap answer",
            "question + evidence retrieved by HyRE\n(preliminary answer not shown)",
            "primary fixed method",
            "#fff3d6",
            "#d4942f",
        ),
        (
            "Rewrite\nrag_rewrite",
            "structured rewritten query\nor query set",
            "question + evidence retrieved by\nthe rewrite",
            "non-HyDE generated-query\ncontrol",
            "#fbeaea",
            "#c65f5f",
        ),
        (
            "Gold passage\ngolden_passage",
            "none; labeled passage supplied",
            "question + gold passage",
            "oracle evidence control",
            "#eee6f6",
            "#7f5aa2",
        ),
        (
            "Gold + neighbors\ngolden_plus_neighbors",
            "gold passage plus nearby\nretrieved legal text",
            "question + gold passage + neighbors",
            "tests context dilution\nor support",
            "#f2ead3",
            "#9c7c38",
        ),
    ]

    ax.text(
        0.5,
        0.965,
        "Canonical method ladder: what is retrieved, what is shown, and what the row means",
        ha="center",
        va="top",
        fontsize=13.5,
        weight="bold",
        color="#1d2630",
    )
    ax.text(
        0.5,
        0.925,
        "Oracle rows are evidence controls; they are not deployable retrieval methods.",
        ha="center",
        va="top",
        fontsize=9,
        color="#4b5663",
    )

    header_y = 0.86
    for header, x, width in columns:
        ax.text(x + width / 2, header_y, header, ha="center", va="center", fontsize=9, weight="bold", color="#1d2630")
    ax.plot([0.02, 0.97], [0.835, 0.835], color="#2f3a45", linewidth=1.1)

    row_h = 0.092
    start_y = 0.79
    for i, (method, query, context, interp, face, edge) in enumerate(rows):
        y = start_y - i * row_h
        texts = [method, query, context, interp]
        for (_, x, width), text in zip(columns, texts):
            patch = FancyBboxPatch(
                (x, y - row_h / 2 + 0.007),
                width,
                row_h - 0.014,
                boxstyle="round,pad=0.012,rounding_size=0.014",
                linewidth=1.1,
                edgecolor=edge,
                facecolor=face,
            )
            ax.add_patch(patch)
            ax.text(
                x + width / 2,
                y,
                text,
                ha="center",
                va="center",
                fontsize=7.7 if x > 0.02 else 8.4,
                color="#1d2630",
                weight="bold" if x == 0.02 else "normal",
            )
        for j in range(1, len(columns) - 1):
            left = columns[j][1] + columns[j][2]
            right = columns[j + 1][1]
            arrow = FancyArrowPatch(
                (left + 0.008, y),
                (right - 0.008, y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#6a737d",
            )
            ax.add_patch(arrow)

    ax.text(
        0.5,
        0.06,
        "All non-oracle retrieval rows use the same top-k answer depth in the main matrix.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4b5663",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "34_canonical_method_ladder.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_method_flow() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 3.0))
    ax.axis("off")
    stages = [
        ("Legal question", "#eaf2f8"),
        ("Call 1:\nsnap answer +\nHyRE passage", "#fff3d6"),
        ("Retrieve with\nHyRE passage", "#e7f6ef"),
        ("Top-k legal\nevidence", "#eaf2f8"),
        ("Call 2:\nanswer original\nquestion from evidence", "#f4e8f7"),
    ]
    xs = [0.10, 0.30, 0.50, 0.70, 0.90]
    y = 0.55
    for i, ((label, color), x) in enumerate(zip(stages, xs)):
        box = FancyBboxPatch(
            (x - 0.075, y - 0.18),
            0.15,
            0.36,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#2f3a45",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, color="#1d2630")
        if i < len(xs) - 1:
            arrow = FancyArrowPatch(
                (x + 0.08, y),
                (xs[i + 1] - 0.08, y),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.3,
                color="#45515e",
            )
            ax.add_patch(arrow)
    fig.tight_layout()
    fig.savefig(FIGURES / "21_snap_hyre_method_flow.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_answer_heatmap(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    labels = []
    matrix = []
    cell_text = []
    for dataset in DATASETS:
        for provider in PROVIDERS:
            labels.append(f"{figure_dataset_label(dataset)}\n{PROVIDER_LABEL[provider]}")
            vals = []
            texts = []
            for mode in MODES:
                if not is_expected(dataset, provider, mode):
                    vals.append(math.nan)
                    texts.append("n/a")
                else:
                    raw_value, status = main_accuracy_value(rows, dataset, provider, mode)
                    if raw_value is None:
                        vals.append(math.nan)
                        texts.append("--")
                        continue
                    value = 100 * raw_value
                    vals.append(value)
                    suffix = "†" if status == "unsigned" else ""
                    texts.append(f"{value:.1f}{suffix}")
            matrix.append(vals)
            cell_text.append(texts)
    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=40, vmax=100)
    ax.set_xticks(range(len(MODES)), [MODE_LABEL[m] for m in MODES], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if math.isnan(val):
                ax.text(j, i, cell_text[i][j], ha="center", va="center", color="#333333", fontsize=8)
            else:
                color = "white" if val > 75 else "#1b1b1b"
                ax.text(j, i, cell_text[i][j], ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Audited answer accuracy at k=5")
    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "22_answer_heatmap.png", dpi=240)
    plt.close(fig)


def fig_completion_grid(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    labels = []
    matrix = []
    for dataset in DATASETS:
        for provider in PROVIDERS:
            labels.append(f"{figure_dataset_label(dataset)}\n{PROVIDER_LABEL[provider]}")
            matrix_row = []
            for mode in MODES:
                if not is_expected(dataset, provider, mode):
                    matrix_row.append(math.nan)
                elif is_unsigned_state_filter_row(dataset, provider, mode):
                    matrix_row.append(3)
                elif (dataset, provider, mode) in rows and is_main_signed_row(rows[(dataset, provider, mode)]):
                    matrix_row.append(1)
                elif (dataset, provider, mode) in rows and is_unfiltered_housing_provenance(dataset, mode):
                    matrix_row.append(2)
                elif (dataset, provider, mode) in rows:
                    matrix_row.append(1)
                else:
                    matrix_row.append(0)
            matrix.append(matrix_row)
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    cmap = plt.matplotlib.colors.ListedColormap(["#f1f2f4", "#2f855a", "#d4942f", "#4f7db8", "#ffffff"])
    im = ax.imshow([[4 if math.isnan(v) else v for v in row] for row in matrix], cmap=cmap, aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(range(len(MODES)), [MODE_LABEL[m] for m in MODES], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if math.isnan(val):
                text = "n/a"
            elif val == 2:
                text = "archived"
            elif val == 3:
                text = "pending"
            elif val == 1:
                text = "included"
            else:
                text = "--"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)
    ax.set_title("Included, archived, and pending cells in the exact-scored matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "23_completion_grid.png", dpi=240)
    plt.close(fig)


def fig_snap_delta(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    items = []
    for dataset in DATASETS:
        for provider in PROVIDERS:
            raw, raw_status = main_accuracy_value(rows, dataset, provider, "rag_simple")
            snap, snap_status = main_accuracy_value(rows, dataset, provider, "snap_hyre")
            if raw is not None and snap is not None:
                status = "unsigned" if "unsigned" in {raw_status, snap_status} else "signed"
                items.append((dataset, provider, 100 * (snap - raw), status))
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    labels = [f"{figure_dataset_label(d)}\n{PROVIDER_LABEL[p]}{'†' if status == 'unsigned' else ''}" for d, p, _, status in items]
    vals = [v for _, _, v, _ in items]
    colors = ["#2f855a" if v >= 0 else "#c43c3c" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="#2f3a45", linewidth=1)
    ax.set_xticks(range(len(vals)), labels, rotation=35, ha="right")
    ax.set_ylabel("Snap-HyRE minus Raw RAG (pp)")
    ax.set_title("Snap-HyRE answer accuracy change against Raw RAG")
    upper = set_symmetric_pp_axis(ax, vals, floor=6.0)
    annotate_delta_bars(ax, vals, upper)
    fig.tight_layout()
    fig.savefig(FIGURES / "24_snap_vs_raw_delta.png", dpi=240)
    plt.close(fig)


def fig_retrieval_answer_delta(rows: dict[tuple[str, str, str], SignedRow], retrieval: dict[tuple[str, str, str], SignedRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    all_xs = []
    all_ys = []
    for mode, marker in [("rag_hyde", "o"), ("snap_hyre", "s"), ("rag_rewrite", "^")]:
        xs, ys, labels = [], [], []
        for dataset in DATASETS:
            for provider in PROVIDERS:
                raw_a, raw_a_status = main_accuracy_value(rows, dataset, provider, "rag_simple")
                meth_a, meth_a_status = main_accuracy_value(rows, dataset, provider, mode)
                raw_hit, _, raw_r_status = main_retrieval_value(retrieval, dataset, provider, "rag_simple")
                meth_hit, _, meth_r_status = main_retrieval_value(retrieval, dataset, provider, mode)
                if raw_a is None or meth_a is None or raw_hit is None or meth_hit is None:
                    continue
                status = "†" if "unsigned" in {raw_a_status, meth_a_status, raw_r_status, meth_r_status} else ""
                xs.append(100 * (meth_hit - raw_hit))
                ys.append(100 * (meth_a - raw_a))
                labels.append(f"{DATASET_SHORT[dataset]}-{PROVIDER_SHORT[provider]}{status}")
        ax.scatter(xs, ys, label=MODE_LABEL[mode], marker=marker, s=70, color=COLORS[mode], edgecolor="white", linewidth=0.8)
        for x_val, y_val, label in zip(xs, ys, labels):
            if abs(x_val) >= 30 or abs(y_val) >= 5:
                ax.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(6, 5), fontsize=6.5, color="#35404a")
        all_xs.extend(xs)
        all_ys.extend(ys)
    ax.axhline(0, color="#3f4750", linewidth=1, linestyle=":")
    ax.axvline(0, color="#3f4750", linewidth=1, linestyle=":")
    if all_xs:
        xmin, xmax = min(min(all_xs), 0.0), max(max(all_xs), 0.0)
        xpad = max(2.0, 0.08 * (xmax - xmin or 1.0))
        ax.set_xlim(xmin - xpad, xmax + xpad)
    set_symmetric_pp_axis(ax, all_ys, floor=8.0)
    ax.set_xlabel("Retrieval exposure change vs Raw RAG (Hit@5 pp)")
    ax.set_ylabel("Answer accuracy delta vs Raw RAG (pp)")
    ax.set_title("When does retrieval accuracy help downstream accuracy?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "25_retrieval_answer_delta.png", dpi=240)
    plt.close(fig)


def fig_oracle_gap(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    items = []
    for dataset in DATASETS:
        for provider in PROVIDERS:
            raw, raw_status = main_accuracy_value(rows, dataset, provider, "rag_simple")
            gold, gold_status = main_accuracy_value(rows, dataset, provider, "golden_passage")
            gpn, gpn_status = main_accuracy_value(rows, dataset, provider, "golden_plus_neighbors")
            suffix = "†" if raw_status == "unsigned" else ""
            if raw is not None and gold is not None:
                items.append((dataset, provider, "Gold", 100 * (gold - raw), suffix))
            if raw is not None and gpn is not None:
                items.append((dataset, provider, "Gold+Nbrs", 100 * (gpn - raw), suffix))
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    group_labels = []
    x = []
    gold_vals = []
    gpn_vals = []
    grouped = defaultdict(dict)
    suffix_by_group = {}
    for dataset, provider, mode, val, suffix in items:
        grouped[(dataset, provider)][mode] = val
        suffix_by_group[(dataset, provider)] = suffix
    for i, ((dataset, provider), vals) in enumerate(grouped.items()):
        group_labels.append(f"{figure_dataset_label(dataset)}\n{PROVIDER_LABEL[provider]}{suffix_by_group.get((dataset, provider), '')}")
        x.append(i)
        gold_vals.append(vals.get("Gold", math.nan))
        gpn_vals.append(vals.get("Gold+Nbrs", math.nan))
    width = 0.38
    ax.bar([v - width / 2 for v in x], gold_vals, width, label="Gold", color=COLORS["golden_passage"])
    ax.bar([v + width / 2 for v in x], gpn_vals, width, label="Gold+Nbrs", color=COLORS["golden_plus_neighbors"])
    ax.axhline(0, color="#2f3a45", linewidth=1)
    ax.set_xticks(x, group_labels, rotation=35, ha="right")
    ax.set_ylabel("Oracle control minus Raw RAG (pp)")
    ax.set_title("Oracle evidence controls and neighbor dilution")
    all_vals = [value for value in gold_vals + gpn_vals if not math.isnan(value)]
    set_symmetric_pp_axis(ax, all_vals, floor=10.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "26_oracle_gap.png", dpi=240)
    plt.close(fig)


def fig_retrieval_by_method(retrieval: dict[tuple[str, str, str], SignedRow]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        labels = [PROVIDER_LABEL[p] for p in PROVIDERS]
        x = list(range(len(PROVIDERS)))
        width = 0.23
        generated_modes = ["rag_hyde", "snap_hyre", "rag_rewrite"]
        offsets = [-width, 0.0, width]
        raw_values = []
        for provider in PROVIDERS:
            hit, _, _ = main_retrieval_value(retrieval, dataset, provider, "rag_simple")
            if hit is not None:
                raw_values.append(100 * hit)
        panel_vals = []
        if raw_values:
            raw_value = raw_values[0]
            panel_vals.append(raw_value)
            ax.axhline(raw_value, color=COLORS["rag_simple"], linestyle="--", linewidth=1.5, label="Raw RAG (shared)")
        for mode, off in zip(generated_modes, offsets):
            vals = []
            for provider in PROVIDERS:
                hit, _, _ = main_retrieval_value(retrieval, dataset, provider, mode)
                vals.append(100 * hit if hit is not None else math.nan)
            panel_vals.extend(value for value in vals if not math.isnan(value))
            ax.bar([i + off for i in x], vals, width, label=MODE_LABEL[mode], color=COLORS[mode])
        title = figure_dataset_label(dataset)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Hit@5 (%)")
        if panel_vals and dataset in {"BarExamQA", "HousingQA"}:
            upper = max(12.5, math.ceil(max(panel_vals) * 1.25 / 5.0) * 5.0)
            ax.set_ylim(0, upper)
        else:
            ax.set_ylim(0, 100)
        for idx, provider in enumerate(PROVIDERS):
            has_any = any(
                main_retrieval_value(retrieval, dataset, provider, mode)[0] is not None
                for mode in RETRIEVAL_MODES
            )
            if not has_any:
                ax.text(idx, 3.0, "open", ha="center", va="bottom", fontsize=7, color="#6c7580")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=False)
    fig.suptitle("Audited retrieval exposure by dataset and model", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIGURES / "27_retrieval_exposure_by_method.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_method_means(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    means = []
    counts = []
    for mode in MODES:
        vals = [
            row.accuracy
            for key, row in rows.items()
            if key[2] == mode and row.accuracy is not None and is_main_signed_row(row)
        ]
        means.append(100 * sum(vals) / len(vals) if vals else math.nan)
        counts.append(len(vals))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(range(len(MODES)), means, color=[COLORS[m] for m in MODES])
    ax.set_xticks(range(len(MODES)), [MODE_LABEL[m] for m in MODES], rotation=35, ha="right")
    ax.set_ylabel("Mean audited accuracy (%)")
    ax.set_title("Main signed-cell descriptive mean (not a leaderboard)")
    for i, (val, count) in enumerate(zip(means, counts)):
        if not math.isnan(val):
            ax.text(i, val + 0.5, f"{val:.1f}\nn={count}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(FIGURES / "28_method_mean_accuracy.png", dpi=240)
    plt.close(fig)


def fig_cost_accuracy(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for mode in MODES:
        vals = [
            100 * row.accuracy
            for key, row in rows.items()
            if key[2] == mode and row.accuracy is not None and is_main_signed_row(row)
        ]
        if not vals:
            continue
        jitter = [(i - (len(vals) - 1) / 2) * 0.018 for i in range(len(vals))]
        ax.scatter([CALLS[mode] + j for j in jitter], vals, s=42, color=COLORS[mode], alpha=0.46)
        ax.scatter([CALLS[mode]], [sum(vals) / len(vals)], s=120, color=COLORS[mode], edgecolor="white", linewidth=1.2, label=MODE_LABEL[mode])
    ax.set_xticks([1, 2], ["1 call", "2 calls"])
    ax.set_ylabel("Audited answer accuracy (%)")
    ax.set_title("Conceptual call count versus answer accuracy")
    ax.set_xlim(0.78, 2.22)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "29_cost_accuracy.png", dpi=240)
    plt.close(fig)


def fig_snap_hyde_delta(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    items = []
    for dataset in DATASETS:
        for provider in PROVIDERS:
            snap = accuracy_value(rows, dataset, provider, "snap_hyre")
            hyde = accuracy_value(rows, dataset, provider, "rag_hyde")
            if snap is not None and hyde is not None:
                items.append((dataset, provider, 100 * (snap - hyde)))
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    labels = [f"{figure_dataset_label(d)}\n{PROVIDER_LABEL[p]}" for d, p, _ in items]
    vals = [v for _, _, v in items]
    colors = ["#2f855a" if v >= 0 else "#c43c3c" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="#2f3a45", linewidth=1)
    ax.set_xticks(range(len(vals)), labels, rotation=35, ha="right")
    ax.set_ylabel("Snap-HyRE minus HyDE (pp)")
    ax.set_title("Snap-HyRE versus HyDE answer accuracy")
    for i, val in enumerate(vals):
        ax.text(i, val + (0.25 if val >= 0 else -0.45), f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "31_snap_vs_hyde_delta.png", dpi=240)
    plt.close(fig)


def fig_dataset_method_deltas(rows: dict[tuple[str, str, str], SignedRow]) -> None:
    methods = ["rag_hyde", "snap_hyre", "rag_rewrite"]
    labels = [MODE_LABEL[m] for m in methods]
    x = list(range(len(DATASETS)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    all_vals = []
    for offset, mode in zip([-width, 0, width], methods):
        vals = []
        counts = []
        for dataset in DATASETS:
            paired = signed_delta_values(rows, dataset, mode, "rag_simple")
            val = mean(paired)
            vals.append(math.nan if val is None else val)
            counts.append(len(paired))
        all_vals.extend([value for value in vals if not math.isnan(value)])
        bars = ax.bar([i + offset for i in x], vals, width, label=MODE_LABEL[mode], color=COLORS[mode])
        for bar, val, count in zip(bars, vals, counts):
            if math.isnan(val):
                continue
            y = val + (0.35 if val >= 0 else -0.35)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:+.1f}\nn={count}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
            )
    ax.axhline(0, color="#2f3a45", linewidth=1)
    ax.set_xticks(x, [figure_dataset_label(dataset) for dataset in DATASETS], rotation=20, ha="right")
    ax.set_ylabel("Mean audited delta vs Raw RAG (pp)")
    ax.set_title("Descriptive unbalanced mean deltas by dataset")
    set_symmetric_pp_axis(ax, all_vals, floor=8.0)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "32_method_delta_vs_raw_by_dataset.png", dpi=240)
    plt.close(fig)


def float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_experiment_summaries() -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    if not EXPERIMENTS.exists():
        return summaries
    with EXPERIMENTS.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            detail_log = row.get("detail_log")
            if not detail_log:
                continue
            total = int(row.get("total") or row.get("n_questions") or 0)
            old = summaries.get(detail_log)
            old_total = int(old.get("total") or old.get("n_questions") or 0) if old else -1
            if old is None or total >= old_total:
                summaries[detail_log] = row
    return summaries


def build_operational_records(rows: dict[tuple[str, str, str], SignedRow]) -> list[dict[str, object]]:
    summaries = load_experiment_summaries()
    records = []
    for key, row in sorted(rows.items()):
        if not is_main_signed_row(row):
            continue
        if not row.detail_log or row.detail_log not in summaries or row.total is None:
            continue
        summary = summaries[row.detail_log]
        total = int(summary.get("total") or summary.get("n_questions") or 0)
        if total != row.total:
            continue
        input_tokens = float_or_none(summary.get("total_input_tokens"))
        output_tokens = float_or_none(summary.get("total_output_tokens"))
        records.append(
            {
                "dataset": row.dataset,
                "provider": row.provider,
                "mode": row.mode,
                "total": row.total,
                "correct": row.correct,
                "accuracy": row.accuracy,
                "avg_latency_sec": float_or_none(summary.get("avg_latency_sec")),
                "avg_llm_calls": float_or_none(summary.get("avg_llm_calls")),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_per_question": None if input_tokens is None else input_tokens / row.total,
                "output_per_question": None if output_tokens is None else output_tokens / row.total,
                "detail_log": row.detail_log,
            }
        )
    return records


def table_operational_metrics(records: list[dict[str, object]]) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Token and scored-answer call accounting for included BarExamQA/HousingQA answer rows. Values cover rows with token fields and count only the call that produces the submitted answer from the original question and any supplied context, including format retries. First-stage query-generation usage is not included, so conceptual end-to-end call counts remain those in Table~\ref{tab:method_ladder}.}",
        r"\label{tab:usage_metrics}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Cells & Tok./q & Input tok./q & Output tok./q & Scored calls/q & Correct / 1M tok. \\",
        r"\midrule",
    ]
    for mode in [m for m in MODES if m != "golden_plus_neighbors"]:
        group = grouped.get(mode, [])
        if not group:
            continue
        token_group = [
            record for record in group
            if record.get("input_tokens") is not None and record.get("output_tokens") is not None
        ]
        total_questions = sum(int(record["total"]) for record in token_group)
        total_tokens = sum(float(record["input_tokens"]) + float(record["output_tokens"]) for record in token_group)
        total_correct = sum(int(record["correct"]) for record in token_group if record.get("correct") is not None)
        tok_per_q = None if total_questions == 0 else total_tokens / total_questions
        correct_per_m = None if total_tokens == 0 else 1_000_000 * total_correct / total_tokens
        in_tok = None if total_questions == 0 else sum(float(record["input_tokens"]) for record in token_group) / total_questions
        out_tok = None if total_questions == 0 else sum(float(record["output_tokens"]) for record in token_group) / total_questions
        calls = mean([record["avg_llm_calls"] for record in group if record["avg_llm_calls"] is not None])
        lines.append(
            f"{tex_escape(MODE_LABEL[mode])} & {len(group)} & "
            f"{num(tok_per_q)} & {num(in_tok)} & {num(out_tok)} & {num(calls, 2)} & {num(correct_per_m, 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TABLES / "current_usage_metrics.tex").write_text("\n".join(lines) + "\n")


def fig_logged_tokens_by_method(records: list[dict[str, object]]) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)
    methods = [mode for mode in MODES if grouped.get(mode)]
    input_vals = [mean([record["input_per_question"] for record in grouped[mode] if record["input_per_question"] is not None]) for mode in methods]
    output_vals = [mean([record["output_per_question"] for record in grouped[mode] if record["output_per_question"] is not None]) for mode in methods]
    x = list(range(len(methods)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.bar([i - width / 2 for i in x], input_vals, width, label="Input", color="#4b79a1")
    ax.bar([i + width / 2 for i in x], output_vals, width, label="Output", color="#d4942f")
    ax.set_xticks(x, [MODE_LABEL[mode] for mode in methods], rotation=35, ha="right")
    ax.set_ylabel("Answer-pass tokens per question")
    ax.set_title("Answer-pass token accounting; generation pass excluded")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "33_token_accounting_by_method.png", dpi=240)
    plt.close(fig)


def write_operational_csv(records: list[dict[str, object]]) -> None:
    out = FIGURES / "current_usage_metrics.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "provider",
                "mode",
                "total",
                "accuracy",
                "avg_latency_sec",
                "avg_llm_calls",
                "input_per_question",
                "output_per_question",
                "detail_log",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["dataset"],
                    record["provider"],
                    record["mode"],
                    record["total"],
                    f"{float(record['accuracy']):.6f}" if record["accuracy"] is not None else "",
                    "" if record["avg_latency_sec"] is None else f"{float(record['avg_latency_sec']):.4f}",
                    "" if record["avg_llm_calls"] is None else f"{float(record['avg_llm_calls']):.4f}",
                    "" if record["input_per_question"] is None else f"{float(record['input_per_question']):.2f}",
                    "" if record["output_per_question"] is None else f"{float(record['output_per_question']):.2f}",
                    record["detail_log"],
                ]
            )


def coerce_ids(value):
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return coerce_ids(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, dict):
        ids = []
        for item in value.values():
            ids.extend(coerce_ids(item))
        return ids
    if isinstance(value, (list, tuple, set)):
        ids = []
        for item in value:
            ids.extend(coerce_ids(item))
        return ids
    return [str(value).strip()] if str(value).strip() else []


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def retrieval_probe_metric(path: Path, kind: str, k: int = 5) -> tuple[int, float, float]:
    rows = load_jsonl(path)
    hits = []
    reciprocal_ranks = []
    for row in rows:
        retrieved = coerce_ids(row.get("effective_retrieved_ids")) or coerce_ids(row.get("retrieved_ids"))
        if kind == "same_source":
            target = set(coerce_ids(row.get("same_source_retrieved_ids")))
        else:
            target = set(coerce_ids(row.get("gold_ids")))
        top = retrieved[:k]
        hits.append(1.0 if target and any(idx in target for idx in top) else 0.0)
        reciprocal_rank = 0.0
        if target:
            for rank, idx in enumerate(top, 1):
                if idx in target:
                    reciprocal_rank = 1.0 / rank
                    break
        reciprocal_ranks.append(reciprocal_rank)
    return len(rows), sum(hits) / len(hits), sum(reciprocal_ranks) / len(reciprocal_ranks)


def retrieval_topk_metrics(path: Path, ks: list[int]) -> dict[str, object]:
    rows = load_jsonl(path)
    hits = {k: [] for k in ks}
    reciprocal_ranks = {k: [] for k in ks}
    for row in rows:
        retrieved = coerce_ids(row.get("effective_retrieved_ids")) or coerce_ids(row.get("retrieved_ids"))
        target = set(coerce_ids(row.get("gold_ids")) or coerce_ids(row.get("gold_idx")))
        for k in ks:
            top = retrieved[:k]
            hits[k].append(1.0 if target and any(idx in target for idx in top) else 0.0)
            rr = 0.0
            if target:
                for rank, idx in enumerate(top, 1):
                    if idx in target:
                        rr = 1.0 / rank
                        break
            reciprocal_ranks[k].append(rr)
    return {
        "n": len(rows),
        "hit": {k: mean(hits[k]) or 0.0 for k in ks},
        "mrr": {k: mean(reciprocal_ranks[k]) or 0.0 for k in ks},
    }


def build_topk_retrieval_rows() -> list[dict[str, object]]:
    rows = []
    for spec in TOPK_RETRIEVAL_SPECS:
        path = ROOT / str(spec["path"])
        if not path.exists():
            continue
        metrics = retrieval_topk_metrics(path, TOPK_KS)
        rows.append({**spec, **metrics})
    return rows


def build_housing_metadata_filter_rows() -> list[dict[str, object]]:
    rows = []
    for spec in HOUSING_METADATA_FILTER_SPECS:
        path = ROOT / str(spec["path"])
        if not path.exists():
            continue
        metrics = retrieval_topk_metrics(path, TOPK_KS)
        rows.append({**spec, **metrics})
    return rows


def aggregate_topk_rows(rows: list[dict[str, object]], dataset: str, mode: str, model_label: str) -> dict[str, object] | None:
    group = [row for row in rows if row["dataset"] == dataset and row["mode"] == mode and row["provider"] != "shared"]
    if not group:
        return None
    return {
        "dataset": dataset,
        "provider": "mean",
        "model": model_label,
        "mode": mode,
        "path": "aggregate",
        "n": min(int(row["n"]) for row in group),
        "hit": {k: mean([row["hit"][k] for row in group]) or 0.0 for k in TOPK_KS},
        "mrr": {k: mean([row["mrr"][k] for row in group]) or 0.0 for k in TOPK_KS},
    }


def topk_cell(row: dict[str, object], metric: str, k: int) -> str:
    values = row[metric]
    assert isinstance(values, dict)
    return f"{100 * float(values[k]):.1f}"


def table_topk_retrieval(rows: list[dict[str, object]]) -> None:
    bar_hyde_mean = aggregate_topk_rows(rows, "BarExamQA", "rag_hyde", "Mean over 3 models")
    bar_snap_mean = aggregate_topk_rows(rows, "BarExamQA", "snap_hyre", "Mean over 3 models")
    housing_hyde_mean = aggregate_topk_rows(rows, "HousingQA", "rag_hyde", "Mean over 2 full models")
    housing_snap_mean = aggregate_topk_rows(rows, "HousingQA", "snap_hyre", "Mean over 2 full models")
    display_rows = []
    display_rows.extend([row for row in rows if row["dataset"] == "BarExamQA" and row["mode"] == "rag_simple"])
    if bar_hyde_mean:
        display_rows.append(bar_hyde_mean)
    if bar_snap_mean:
        display_rows.append(bar_snap_mean)
    display_rows.extend(
        row
        for row in rows
        if row["dataset"] == "BarExamQA" and row["mode"] in {"rag_hyde", "snap_hyre"}
    )
    display_rows.extend([row for row in rows if row["dataset"] == "HousingQA" and row["mode"] == "rag_simple"])
    if housing_hyde_mean:
        display_rows.append(housing_hyde_mean)
    if housing_snap_mean:
        display_rows.append(housing_snap_mean)
    display_rows.extend(
        row
        for row in rows
        if row["dataset"] == "HousingQA" and row["mode"] in {"rag_hyde", "snap_hyre"}
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Top-$k$ retrieval diagnostics from complete top-10 caches. Values are percentages. BarExamQA includes raw-question retrieval plus each model-specific generated-query cache; mean rows average the three generated-query models. HousingQA rows use the state-filtered retrieval interface and include only complete full-corpus caches.}",
        r"\label{tab:topk_retrieval}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Dataset & Model/scope & Method & $n$ & Hit@1 & Hit@3 & Hit@5 & Hit@10 & MRR@10 \\",
        r"\midrule",
    ]
    last_dataset = None
    for row in display_rows:
        dataset = str(row["dataset"])
        if last_dataset is not None and dataset != last_dataset:
            lines.append(r"\addlinespace")
        lines.append(
            f"{tex_escape(dataset)} & {tex_escape(str(row['model']))} & "
            f"{tex_escape(MODE_LABEL[str(row['mode'])])} & {int(row['n'])} & "
            f"{topk_cell(row, 'hit', 1)} & {topk_cell(row, 'hit', 3)} & "
            f"{topk_cell(row, 'hit', 5)} & {topk_cell(row, 'hit', 10)} & "
            f"{topk_cell(row, 'mrr', 10)} \\\\"
        )
        last_dataset = dataset
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TABLES / "topk_retrieval_summary.tex").write_text("\n".join(lines) + "\n")


def table_housing_metadata_filter(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{HousingQA raw retrieval with and without the jurisdiction metadata filter. Values are percentages from complete top-10 caches.}",
        r"\label{tab:housing_metadata_filter}",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Corpus scope & Hit@1 & Hit@3 & Hit@5 & Hit@10 & MRR@10 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(str(row['scope']))} & "
            f"{topk_cell(row, 'hit', 1)} & {topk_cell(row, 'hit', 3)} & "
            f"{topk_cell(row, 'hit', 5)} & {topk_cell(row, 'hit', 10)} & "
            f"{topk_cell(row, 'mrr', 10)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TABLES / "housing_metadata_filter.tex").write_text("\n".join(lines) + "\n")


def write_topk_retrieval_csv(rows: list[dict[str, object]]) -> None:
    out = FIGURES / "topk_retrieval_metrics.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "provider", "model", "mode", "n", "k", "hit", "mrr", "source_path"])
        for row in rows:
            for k in TOPK_KS:
                writer.writerow(
                    [
                        row["dataset"],
                        row["provider"],
                        row["model"],
                        row["mode"],
                        row["n"],
                        k,
                        f"{float(row['hit'][k]):.6f}",
                        f"{float(row['mrr'][k]):.6f}",
                        row["path"],
                    ]
                )


def fig_topk_retrieval_curves(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.2), sharex=True, constrained_layout=True)
    panel_specs = [
        ("BarExamQA", ["rag_simple", "rag_hyde", "snap_hyre"]),
        ("HousingQA", ["rag_simple", "rag_hyde", "snap_hyre"]),
    ]
    for col, (dataset, methods) in enumerate(panel_specs):
        for mode in methods:
            if mode in {"rag_hyde", "snap_hyre"}:
                model_count = 3 if dataset == "BarExamQA" else 2
                row = aggregate_topk_rows(rows, dataset, mode, f"Mean over {model_count} models")
            else:
                candidates = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode]
                row = candidates[0] if candidates else None
            if not row:
                continue
            hit_values = [100 * float(row["hit"][k]) for k in TOPK_KS]
            mrr_values = [100 * float(row["mrr"][k]) for k in TOPK_KS]
            label = MODE_LABEL[mode] + (" mean" if row["provider"] == "mean" else "")
            axes[0, col].plot(TOPK_KS, hit_values, marker="o", linewidth=2.0, color=COLORS[mode], label=label)
            axes[1, col].plot(TOPK_KS, mrr_values, marker="o", linewidth=2.0, color=COLORS[mode], label=label)
        axes[0, col].set_title("HousingQA (state-filtered)" if dataset == "HousingQA" else DATASET_LABEL[dataset])
        axes[1, col].set_xlabel("k")
        axes[0, col].set_ylabel("Hit@k (%)")
        axes[1, col].set_ylabel("MRR@k (%)")
        for row_ax in axes[:, col]:
            row_ax.set_xticks(TOPK_KS)
            row_ax.set_ylim(bottom=0)
        axes[0, col].legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylim(0, 20)
    axes[1, 0].set_ylim(0, 10)
    housing_hit_vals = []
    housing_mrr_vals = []
    for row in rows:
        if row["dataset"] == "HousingQA":
            housing_hit_vals.extend([100 * float(row["hit"][k]) for k in TOPK_KS])
            housing_mrr_vals.extend([100 * float(row["mrr"][k]) for k in TOPK_KS])
    if housing_hit_vals:
        axes[0, 1].set_ylim(0, nice_upper(housing_hit_vals, floor=40.0, ceiling=100.0))
    if housing_mrr_vals:
        axes[1, 1].set_ylim(0, nice_upper(housing_mrr_vals, floor=25.0, ceiling=100.0))
    fig.savefig(FIGURES / "35_topk_retrieval_curves.png", dpi=240)
    plt.close(fig)


def fig_barexam_retrieval_deltas(rows: list[dict[str, object]]) -> None:
    raw_candidates = [
        row for row in rows
        if row["dataset"] == "BarExamQA" and row["mode"] == "rag_simple"
    ]
    if not raw_candidates:
        return
    raw = raw_candidates[0]
    methods = ["rag_hyde", "snap_hyre"]
    provider_rows = [
        row for row in rows
        if row["dataset"] == "BarExamQA" and row["provider"] in PROVIDERS and row["mode"] in methods
    ]
    if not provider_rows:
        return

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = [0, 1]
    width = 0.34
    all_vals = []
    for offset, mode in zip([-width / 2, width / 2], methods):
        vals = []
        for metric in ["hit", "mrr"]:
            metric_vals = []
            for provider in PROVIDERS:
                matches = [
                    row for row in provider_rows
                    if row["provider"] == provider and row["mode"] == mode
                ]
                if matches:
                    metric_vals.append(100 * (float(matches[0][metric][5]) - float(raw[metric][5])))
            vals.append(mean(metric_vals) if metric_vals else math.nan)
        all_vals.extend([v for v in vals if not math.isnan(v)])
        bars = ax.bar(
            [i + offset for i in x],
            vals,
            width,
            color=COLORS[mode],
            label=MODE_LABEL[mode],
        )
        for bar, val in zip(bars, vals):
            if math.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.22,
                f"+{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
    ax.axhline(0, color="#2f3a45", linewidth=1)
    ax.grid(axis="y", color="#dfe5ec", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Gain over raw RAG (pp)")
    ax.set_xticks(x, ["Hit@5", "MRR@5"])
    ax.set_ylim(0, nice_upper(all_vals, floor=6.0, ceiling=14.0))
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "25_barexam_retrieval_deltas.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_exemplar_probe_rows() -> list[dict[str, object]]:
    rows = []
    for spec in EXEMPLAR_PROBES:
        row = {
            "dataset": spec["dataset"],
            "metric": spec["metric"],
            "kind": spec["kind"],
        }
        for key in ["raw", "snap_hyre", "exemplar"]:
            n, hit5, mrr5 = retrieval_probe_metric(ROOT / spec[key], spec["kind"])
            row[f"{key}_n"] = n
            row[f"{key}_hit5"] = hit5
            row[f"{key}_mrr5"] = mrr5
        rows.append(row)
    return rows


def table_exemplar_probe(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Probe-only 20-question retrieval comparison for Gemma 4 26B with one fixed real-passage exemplar per dataset. Cells are Hit@5/MRR@5 percentages. These rows are not part of the main matrix.}",
        r"\label{tab:exemplar_probe}",
        r"\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Raw & Snap-HyRE & Exemplar \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(str(row['dataset']))} & "
            f"{100 * row['raw_hit5']:.1f}/{100 * row['raw_mrr5']:.1f} & "
            f"{100 * row['snap_hyre_hit5']:.1f}/{100 * row['snap_hyre_mrr5']:.1f} & "
            f"{100 * row['exemplar_hit5']:.1f}/{100 * row['exemplar_mrr5']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "exemplar_probe_q20.tex").write_text("\n".join(lines) + "\n")


def fig_exemplar_probe(rows: list[dict[str, object]]) -> None:
    labels = [str(row["dataset"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    series = [
        ("raw", "Raw", "#2f6f9f"),
        ("snap_hyre", "Snap-HyRE", "#d4942f"),
        ("exemplar", "Exemplar", "#2f855a"),
    ]
    for offset, (key, label, color) in zip([-width, 0, width], series):
        vals = [100 * float(row[f"{key}_hit5"]) for row in rows]
        bars = ax.bar([i + offset for i in x], vals, width, label=label, color=color)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.2,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Hit@5 (%)")
    ax.set_title("Probe-only exemplar changes retrieval exposure (n=20)")
    all_vals = [100 * float(row[f"{key}_hit5"]) for row in rows for key in ["raw", "snap_hyre", "exemplar"]]
    ax.set_ylim(0, nice_upper(all_vals, floor=20.0, ceiling=100.0))
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "30_exemplar_probe_q20.png", dpi=240)
    plt.close(fig)


def write_exemplar_metrics_csv(rows: list[dict[str, object]]) -> None:
    out = FIGURES / "exemplar_probe_q20_metrics.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mode", "n", "hit5_or_same_source5", "mrr5", "source"])
        for row in rows:
            for key, label in [
                ("raw", "raw_question"),
                ("snap_hyre", "canonical_snap_hyre"),
                ("exemplar", "snap_hyre_exemplar_realpassage"),
            ]:
                writer.writerow(
                    [
                        row["dataset"],
                        label,
                        row[f"{key}_n"],
                        f"{float(row[f'{key}_hit5']):.6f}",
                        f"{float(row[f'{key}_mrr5']):.6f}",
                        "docs/retrieval_passage_exemplar_probe_2026-05-20.md",
                    ]
                )


def write_metrics_csv(rows: dict[tuple[str, str, str], SignedRow], retrieval: dict[tuple[str, str, str], SignedRow]) -> None:
    out = FIGURES / "current_figure_metrics.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "provider", "mode", "accuracy", "hit5", "mrr5", "source"])
        for key in sorted(set(rows) | set(retrieval)):
            row = rows.get(key)
            ret = retrieval.get(key)
            writer.writerow(
                [
                    key[0],
                    key[1],
                    key[2],
                    "" if row is None or row.accuracy is None else f"{row.accuracy:.6f}",
                    "" if ret is None or ret.hit5 is None else f"{ret.hit5:.6f}",
                    "" if ret is None or ret.mrr5 is None else f"{ret.mrr5:.6f}",
                    "docs/signoff_log.md",
                ]
            )


def fig_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x,
        y,
        w,
        h,
        text,
        face,
        edge="#2f3a45",
        fontsize=10,
        weight="normal",
        linestyle="-",
        lw=1.25,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=lw,
            linestyle=linestyle,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#1d2630",
            weight=weight,
        )

    def arrow(start, end, color="#4b5663", rad=0.0, linestyle="-", lw=1.45):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=lw,
                linestyle=linestyle,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    def label(x, y, text, color="#4b5663", ha="center"):
        ax.text(
            x,
            y,
            text,
            ha=ha,
            va="center",
            fontsize=8.4,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
        )

    colors = {
        "call": ("#eaf2f8", "#2f6f9f"),
        "answer": ("#fff3d6", "#d4942f"),
        "search": ("#e7f6ef", "#2f855a"),
        "neutral": ("#ffffff", "#8290a3"),
    }

    # Legend: color carries the role of each object in the pipeline.
    legend_y = 0.93
    legend_items = [
        (0.20, "model call", "call"),
        (0.38, "answer/reasoning", "answer"),
        (0.56, "search/evidence", "search"),
        (0.78, "question", "neutral"),
    ]
    for x, text, key in legend_items:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.038, legend_y - 0.020),
                0.024,
                0.033,
                boxstyle="round,pad=0.003,rounding_size=0.005",
                linewidth=1.0,
                edgecolor=colors[key][1],
                facecolor=colors[key][0],
            )
        )
        ax.text(x - 0.006, legend_y, text, ha="left", va="center", fontsize=7.4, color="#4b5663")

    # Main flow.
    box(0.035, 0.43, 0.12, 0.16, "legal\nquestion", *colors["neutral"], weight="bold")
    box(0.21, 0.37, 0.15, 0.25, "Call 1\nmodel initial\nreasoning", *colors["call"], fontsize=10.2, weight="bold")
    box(0.43, 0.66, 0.17, 0.16, "private initial\nreasoning", *colors["answer"], fontsize=10.2, weight="bold")
    box(0.43, 0.33, 0.17, 0.16, "search text\nfor retrieval", *colors["search"], fontsize=10.0, weight="bold")
    box(0.67, 0.33, 0.12, 0.16, "retrieve\nlegal evidence", *colors["search"], fontsize=9.1)
    box(0.67, 0.08, 0.12, 0.14, "top-5\nevidence", *colors["search"], fontsize=9.6, weight="bold", lw=1.1)
    box(0.86, 0.33, 0.11, 0.16, "Call 2\nanswer from\nevidence", *colors["call"], fontsize=8.7, weight="bold")
    box(0.86, 0.08, 0.11, 0.14, "final\nanswer", *colors["answer"], fontsize=10.2, weight="bold")

    arrow((0.155, 0.51), (0.21, 0.51), "#4b5663")
    arrow((0.36, 0.56), (0.43, 0.73), colors["answer"][1], rad=0.12)
    arrow((0.36, 0.45), (0.43, 0.41), colors["search"][1], rad=-0.08)
    arrow((0.515, 0.66), (0.515, 0.49), colors["answer"][1])
    label(0.54, 0.57, "guides\nsearch text", colors["answer"][1], ha="left")
    arrow((0.60, 0.41), (0.67, 0.41), colors["search"][1])
    arrow((0.73, 0.33), (0.73, 0.22), colors["search"][1])
    arrow((0.79, 0.16), (0.86, 0.36), colors["search"][1], rad=-0.16)
    arrow((0.915, 0.33), (0.915, 0.22), colors["answer"][1])


    fig.tight_layout()
    fig.savefig(FIGURES / "20_snap_hyre_pipeline_art.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    rows, retrieval = parse_signoff()
    write_csv(rows, retrieval)
    table_answer_matrix(rows)
    table_result_anatomy(rows, retrieval)
    table_retrieval_matrix(retrieval)
    table_snap_deltas(rows)
    table_snap_vs_controls(rows)
    table_oracle_summary(rows)
    table_neighbor_dilution(rows)
    table_caveats(rows)
    table_completion(rows)
    operational_records = build_operational_records(rows)
    table_operational_metrics(operational_records)
    set_style()
    fig_method_overview()
    exemplar_rows = build_exemplar_probe_rows()
    table_exemplar_probe(exemplar_rows)
    topk_rows = build_topk_retrieval_rows()
    housing_filter_rows = build_housing_metadata_filter_rows()
    table_topk_retrieval(topk_rows)
    table_housing_metadata_filter(housing_filter_rows)
    fig_topk_retrieval_curves(topk_rows)
    fig_barexam_retrieval_deltas(topk_rows)
    write_exemplar_metrics_csv(exemplar_rows)
    write_topk_retrieval_csv(topk_rows)
    write_operational_csv(operational_records)
    write_metrics_csv(rows, retrieval)
    print(
        f"built {len(rows)} audited answer rows, {len(retrieval)} retrieval rows, "
        f"{len(exemplar_rows)} exemplar-probe rows, and {len(topk_rows)} top-k retrieval rows"
    )


if __name__ == "__main__":
    main()
