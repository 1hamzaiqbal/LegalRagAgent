"""Shared utilities for evaluation scripts."""

import json
import os
import re
import sys
import pandas as pd

# Ensure parent directory is on sys.path for imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_qa_with_gold() -> pd.DataFrame:
    """Load QA pairs whose gold passages exist in the current vector store."""
    from rag_utils import get_vectorstore
    vs = get_vectorstore()
    corpus_size = vs._collection.count()
    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")
    passages = pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", nrows=corpus_size)
    passage_ids = set(passages["idx"].tolist())
    qa_in = qa[qa["gold_idx"].isin(passage_ids)].copy()

    def _full_q(row):
        prompt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
        q = str(row["question"])
        return (prompt + " " + q).strip()

    qa_in["full_q"] = qa_in.apply(_full_q, axis=1)
    return qa_in


def _row_to_query(index, row) -> dict:
    subj_name = str(row["subject"]).lower().replace(" ", "").replace(".", "")
    return {
        "label": f"qa_{subj_name}_{index}",
        "question": row["full_q"],
        "gold_idx": row["gold_idx"],
        "correct_answer": row["answer"],
        "choices": {
            "A": str(row["choice_a"]) if pd.notna(row["choice_a"]) else "",
            "B": str(row["choice_b"]) if pd.notna(row["choice_b"]) else "",
            "C": str(row["choice_c"]) if pd.notna(row["choice_c"]) else "",
            "D": str(row["choice_d"]) if pd.notna(row["choice_d"]) else "",
        },
        "subject": row["subject"],
        "gold_passage": str(row["gold_passage"]) if pd.notna(row.get("gold_passage")) else "",
    }


def select_qa_queries(n: int = 10):
    """Select a deterministic random sample of questions using a fixed seed."""
    qa = load_qa_with_gold()
    sampled_qa = qa.sample(n=min(n, len(qa)), random_state=42)
    return [_row_to_query(index, row) for index, row in sampled_qa.iterrows()]


def select_qa_queries_by_labels(labels: list[str]) -> list[dict]:
    """Return QA records in the exact label order requested."""
    qa = load_qa_with_gold()
    by_label = {}
    for index, row in qa.iterrows():
        query = _row_to_query(index, row)
        by_label[query["label"]] = query
    missing = [label for label in labels if label not in by_label]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Unknown QA labels: {preview}")
    return [by_label[label] for label in labels]


def load_eval_labels(path: str) -> list[str]:
    """Load an ordered label list from a JSON, JSONL, or prior detail log."""
    if path.endswith(".jsonl"):
        labels = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                labels.append(json.loads(line)["label"])
        return labels

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
        return [str(item) for item in payload["labels"]]
    raise ValueError(f"Unsupported label manifest format: {path}")


MC_LETTER_PATTERNS = [
    r'\*\*answer:\s*\(([A-D])\)\*\*',
    r'\banswer\s+is\s+\(?([A-D])\)?\b',
    r'\bcorrect\s+answer[:\s]+\(?([A-D])\)?\b',
    r'\b\(([A-D])\)\s+is\s+correct\b',
    r'\boption\s+\(?([A-D])\)?\s+is\s+correct\b',
    r'\bselect(?:ing|ed|s)?\s+\(?([A-D])\)?\b',
]


def extract_mc_letter(answer: str) -> str | None:
    """Extract the MC letter (A-D) from an answer string, or None if not found.

    Uses the LAST match if the LLM self-corrects mid-response.
    """
    if not answer:
        return None
    last_match = None
    for pat in MC_LETTER_PATTERNS:
        for m in re.finditer(pat, answer, re.IGNORECASE):
            last_match = m.group(1).upper()
    return last_match


def check_mc_correctness(answer: str, correct_letter: str) -> bool:
    """Check if the answer selects the correct MC letter."""
    if not correct_letter or not answer:
        return False
    chosen = extract_mc_letter(answer)
    return chosen == correct_letter.upper() if chosen else False


def capture_balance():
    """Capture current DeepSeek balance. Returns (balance_dict, initial_totals)."""
    from main import _get_deepseek_balance
    balance = _get_deepseek_balance()
    totals = {}
    if balance.get("is_available"):
        for info in balance.get("balance_infos", []):
            totals[info.get("currency")] = float(info.get("total_balance", 0.0))
    return balance, totals


def compute_cost(initial_totals: dict) -> list[str]:
    """Compute cost strings by comparing current balance against initial_totals."""
    from main import _get_deepseek_balance
    cost_strs = []
    if not initial_totals:
        return cost_strs
    final_balance = _get_deepseek_balance()
    if final_balance.get("is_available"):
        for fin_info in final_balance.get("balance_infos", []):
            currency = fin_info.get("currency")
            fin_tot = float(fin_info.get("total_balance", 0.0))
            init_tot = initial_totals.get(currency, 0.0)
            spent = init_tot - fin_tot
            if spent > 0.0001:
                cost_strs.append(f"{spent:.4f} {currency}")
            elif init_tot > 0:
                cost_strs.append(f"< 0.01 {currency}")
    return cost_strs
