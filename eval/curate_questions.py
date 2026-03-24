"""Analyze existing eval logs to find high-signal diagnostic questions.

Reads JSONL detail logs and text summary logs, cross-references results across
different eval modes, and outputs a ranked candidate list for curation.

Usage:
    uv run python eval/curate_questions.py                  # Analyze and print candidates
    uv run python eval/curate_questions.py --write          # Also write curated_30.csv
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

LOGS_DIR = "logs"


def load_jsonl_results(path: str) -> dict[str, dict]:
    """Load JSONL detail log, indexed by label."""
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record.get("label", "")
            if label:
                results[label] = record
    return results


def parse_txt_results(path: str) -> dict[str, dict]:
    """Parse text-format eval logs (legacy format) for per-question results."""
    results = {}
    current_label = None

    with open(path) as f:
        for line in f:
            # Match: [N/M] Evaluating qa_label...
            m = re.match(r'\[(\d+)/\d+\]\s+Evaluating\s+(qa_\S+)\.\.\.', line)
            if m:
                current_label = m.group(2)
                continue

            # Match: [N/M] qa_label  PASS/FAIL  gold=X pred=Y
            m = re.match(r'\[\d+/\d+\]\s+(qa_\S+)\s+(PASS|FAIL)\s+gold=(\w+)\s+pred=(\w+)', line)
            if m:
                label = m.group(1)
                results[label] = {
                    "label": label,
                    "is_correct": m.group(2) == "PASS",
                    "correct_answer": m.group(3),
                    "predicted_answer": m.group(4),
                }
                continue

            # Match: -> Result: CORRECT/WRONG (Ans: X, Chose: Y)
            if current_label and "-> Result:" in line:
                m = re.search(r'Result:\s+(CORRECT|WRONG)\s+\(Ans:\s+(\w+),\s+Chose:\s+(\w+)\)', line)
                if m:
                    results[current_label] = {
                        "label": current_label,
                        "is_correct": m.group(1) == "CORRECT",
                        "correct_answer": m.group(2),
                        "predicted_answer": m.group(3),
                    }
                current_label = None

    return results


def discover_logs() -> dict[str, dict[str, dict]]:
    """Discover and load all eval logs, grouped by mode."""
    mode_results = {}

    # Map filename patterns to mode names
    mode_patterns = [
        ("eval_llm_only", "llm_only"),
        ("eval_baseline_deepseek", "llm_only"),
        ("eval_golden_deepseek", "golden_passage"),
        ("eval_bm25_baseline", "rag_simple"),
        ("eval_rag_rewrite_deepseek", "rag_rewrite"),
        ("eval_qa_deepseek", "full_pipeline"),
    ]

    for filename in sorted(os.listdir(LOGS_DIR)):
        filepath = os.path.join(LOGS_DIR, filename)

        for pattern, mode in mode_patterns:
            if pattern not in filename:
                continue

            # Prefer JSONL detail logs
            if filename.endswith("_detail.jsonl"):
                results = load_jsonl_results(filepath)
                if results:
                    # Keep the largest result set per mode
                    if mode not in mode_results or len(results) > len(mode_results[mode]):
                        mode_results[mode] = results
                        print(f"  Loaded {mode:<16} ({len(results):>4} questions) from {filename}")
                break

            # Fall back to text logs
            if filename.endswith(".txt"):
                results = parse_txt_results(filepath)
                if results and mode not in mode_results:
                    mode_results[mode] = results
                    print(f"  Loaded {mode:<16} ({len(results):>4} questions) from {filename} (txt)")
                break

    return mode_results


def analyze_questions(mode_results: dict[str, dict[str, dict]]) -> pd.DataFrame:
    """Cross-reference results across modes to find diagnostic questions."""
    # Collect all labels
    all_labels = set()
    for results in mode_results.values():
        all_labels.update(results.keys())

    rows = []
    for label in sorted(all_labels):
        row = {"label": label}
        for mode, results in mode_results.items():
            if label in results:
                row[f"{mode}_correct"] = results[label].get("is_correct", False)
                row[f"{mode}_pred"] = results[label].get("predicted_answer", "?")
            else:
                row[f"{mode}_correct"] = None
                row[f"{mode}_pred"] = None

        # Extract subject from label
        parts = label.split("_")
        row["subject"] = parts[1] if len(parts) > 1 else "unknown"
        row["idx"] = parts[-1] if len(parts) > 1 else label

        # Classify diagnostic value
        reasons = []
        llm = row.get("llm_only_correct")
        rag_rw = row.get("rag_rewrite_correct")
        rag_s = row.get("rag_simple_correct")
        pipeline = row.get("full_pipeline_correct")
        golden = row.get("golden_passage_correct")

        if llm is True and pipeline is False:
            reasons.append("RAG_NOISE (LLM correct, pipeline wrong)")
        if llm is True and rag_rw is False:
            reasons.append("REWRITE_NOISE (LLM correct, rag_rewrite wrong)")
        if llm is False and rag_rw is True:
            reasons.append("RAG_HELPS (LLM wrong, rag_rewrite correct)")
        if llm is False and pipeline is True:
            reasons.append("PIPELINE_HELPS (LLM wrong, pipeline correct)")
        if rag_s is False and rag_rw is True:
            reasons.append("REWRITE_HELPS (simple_rag wrong, rewrite correct)")
        if golden is False and llm is True:
            reasons.append("GOLDEN_MISLEADS (golden wrong, LLM correct)")
        if llm is False and golden is True:
            reasons.append("GOLDEN_HELPS (LLM wrong, golden correct)")

        # Count how many modes got it right
        mode_correct = [v for k, v in row.items() if k.endswith("_correct") and v is not None]
        row["modes_correct"] = sum(1 for v in mode_correct if v)
        row["modes_total"] = len(mode_correct)
        row["diagnostic_reasons"] = "; ".join(reasons) if reasons else "CONSISTENT"
        row["is_diagnostic"] = len(reasons) > 0
        row["diagnostic_count"] = len(reasons)

        rows.append(row)

    return pd.DataFrame(rows)


def select_curated(df: pd.DataFrame, n_per_subject: int = 5) -> pd.DataFrame:
    """Select curated questions: diagnostic ones first, balanced by subject."""
    # Prioritize diagnostic questions
    diagnostic = df[df["is_diagnostic"]].sort_values("diagnostic_count", ascending=False)
    consistent = df[~df["is_diagnostic"]]

    selected = []
    subjects = sorted(df["subject"].unique())
    target_per_subject = n_per_subject

    for subject in subjects:
        subj_diag = diagnostic[diagnostic["subject"] == subject]
        subj_cons = consistent[consistent["subject"] == subject]

        # Take diagnostic first, then fill with consistent
        subj_selected = list(subj_diag["label"].head(target_per_subject))
        remaining = target_per_subject - len(subj_selected)
        if remaining > 0:
            subj_selected.extend(list(subj_cons["label"].head(remaining)))

        selected.extend(subj_selected)

    return df[df["label"].isin(selected)].copy()


def main():
    parser = argparse.ArgumentParser(description="Analyze logs to curate diagnostic questions")
    parser.add_argument("--write", action="store_true", help="Write curated_30.csv")
    parser.add_argument("--n-per-subject", type=int, default=5, help="Questions per subject (default: 5)")
    args = parser.parse_args()

    print("Discovering eval logs...\n")
    mode_results = discover_logs()

    if not mode_results:
        print("\nNo eval logs found. Run some evaluations first.")
        sys.exit(1)

    print(f"\nModes found: {list(mode_results.keys())}")
    print("Analyzing cross-mode agreement...\n")

    df = analyze_questions(mode_results)

    # Summary
    diagnostic = df[df["is_diagnostic"]]
    print(f"Total questions analyzed: {len(df)}")
    print(f"Diagnostic (mode disagreement): {len(diagnostic)}")
    print(f"Consistent (all modes agree): {len(df) - len(diagnostic)}")

    # Breakdown by reason
    print("\nDiagnostic breakdown:")
    all_reasons = []
    for reasons in diagnostic["diagnostic_reasons"]:
        for r in reasons.split("; "):
            tag = r.split(" (")[0] if " (" in r else r
            all_reasons.append(tag)

    from collections import Counter
    for reason, count in Counter(all_reasons).most_common():
        print(f"  {reason:<25} {count}")

    # Subject breakdown
    print(f"\nBy subject:")
    for subject in sorted(df["subject"].unique()):
        subj = df[df["subject"] == subject]
        n_diag = len(subj[subj["is_diagnostic"]])
        print(f"  {subject:<15} {len(subj)} total, {n_diag} diagnostic")

    # Show top diagnostic questions
    print(f"\nTop diagnostic questions:")
    print(f"{'Label':<35} {'Reasons'}")
    print("-" * 90)
    for _, row in diagnostic.sort_values("diagnostic_count", ascending=False).head(20).iterrows():
        print(f"  {row['label']:<33} {row['diagnostic_reasons']}")

    if args.write:
        # Load full QA data to get question text and choices
        qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")
        curated_df = select_curated(df, n_per_subject=args.n_per_subject)
        curated_labels = set(curated_df["label"].values)

        # Build question-text → label index from the detail logs that have question text.
        # Old eval scripts used different label formats, so we match via question text.
        question_to_label = {}
        for filename in sorted(os.listdir(LOGS_DIR)):
            if not filename.endswith("_detail.jsonl"):
                continue
            filepath = os.path.join(LOGS_DIR, filename)
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    q = record.get("question", "")
                    label = record.get("label", "")
                    if q and label and label in curated_labels:
                        question_to_label[q[:200]] = label  # use prefix as key

        # Match QA rows by question text prefix
        curated_rows = []
        for _, row in qa.iterrows():
            q_prefix = str(row["question"])[:200]
            if q_prefix in question_to_label:
                label = question_to_label[q_prefix]
                reason_row = curated_df[curated_df["label"] == label].iloc[0]
                row_dict = row.to_dict()
                row_dict["reason"] = reason_row["diagnostic_reasons"]
                curated_rows.append(row_dict)

        if curated_rows:
            out = pd.DataFrame(curated_rows)
            out_path = os.path.join(os.path.dirname(__file__), "question_sets", "curated_30.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out.to_csv(out_path, index=False)
            print(f"\nWrote {len(out)} curated questions to {out_path}")
        else:
            print("\nNo questions matched between analysis and QA dataset.")
            print("This may be because label formats differ. Check label matching.")


if __name__ == "__main__":
    main()
