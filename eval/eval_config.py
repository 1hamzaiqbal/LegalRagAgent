"""Evaluation configuration, question loading, and answer extraction.

Shared by eval_harness.py and eval_analyze.py.
"""
import os
import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class EvalConfig:
    mode: str = "full_pipeline"       # full_pipeline | llm_only | rag_rewrite | rag_simple | golden_passage
    provider: str = "deepseek"        # any key from llm_config.PROVIDERS
    questions: str = "30"             # "curated" | "full" | integer N
    seed: int = 42
    skill_dir: str = "skills"
    verbose: bool = False
    tag: str = ""                     # optional label for the run
    source_filter: str = ""           # optional metadata filter, e.g. "mbe" to search MBE docs only


EVAL_MODES = {
    "full_pipeline":       "Full agentic pipeline (planner → executor → synthesizer)",
    "llm_only":            "Direct LLM answer, no retrieval",
    "rag_rewrite":         "Query rewrite → hybrid retrieval → synthesize",
    "rag_simple":          "Raw question → hybrid retrieval → synthesize",
    "golden_passage":      "LLM answer with gold passage injected as context",
    "golden_arbitration":       "LLM answers naive, then reviews golden passage (neutral framing)",
    "golden_arb_conservative":  "LLM answers naive, then reviews golden passage (biased toward keeping)",
    "rag_arbitration":          "LLM answers naive, then reviews retrieved passages (conservative)",
    "rag_hyde":                 "HyDE: LLM generates hypothetical answer, embeds it to retrieve",
    "rag_hyde_arb":             "HyDE retrieval + snap-then-review arbitration (conservative)",
    "rag_multi_hyde":           "Multi-HyDE: 3 hypothetical passages (rule/exception/application)",
    "rag_snap_hyde":            "Snap-informed HyDE: answer first, then targeted retrieval",
}


def load_questions(config: EvalConfig) -> pd.DataFrame:
    """Load questions based on config.questions: 'curated', 'full', or integer N."""
    if config.questions == "curated":
        path = os.path.join(os.path.dirname(__file__), "question_sets", "curated_30.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Curated question set not found at {path}. "
                "Run eval/curate_questions.py first to generate it."
            )
        return pd.read_csv(path)

    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")

    if config.questions == "full":
        return qa.reset_index(drop=True)

    n = int(config.questions)
    return qa.sample(n=min(n, len(qa)), random_state=config.seed).reset_index(drop=True)


def extract_answer_mc(text: str) -> str | None:
    """Extract multiple-choice answer letter (A-D) from LLM response."""
    # Strip markdown bold markers so **Answer:** (D) becomes Answer: (D)
    cleaned = text.replace('*', '')
    patterns = [
        r'(?:Answer|ANSWER)[:\s]*\(?([A-D])\)?',
        r'\b([A-D])\b\s*(?:is correct|is the (?:best|correct|strongest))',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        if matches:
            return matches[-1]  # last match = conclusion
    return None


def format_question_prompt(row: pd.Series) -> str:
    """Format a bar exam MC question into a standard prompt string."""
    parts = [str(row["question"])]

    choices = []
    for letter in ["A", "B", "C", "D"]:
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]):
            choices.append(f"  ({letter}) {row[col]}")

    if choices:
        parts.append("\n".join(choices))

    parts.append("\nProvide your answer as: Answer: (X)")
    return "\n\n".join(parts)
