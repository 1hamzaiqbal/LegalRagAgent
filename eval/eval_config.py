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
    dataset: str = "barexam"          # "barexam" | "housing" | "legal_rag" | "australian" | "casehold"
    embedding_model: str = ""         # override embedding model for retrieval (e.g., "BAAI/bge-m3")


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
    "snap_hyde_aligned":        "Snap-HyDE aligned: HyDE for dense retrieval, raw question for cross-encoder reranking",
    "gap_hyde":                 "Gap-informed HyDE: snap + gaps + evidence in final (full context)",
    "gap_hyde_ev":              "Gap-informed HyDE: evidence only in final (no snap, no gap labels)",
    "gap_hyde_nosnap":          "Gap-informed HyDE: gaps + evidence in final (no snap answer)",
    "gap_hyde_flat":            "Gap-informed HyDE: snap + flat evidence in final (no gap structure)",
    "gap_rag":                  "Gap-informed RAG: snap + gaps + evidence in final (full context)",
    "gap_rag_nosnap":           "Gap RAG without snap in final — tests anchoring hypothesis",
    "gap_vectorless":           "Gap + vectorless: per-gap LLM knowledge, no snap in final",
    "subagent_rag":             "Subagent RAG: per-gap RAG + LLM summarization → reports only (no snap)",
    "subagent_hybrid":          "Subagent hybrid: per-gap RAG + LLM knowledge → combined reports (no snap)",
    "subagent_rag_evidence":    "Subagent RAG + evidence: reports + raw passages (no snap)",
    "snap_rag":                 "Snap + simple RAG: snap answer then retrieve with raw question, re-answer with both",
    "snap_rag_nosnap":          "Snap + simple RAG: snap then retrieve, but final call only sees evidence (control)",
    "vectorless_direct":        "Vectorless RAG: snap → generate doctrinal note (rule/exception/trigger/alt) → answer",
    "vectorless_role":          "Vectorless RAG: snap → role-conditioned note (textbook/casebook/barprep via --tag) → answer",
    "vectorless_elements":      "Vectorless RAG: snap → identify dispositive legal elements → answer",
    "vectorless_choice_map":    "Vectorless RAG: snap → map rule + distractor + decisive fact → answer",
    "vectorless_hybrid":        "Hybrid: vectorless knowledge + vector RAG evidence pooled → answer (4 calls)",
    "vectorless_keyword":       "Keyword search: snap → LLM generates search terms → multi-keyword retrieval → answer",
    "rag_devil_hyde":           "Devil's advocate HyDE: retrieve for AND against snap answer",
    "rag_top2_hyde":            "Top-2 HyDE: retrieve for snap answer + second-choice answer",
    "confidence_gated":         "Confidence-gated: 3 snap votes, unanimous=skip RAG, disagreement=Snap-HyDE",
    "decompose":                "Decompose-then-answer: split into sub-questions, answer each, synthesize (no RAG)",
    "decompose_rag":            "Decompose + Snap-HyDE: sub-questions with per-issue retrieval, then synthesize",
    "ce_threshold":             "CE-thresholded Snap-HyDE: discard low-scoring evidence, fall back to snap answer",
    "conf_ce_threshold":        "Confidence-gated + CE threshold: 3-vote gating, then CE threshold on RAG path",
    "snap_hyde_aspect":         "Snap-HyDE + aspect queries: HyDE passage + rule/exception queries for diverse retrieval",
    "ce_threshold_k3":          "CE-thresholded Snap-HyDE with k=3: fewer passages, higher quality",
    "self_verify":              "Self-verification: snap answer then review for errors (2 calls, no RAG)",
    "double_snap":              "Double-snap: two answers, agree=use, disagree=CE-threshold RAG (2-4 calls)",
    "snap_debate":              "Snap-debate: snap then adversarial critique (2 calls, no RAG)",
}


def load_questions(config: EvalConfig) -> pd.DataFrame:
    """Load questions based on config.questions: 'curated', 'full', or integer N."""
    if config.dataset == "housing":
        return _load_housing_questions(config)
    if config.dataset == "legal_rag":
        return _load_generic_questions(config, "datasets/legal_rag_qa/questions.csv")
    if config.dataset == "australian":
        return _load_generic_questions(config, "datasets/australian_legal_qa/questions.csv")
    if config.dataset == "casehold":
        return _load_generic_questions(config, "datasets/casehold/test.csv")

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


def _load_housing_questions(config: EvalConfig) -> pd.DataFrame:
    """Load HousingQA questions (Yes/No format)."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qa = pd.read_csv(os.path.join(base, "datasets/housing_qa/questions.csv"))

    if config.questions == "full":
        return qa.reset_index(drop=True)

    n = int(config.questions)
    return qa.sample(n=min(n, len(qa)), random_state=config.seed).reset_index(drop=True)


def _load_generic_questions(config: EvalConfig, csv_path: str) -> pd.DataFrame:
    """Load questions from a CSV, sample N if requested."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qa = pd.read_csv(os.path.join(base, csv_path))

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


def extract_answer_mc5(text: str) -> str | None:
    """Extract 5-way multiple-choice answer letter (A-E) from LLM response."""
    cleaned = text.replace('*', '')
    patterns = [
        r'(?:Answer|ANSWER)[:\s]*\(?([A-E])\)?',
        r'\b([A-E])\b\s*(?:is correct|is the (?:best|correct|strongest))',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        if matches:
            return matches[-1]
    return None


def extract_answer_yn(text: str) -> str | None:
    """Extract Yes/No answer from LLM response."""
    cleaned = text.replace('*', '').strip()
    patterns = [
        r'(?:Answer|ANSWER)[:\s]*(Yes|No)\b',
        r'(?:Final answer|FINAL ANSWER)[:\s]*(Yes|No)\b',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        if matches:
            return matches[-1].capitalize()
    # Fallback: last standalone Yes/No in the text
    matches = re.findall(r'\b(Yes|No)\b', cleaned, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    return None


def format_question_prompt(row: pd.Series, dataset: str = "barexam") -> str:
    """Format a question into a standard prompt string."""
    if dataset == "housing":
        return format_housing_prompt(row)
    if dataset == "casehold":
        return format_casehold_prompt(row)
    if dataset in ("legal_rag", "australian"):
        return format_open_prompt(row)

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


def format_housing_prompt(row: pd.Series) -> str:
    """Format a HousingQA Yes/No question into a prompt string."""
    state = str(row.get("state", ""))
    question = str(row["question"])
    prompt = f"Regarding {state} housing law:\n\n{question}"
    prompt += "\n\nAnswer Yes or No. Provide your answer as: Answer: Yes or Answer: No"
    return prompt


def format_casehold_prompt(row: pd.Series) -> str:
    """Format a CaseHOLD 5-way MC question."""
    context = str(row["question"])  # 'question' col holds citing context
    choices = []
    for letter in ["A", "B", "C", "D", "E"]:
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]):
            choices.append(f"  ({letter}) {row[col]}")

    prompt = (
        f"The following excerpt from a court opinion cites a legal holding. "
        f"Which of the following holdings is most likely being referenced?\n\n"
        f"## Citing Context\n{context}\n\n"
        f"## Holdings\n" + "\n".join(choices) +
        f"\n\nProvide your answer as: Answer: (X)"
    )
    return prompt


def format_open_prompt(row: pd.Series) -> str:
    """Format an open-ended legal question (legal-rag-qa, australian)."""
    question = str(row["question"])
    return f"{question}\n\nProvide a detailed answer."
