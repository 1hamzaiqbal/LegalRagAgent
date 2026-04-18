"""Multi-model evaluation harness for the Legal RAG pipeline.

Supports multiple eval modes, providers, question sets, and skill overrides.
Produces rich per-question JSONL logs and appends run summaries to experiments.jsonl.

Usage:
    uv run python eval/eval_harness.py --mode llm_only --provider deepseek --questions 10
    uv run python eval/eval_harness.py --mode full_pipeline --provider gemma --questions curated
    uv run python eval/eval_harness.py --mode rag_rewrite --provider deepseek --questions 30 --tag "aspect-queries"
    uv run python eval/eval_harness.py --mode golden_passage --provider openai --questions 100
    uv run python eval/eval_harness.py --mode full_pipeline --skill-dir skills_v2 --questions curated
"""
import argparse
import json
import os
import re
import subprocess
import sys
from typing import List
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from eval_config import EvalConfig, EVAL_MODES, load_questions, extract_answer_mc, extract_answer_mc5, extract_answer_yn, format_question_prompt
from main import (
    run as run_pipeline,
    _llm_call,
    _get_metrics,
    _reset_llm_call_counter,
    _parse_json,
    load_skill,
)
from llm_config import get_provider_info, _get_llm_cached
from rag_utils import retrieve_documents_multi_query, get_vectorstore


# ---------------------------------------------------------------------------
# Mode Runner Functions
# ---------------------------------------------------------------------------

def _fmt(row: pd.Series, config: EvalConfig) -> str:
    """Format question prompt based on dataset."""
    return format_question_prompt(row, dataset=config.dataset)


def _extract_answer(text: str, config: EvalConfig) -> str | None:
    """Extract answer using the right extractor for the dataset."""
    if config.dataset == "housing":
        return extract_answer_yn(text)
    if config.dataset == "casehold":
        return extract_answer_mc5(text)
    if config.dataset in ("legal_rag", "australian"):
        return text  # open-ended: return full text, scored by LLM judge
    return extract_answer_mc(text)


def _fmt_intermediate(row: pd.Series, config: EvalConfig) -> str:
    """Format a question for retrieval-side generation steps.

    Unlike `_fmt()`, this strips answer-format instructions and, for MC tasks,
    removes answer letters from the choices so intermediate generators are not
    pushed toward emitting `Answer: (X)` artifacts.
    """
    if config.dataset in ("legal_rag", "australian"):
        return str(row["question"])

    if config.dataset == "housing":
        state = str(row.get("state", ""))
        return f"Regarding {state} housing law:\n\n{row['question']}"

    if config.dataset == "casehold":
        context = str(row["question"])
        holdings = []
        for letter in ["A", "B", "C", "D", "E"]:
            col = f"choice_{letter.lower()}"
            if col in row and pd.notna(row[col]):
                holdings.append(f"- {row[col]}")

        parts = [
            "The following excerpt from a court opinion cites a legal holding.",
            f"## Citing Context\n{context}",
        ]
        if holdings:
            parts.append("## Candidate Holdings\n" + "\n".join(holdings))
        return "\n\n".join(parts)

    stem = str(row["question"])
    choices = []
    for letter in ["A", "B", "C", "D"]:
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]):
            choices.append(f"- {row[col]}")

    if not choices:
        return stem

    return "\n\n".join([
        stem,
        "## Candidate Answer Framing\n" + "\n".join(choices),
    ])


def _contains_answer_artifact(text: str) -> bool:
    """Detect explicit answer labels leaking into intermediate artifacts."""
    if not text:
        return False
    return bool(re.search(r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:", text))


def _sanitize_intermediate_text(text: str, fallback: str = "") -> str:
    """Strip answer-label artifacts and prompt-ish headers from intermediate text."""
    cleaned = text or ""

    # Remove standalone answer lines such as `Answer: (B)` or `Final Answer: Yes`.
    cleaned = re.sub(
        r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:\s*(?:\(?[A-E]\)?|yes|no|irrelevant)\s*$",
        "",
        cleaned,
    )

    # Remove common generated headings while keeping the substantive passage.
    cleaned = re.sub(
        r"(?im)^\s*(?:\*\*)?(?:relevant legal passage|legal reference passage|passage)(?:\*\*)?\s*:\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*(?:\*\*)?(?:relevant legal passage|legal reference passage|passage)(?:\*\*)?\s*:\s*",
        "",
        cleaned,
    )

    lines = []
    prev_blank = False
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        lines.append(line)
        prev_blank = is_blank

    cleaned = "\n".join(lines).strip()
    fallback = (fallback or "").strip()
    return cleaned or fallback


def _question_only_hyde_user(question_text: str) -> str:
    """Structured user payload for question-only HyDE generation."""
    return (
        "## Task\n"
        "Write a short passage from a legal reference that would be useful for answering the "
        "question below.\n\n"
        "## Original Legal Question\n"
        f"{question_text}\n\n"
        "## Passage Requirements\n"
        "- Write 2-3 sentences in legal reference style\n"
        "- State the controlling rule, doctrine, holding, exception, or principle directly\n"
        "- Focus on the legal issue most likely to determine the answer\n"
        "- Do not discuss the question itself, answer in exam style, or say 'the answer is'\n"
    )


def _snap_hyde_user(question_text: str, snap_answer: str, gap_focus: str = "") -> str:
    """Shared user payload for snap-informed HyDE generation."""
    reasoning = (snap_answer or "").strip()
    if gap_focus:
        reasoning = (
            f"{reasoning}\n\n"
            f"Focus on verifying or correcting this specific issue:\n{gap_focus}"
        ).strip()
    return (
        f"## Student's Answer and Reasoning\n{reasoning}\n\n"
        f"## Original Question\n{question_text}"
    )


def _generate_hyde(config: EvalConfig, role: str, user: str, label: str, fallback: str) -> dict:
    """Generate and sanitize a HyDE-style intermediate passage."""
    raw = _llm_call(_system_prompt(config, role), user, label=label)
    return {
        "text": _sanitize_intermediate_text(raw, fallback=fallback),
        "raw": raw,
        "contains_answer": _contains_answer_artifact(raw),
    }


def _report_prompt(max_words: int = 100, include_model_knowledge: bool = False) -> str:
    """Shared prompt for report-writing intermediate steps."""
    if include_model_knowledge:
        return (
            "You are a legal research assistant. You have retrieved passages AND your own legal knowledge. "
            "Write a brief, focused report answering the sub-question by combining both sources. "
            "State what the law says directly. Flag if retrieved passages conflict with known law. "
            f"No answer letters. Keep under {max_words} words."
        )
    return (
        "You are a legal research assistant. Read the retrieved passages and write a brief, focused "
        "report that states the relevant legal rules, doctrines, holdings, and uncertainties directly. "
        "If the passages are irrelevant or unhelpful, say so clearly. "
        f"No answer letters. Keep under {max_words} words."
    )


def _generate_report(system: str, user: str, label: str, fallback: str) -> dict:
    """Generate and sanitize a report-style intermediate artifact."""
    raw = _llm_call(system, user, label=label)
    return {
        "text": _sanitize_intermediate_text(raw, fallback=fallback),
        "raw": raw,
        "contains_answer": _contains_answer_artifact(raw),
    }


def _system_prompt(config: EvalConfig, role: str = "answer") -> str:
    """Get dataset-appropriate system prompt."""
    if config.dataset == "housing":
        prompts = {
            "answer": (
                "You are a legal expert specializing in housing law. Answer the Yes/No question below. "
                "Reason step by step, then give your final answer as: Answer: Yes or Answer: No"
            ),
            "rag": (
                "You are a legal expert specializing in housing law. Reason through the question "
                "step by step. Retrieved passages are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Give your final answer as: Answer: Yes or Answer: No"
            ),
            "research": (
                "You are a legal expert specializing in housing law. Reason through the question "
                "step by step. Research findings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Give your final answer as: Answer: Yes or Answer: No"
            ),
            "hyde": (
                "You are a legal textbook author specializing in housing law. Given a legal question, "
                "write a short passage (2-3 sentences) that would appear in a reference guide as the answer. "
                "Write in the style of a legal reference — state the statute, rule, or "
                "regulation directly. Do not discuss the question itself or say 'the answer is'."
            ),
            "snap_hyde": (
                "You are a legal textbook author specializing in housing law. A student has answered a legal question "
                "and provided their reasoning. Write a short passage (2-3 sentences) from a legal reference that "
                "would be most relevant to verifying or correcting this answer. Focus on the specific "
                "statute, regulation, or rule at the heart of the question. Write in reference style — "
                "state the law directly."
            ),
        }
        prompts["devil_hyde"] = (
            "You are a legal textbook author specializing in housing law. A student has answered a legal question. "
            "Your job is to play DEVIL'S ADVOCATE: write a short passage (2-3 sentences) from a legal reference "
            "that would CHALLENGE or CONTRADICT the student's answer. Focus on the rule, exception, or statute "
            "that supports the OPPOSITE conclusion. Write in reference style — state the law directly."
        )
        prompts["top2_snap"] = (
            "You are a legal expert specializing in housing law. Answer the Yes/No question below. "
            "Reason step by step. Identify what your FIRST choice answer is, and also what the ALTERNATIVE "
            "answer would be and why someone might argue for it. "
            "Give your final answer as: Answer: Yes or Answer: No"
        )
        prompts["top2_hyde"] = (
            "You are a legal textbook author specializing in housing law. A student has answered a legal question. "
            "Write a short passage (2-3 sentences) from a legal reference that would support the ALTERNATIVE "
            "or SECOND-CHOICE answer — the answer the student considered but rejected. Focus on the specific "
            "statute, regulation, or rule that would support that alternative. Write in reference style."
        )
        return prompts.get(role, prompts["answer"])
    if config.dataset == "casehold":
        prompts = {
            "answer": (
                "You are a legal expert specializing in case law. Read the citing context from a court opinion "
                "and determine which holding is most likely being referenced. "
                "Reason step by step, then give your final answer as: Answer: (X)"
            ),
            "rag": (
                "You are a legal expert specializing in case law. Reason through the question "
                "step by step. Retrieved holdings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Give your final answer as: Answer: (X)"
            ),
            "research": (
                "You are a legal expert specializing in case law. Reason through the question "
                "step by step. Research findings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Give your final answer as: Answer: (X)"
            ),
            "hyde": (
                "You are a legal textbook author. Given a court opinion excerpt that cites a holding, "
                "write a short passage (2-3 sentences) stating the likely holding being referenced. "
                "Write in the style of a case holding — state the rule directly."
            ),
            "snap_hyde": (
                "You are a legal textbook author. A student has identified what they think is the correct "
                "holding for a citation. Write a short passage (2-3 sentences) from a legal reference "
                "that would be most relevant to verifying this holding. Write in reference style."
            ),
        }
        return prompts.get(role, prompts["answer"])
    if config.dataset in ("legal_rag", "australian"):
        domain = "criminal law" if config.dataset == "legal_rag" else "Australian law"
        prompts = {
            "answer": (
                f"You are a legal expert specializing in {domain}. Answer the question below "
                f"thoroughly and accurately. Provide a detailed answer."
            ),
            "rag": (
                f"You are a legal expert specializing in {domain}. Reason through the question "
                f"step by step. Retrieved passages are provided — use them to verify or "
                f"refine your reasoning, but think through the problem independently first. "
                f"Provide a detailed answer."
            ),
            "research": (
                f"You are a legal expert specializing in {domain}. Reason through the question "
                f"step by step. Research findings are provided — use them to verify or "
                f"refine your reasoning, but think through the problem independently first. "
                f"Provide a detailed answer."
            ),
            "hyde": (
                f"You are a legal textbook author specializing in {domain}. Given a legal question, "
                f"write a short passage (2-3 sentences) that would appear in a reference guide as the answer. "
                f"Write in the style of a legal reference — state the rule directly."
            ),
            "snap_hyde": (
                f"You are a legal textbook author specializing in {domain}. A student has answered a legal question "
                f"and provided their reasoning. Write a short passage (2-3 sentences) from a legal reference that "
                f"would be most relevant to verifying or correcting this answer. Write in reference style."
            ),
        }
        return prompts.get(role, prompts["answer"])
    # BarExam defaults
    prompts = {
        "answer": (
            "You are a legal expert. Answer the multiple-choice question below. "
            "Reason step by step, then give your final answer as: Answer: (X)"
        ),
        "rag": _RAG_SYSTEM,
        "hyde": (
            "You are a legal textbook author. A legal question is provided below. "
            "Write a short passage (2-3 sentences) from a legal reference that would be "
            "most relevant to answering this question. Focus on the specific doctrine, rule, "
            "or exception at the heart of the question. Write in reference style — state the "
            "law directly. Do not discuss the question itself or say 'the answer is'."
        ),
        "snap_hyde": (
            "You are a legal textbook author. A student has answered a legal question and provided "
            "their reasoning. Write a short passage (2-3 sentences) from a legal reference that "
            "would be most relevant to verifying or correcting this answer. Focus on the specific "
            "doctrine, rule, or exception at the heart of the question. Write in reference style — "
            "state the law directly."
        ),
        "devil_hyde": (
            "You are a legal textbook author. A student has answered a legal question. "
            "Your job is to play DEVIL'S ADVOCATE: write a short passage (2-3 sentences) from a legal reference "
            "that would CHALLENGE or CONTRADICT the student's answer. Focus on the doctrine, rule, or exception "
            "that supports the OPPOSITE conclusion. Write in reference style — state the law directly."
        ),
        "top2_snap": (
            "You are a legal expert. Answer the multiple-choice question below. "
            "Reason step by step. Identify what your FIRST choice answer is, and also what your SECOND choice "
            "would be and why it's a plausible alternative. "
            "Give your final answer as: Answer: (X)"
        ),
        "top2_hyde": (
            "You are a legal textbook author. A student has answered a legal question with a multiple-choice selection. "
            "Write a short passage (2-3 sentences) from a legal reference that would support the SECOND-CHOICE "
            "answer — the answer the student considered but rejected. Focus on the specific doctrine, rule, or "
            "exception that makes the alternative answer plausible. Write in reference style — state the law directly."
        ),
    }
    return prompts.get(role, prompts["answer"])


DATASET_COLLECTIONS = {
    "barexam": "legal_passages",
    "housing": "housing_statutes",
    "legal_rag": "legal_rag_passages",
    "australian": "australian_legal",
    "casehold": "casehold_holdings",
}


def _judge_open_answer(question: str, gold: str, predicted: str, config: EvalConfig) -> bool:
    """Use LLM to judge whether an open-ended answer is correct.

    Returns True if the predicted answer captures the key facts from the gold answer.
    Uses a simple binary correct/incorrect judgment to keep scoring consistent.
    """
    judge_system = (
        "You are a legal exam grader. Compare the student's answer to the reference answer. "
        "Judge whether the student's answer captures the key legal facts, rules, and conclusions "
        "from the reference answer. Minor differences in wording or additional context are acceptable. "
        "The student's answer must get the core legal point RIGHT to be correct.\n\n"
        "Respond with exactly one word: CORRECT or INCORRECT"
    )
    judge_user = (
        f"## Question\n{question}\n\n"
        f"## Reference Answer\n{gold}\n\n"
        f"## Student's Answer\n{predicted}"
    )
    verdict = _llm_call(judge_system, judge_user, label="judge")
    return "CORRECT" in verdict.upper()


def _collection_for_config(config: EvalConfig) -> str:
    """Return the ChromaDB collection name for the dataset.

    Supports EVAL_COLLECTION_OVERRIDE env var for embedding A/B testing.
    When set, uses the override collection name instead of the default.
    """
    override = os.getenv("EVAL_COLLECTION_OVERRIDE", "").strip()
    if override:
        return override
    return DATASET_COLLECTIONS.get(config.dataset, "legal_passages")


def run_full_pipeline(row: pd.Series, config: EvalConfig) -> dict:
    """Run the full agentic pipeline and capture complete state."""
    question = format_question_prompt(row, dataset=config.dataset)
    result = run_pipeline(question, print_output=False)

    # Serialize PlanningStep objects
    planning_table = []
    for s in result.get("planning_table", []):
        if hasattr(s, "model_dump"):
            planning_table.append(s.model_dump())
        elif hasattr(s, "__dict__"):
            planning_table.append(vars(s))
        else:
            planning_table.append(s)

    # Check if gold passage was retrieved
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [ev.get("idx", "") for ev in result.get("evidence_store", [])]
    gold_retrieved = gold_idx in retrieved_ids if gold_idx else False

    return {
        "final_answer": result.get("final_answer", ""),
        "collections": result.get("collections", []),
        "planning_table": planning_table,
        "evidence_store": result.get("evidence_store", []),
        "audit_log": result.get("audit_log", []),
        "completeness_verdict": result.get("completeness_verdict", {}),
        "parallel_rounds": result.get("parallel_round", 1) - 1,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_retrieved,
    }


def run_llm_only(row: pd.Series, config: EvalConfig) -> dict:
    """Direct LLM answer with no retrieval."""
    question = _fmt(row, config)
    answer = _llm_call(_system_prompt(config, "answer"), question, label="llm_only")
    return {"final_answer": answer}


def run_golden_passage(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answer with the gold passage injected as context."""
    question = _fmt(row, config)
    gold = str(row.get("gold_passage", ""))
    if not gold or gold == "nan":
        return run_llm_only(row, config)

    system = _system_prompt(config, "rag")
    user = f"## Reference Passage\n{gold}\n\n## Question\n{question}"
    answer = _llm_call(system, user, label="golden_passage")
    return {"final_answer": answer}


def _golden_arb_common(row: pd.Series, config: EvalConfig, arb_system: str, label_prefix: str) -> dict:
    """Shared logic for golden arbitration variants."""
    question = _fmt(row, config)
    gold = str(row.get("gold_passage", ""))
    if not gold or gold == "nan":
        return run_llm_only(row, config)

    # Step 1: Naive LLM answer (the "snap")
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label=f"{label_prefix}/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Show evidence and ask to confirm or revise
    arb_user = (
        f"## Your Previous Answer\n{snap_answer}\n\n"
        f"## Reference Passage\n{gold}\n\n"
        f"## Question\n{question}"
    )
    final_answer = _llm_call(arb_system, arb_user, label=f"{label_prefix}/arbitrate")
    final_letter = _extract_answer(final_answer, config)

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "final_letter": final_letter,
        "changed": snap_letter != final_letter,
    }


def run_golden_arbitration(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then sees golden passage — neutral framing (no bias toward keeping/changing)."""
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given a reference passage that may contain relevant legal authority. "
        "Review the passage carefully against your previous reasoning. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    return _golden_arb_common(row, config, arb_system, "golden_arb")


def run_golden_arb_conservative(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then sees golden passage — conservative framing (biased toward keeping original)."""
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given a reference passage that may contain relevant legal authority. "
        "Review the passage carefully. If the evidence supports your original answer, keep it. "
        "If the evidence clearly points to a different answer, change it. "
        "Do not change your answer unless the evidence gives you a strong reason to. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    return _golden_arb_common(row, config, arb_system, "golden_arb_cons")


def _retrieve_and_format(row: pd.Series, queries: List[str], k: int = 5,
                         label_prefix: str = "rag", where: dict = None,
                         collection: str = "legal_passages",
                         rerank_query: str = None) -> dict:
    """Shared retrieval + evidence formatting. Returns dict with passages, evidence_store, metadata.

    Args:
        rerank_query: If provided, cross-encoder reranks against this text instead of
            the retrieval queries. Decouples dense retrieval from reranking (e.g., HyDE
            for embedding but raw question for cross-encoder).
    """
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None
    vs = get_vectorstore(collection, embedding_model=embedding_model)
    docs = retrieve_documents_multi_query(queries=queries, k=k, vectorstore=vs, where=where,
                                          rerank_query=rerank_query)

    passages = []
    evidence_store = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content
        idx = str(doc.metadata.get("idx", f"{label_prefix}_{i}"))
        ce_score = doc.metadata.get("cross_encoder_score", 0.0)
        passages.append(f"[Source {i}]\n{text}")
        evidence_store.append({
            "idx": idx,
            "text": text,
            "source": doc.metadata.get("source", "unknown"),
            "cross_encoder_score": ce_score,
        })

    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [ev["idx"] for ev in evidence_store]

    return {
        "passages": passages,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
        "max_ce_score": max((ev["cross_encoder_score"] for ev in evidence_store), default=0.0),
    }


def _rewrite_query(question: str, label: str = "rag_rewrite/rewrite") -> List[str]:
    """LLM query rewrite → list of queries (primary + alternatives)."""
    rewrite_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {question}\n"
        f"Authority target: \n"
        f"Retrieval hints: none"
    )
    raw_rewrite = _llm_call(load_skill("query_rewriter"), rewrite_prompt, label=label)
    parsed = _parse_json(raw_rewrite)

    if parsed and "primary" in parsed:
        return [parsed["primary"]] + parsed.get("alternatives", [])
    return [question]


def run_rag_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """HyDE: generate hypothetical answer passage, embed it, retrieve similar real passages."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Generate hypothetical passage
    hyde = _generate_hyde(
        config,
        "hyde",
        _question_only_hyde_user(question_intermediate),
        label="hyde/generate",
        fallback=question_intermediate,
    )

    # Step 2: Retrieve using the hypothetical passage as query
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="hyde",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Answer with evidence
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="hyde/answer")

    return {
        "final_answer": answer,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_multi_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Multi-HyDE: generate 3 hypothetical passages (rule/exception/application), pool retrievals."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    system = (
        "You are a legal textbook author. Given a legal question, write THREE short passages "
        "(2-3 sentences each) that would appear in a study guide, targeting different dimensions:\n"
        "1. RULE: The governing legal rule or doctrine\n"
        "2. EXCEPTION: Key exceptions, defenses, or limitations\n"
        "3. APPLICATION: How the rule applies to specific facts\n\n"
        "Write each passage in the style of a legal reference. Separate with blank lines. "
        "Do not label them or discuss the question itself."
    )
    raw = _llm_call(system, question_intermediate, label="multi_hyde/generate")

    # Split into separate passages for retrieval, filter out empty
    raw_hyde_passages = [p.strip() for p in raw.split("\n\n") if p.strip() and len(p.strip()) > 30]
    hyde_passages = [
        _sanitize_intermediate_text(p, fallback=question_intermediate)
        for p in raw_hyde_passages
    ]
    if not hyde_passages:
        raw_hyde_passages = [raw]
        hyde_passages = [_sanitize_intermediate_text(raw, fallback=question_intermediate)]

    # Retrieve with each passage, pool results
    retrieval = _retrieve_and_format(row, hyde_passages, k=5, label_prefix="multi_hyde",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="multi_hyde/answer")

    return {
        "final_answer": answer,
        "hyde_passages": hyde_passages,
        "hyde_passages_raw": raw_hyde_passages,
        "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in raw_hyde_passages),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_snap_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-informed HyDE: LLM answers first, then generates targeted HyDE passage based on its reasoning."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_hyde/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage informed by the snap reasoning
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="snap_hyde/generate",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="snap_hyde",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Answer with evidence (direct, not arbitration — 70B does better without conservative bias)
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_hyde/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_snap_hyde_aligned(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-HyDE with question-aligned reranking.

    Dense retrieval uses the HyDE passage (testing embedding model's passage→passage ability),
    but cross-encoder reranks against the raw question (same as rag_simple).
    This isolates the embedding model's contribution from the reranking step.
    """
    question = _fmt(row, config)
    raw_question = str(row["question"])
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_hyde_aligned/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage informed by the snap reasoning
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="snap_hyde_aligned/generate",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve using HyDE for dense embedding, but rerank against raw question
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="snap_hyde_aligned",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config),
                                     rerank_query=raw_question)
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Answer with evidence
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_hyde_aligned/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def _gap_analysis(snap_answer: str, question: str) -> list[dict]:
    """Analyze gaps in the snap answer. Returns 0-3 structured gaps.

    Uses loose text format instead of JSON — more robust with small models.
    Each gap has: description (what's uncertain), sub_question (focused query).
    Returns empty list if model finds no gaps (high confidence).
    """
    system = (
        "You are a legal reasoning analyst. A student answered a legal question. "
        "Identify the single most important evidence gap that could change the answer.\n\n"
        "Use this format:\n"
        "- gap: <what specific rule, fact, or exception is uncertain> | ask: <focused sub-question>\n\n"
        "Rules:\n"
        "- Give exactly 1 gap — the one most likely to change the answer.\n"
        "- Only give 2 gaps if there are truly two independent uncertainties.\n"
        "- If the reasoning is solid and you are confident in the answer, reply exactly: NONE"
    )
    user = (
        f"## Student's Answer and Reasoning\n{snap_answer}\n\n"
        f"## Original Question\n{question}"
    )
    raw = _llm_call(system, user, label="gap/analyze")

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) == 1 and lines[0].upper().rstrip(".") == "NONE":
        return []

    gaps = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("-"):
            continue
        body = line.lstrip("- ").strip()
        # Case-insensitive parsing for model output variation
        body_lower = body.lower()
        if "| ask:" in body_lower:
            split_pos = body_lower.index("| ask:")
            desc = body[:split_pos].strip()
            subq = body[split_pos + 6:].strip()
        else:
            desc = body
            subq = ""
        # Strip "gap:" prefix if present (case-insensitive)
        if desc.lower().startswith("gap:"):
            desc = desc[4:].strip()
        subq = subq or desc
        if desc:
            gaps.append({"description": desc, "sub_question": subq})
    return gaps[:3]


GAP_MIN_CE = -100.0  # disabled — pass all evidence through (CE=1.0 was filtering 90%+ of passages)


def _gap_retrieve(gap: dict, question: str, row: pd.Series,
                  config: EvalConfig, gap_idx: int,
                  method: str = "hyde",
                  snap_answer: str = "") -> dict | None:
    """Run one gap investigation and return the gathered context.

    Supported methods:
      - 'hyde': generate a gap-focused hypothetical passage, then retrieve real passages
      - 'rag': retrieve directly from the gap sub-question
      - 'vectorless': generate a parametric legal note only (no corpus retrieval)
      - 'subagent_rag': retrieve passages, then summarize them into a short report
      - 'subagent_hybrid': retrieve passages, then synthesize a report with model knowledge
    """
    raw_question = str(row["question"])
    desc = gap.get("description", "")
    subq = gap.get("sub_question", desc)

    if not desc and not subq:
        return None  # malformed gap, skip

    if method == "vectorless":
        # LLM generates knowledge per gap — no vector store
        gen_user = (
            f"## Evidence Gap\n{desc}\n\n"
            f"## Sub-question\n{subq}\n\n"
            f"## Original Question\n{question}"
        )
        knowledge_raw = _llm_call(_VECTORLESS_DIRECT, gen_user, label=f"gap/vless_{gap_idx}")
        knowledge = _sanitize_intermediate_text(knowledge_raw, fallback=knowledge_raw)
        return {
            "gap": gap,
            "passages": [f"[Generated Note]\n{knowledge}"],
            "evidence_store": [{"idx": f"vless_{gap_idx}", "text": knowledge, "cross_encoder_score": 0}],
            "max_ce_score": 0,
            "report": knowledge,
            "report_raw": knowledge_raw,
            "report_contains_answer_artifact": _contains_answer_artifact(knowledge_raw),
        }

    if method == "subagent_rag":
        # Subagent: RAG retrieves → LLM reads and summarizes findings
        query = subq or desc
        retrieval = _retrieve_and_format(
            row, [query], k=5, label_prefix=f"sub_rag_{gap_idx}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
            rerank_query=raw_question,
        )
        passage_text = "\n\n".join(retrieval["passages"])
        # Subagent reads passages and writes a focused report
        report_user = (
            f"## Sub-question\n{subq}\n\n"
            f"## Retrieved Passages\n{passage_text}\n\n"
            f"## Original Question\n{question}"
        )
        report = _generate_report(
            _report_prompt(100),
            report_user,
            label=f"gap/sub_rag_{gap_idx}",
            fallback="Retrieved passages were not helpful.",
        )
        return {
            "gap": gap,
            "passages": retrieval["passages"],
            "evidence_store": retrieval["evidence_store"],
            "max_ce_score": retrieval["max_ce_score"],
            "report": report["text"],
            "report_raw": report["raw"],
            "report_contains_answer_artifact": report["contains_answer"],
        }

    if method == "subagent_hyde":
        # Subagent: HyDE retrieval (snap-informed) + LLM summarization
        gap_focus = []
        if desc:
            gap_focus.append(f"Evidence gap to verify: {desc}")
        if subq and subq != desc:
            gap_focus.append(f"Focused sub-question: {subq}")
        hyde = _generate_hyde(
            config,
            "snap_hyde",
            _snap_hyde_user(question, snap_answer, "\n".join(gap_focus)),
            label=f"gap/sub_hyde_{gap_idx}",
            fallback=subq or desc or question,
        )

        retrieval = _retrieve_and_format(
            row, [hyde["text"]], k=5, label_prefix=f"sub_hyde_{gap_idx}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
            rerank_query=raw_question,
        )
        passage_text = "\n\n".join(retrieval["passages"])
        report_user = (
            f"## Sub-question\n{subq}\n\n"
            f"## Retrieved Passages\n{passage_text}\n\n"
            f"## Original Question\n{question}"
        )
        report = _generate_report(
            _report_prompt(100),
            report_user,
            label=f"gap/sub_hyde_rpt_{gap_idx}",
            fallback="Retrieved passages were not helpful.",
        )
        return {
            "gap": gap,
            "passages": retrieval["passages"],
            "evidence_store": retrieval["evidence_store"],
            "max_ce_score": retrieval["max_ce_score"],
            "report": report["text"],
            "report_raw": report["raw"],
            "report_contains_answer_artifact": report["contains_answer"],
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
        }

    if method == "subagent_hybrid":
        # Subagent: RAG retrieves + LLM generates knowledge → combined report
        query = subq or desc
        retrieval = _retrieve_and_format(
            row, [query], k=3, label_prefix=f"sub_hyb_{gap_idx}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
            rerank_query=raw_question,
        )
        passage_text = "\n\n".join(retrieval["passages"])
        # Subagent synthesizes retrieved evidence + own knowledge
        report_user = (
            f"## Sub-question\n{subq}\n\n"
            f"## Retrieved Passages\n{passage_text}\n\n"
            f"## Original Question\n{question}"
        )
        report = _generate_report(
            _report_prompt(120, include_model_knowledge=True),
            report_user,
            label=f"gap/sub_hyb_{gap_idx}",
            fallback="Retrieved passages were not helpful.",
        )
        return {
            "gap": gap,
            "passages": retrieval["passages"],
            "evidence_store": retrieval["evidence_store"],
            "max_ce_score": retrieval["max_ce_score"],
            "report": report["text"],
            "report_raw": report["raw"],
            "report_contains_answer_artifact": report["contains_answer"],
        }

    if method == "hyde":
        gap_focus = []
        if desc:
            gap_focus.append(f"Evidence gap to verify: {desc}")
        if subq and subq != desc:
            gap_focus.append(f"Focused sub-question: {subq}")
        hyde = _generate_hyde(
            config,
            "snap_hyde",
            _snap_hyde_user(question, snap_answer, "\n".join(gap_focus)),
            label=f"gap/hyde_{gap_idx}",
            fallback=subq or desc or question,
        )
        query = hyde["text"]
    else:
        query = subq or desc

    retrieval = _retrieve_and_format(
        row, [query], k=5, label_prefix=f"gap_{method}_{gap_idx}",
        where=_where_from_config(config),
        collection=_collection_for_config(config),
        rerank_query=raw_question,
    )

    if retrieval["max_ce_score"] < GAP_MIN_CE:
        return None

    result = {
        "gap": gap,
        "passages": retrieval["passages"],
        "evidence_store": retrieval["evidence_store"],
        "max_ce_score": retrieval["max_ce_score"],
    }
    if method == "hyde":
        result["hyde_passage"] = hyde["text"]
        result["hyde_passage_raw"] = hyde["raw"]
        result["hyde_contains_answer_artifact"] = hyde["contains_answer"]
    return result


def _gap_final_answer(snap_answer: str, question: str, gaps: list[dict],
                      gap_results: list[dict | None], config: EvalConfig,
                      final_input: str = "full") -> str:
    """Assemble the final-answer prompt from the chosen gap artifacts.

    Supported final_input values:
      - 'full': snap answer + structured gap descriptions + evidence
      - 'evidence_only': flat evidence only
      - 'no_snap': structured gaps + evidence, but hide the snap answer
      - 'snap_and_evidence': snap answer + flat evidence, without gap structure
      - 'reports_nosnap': subagent/vectorless reports only
      - 'reports_and_evidence': reports plus the supporting raw passages
      - 'reports_and_snap': reports plus the snap answer (no raw passages)
      - 'reports_snap_evidence': reports + snap + raw passages (maximum info)
    """
    # Build evidence from gap results
    all_passages = []
    gap_sections = []
    for i, (gap, result) in enumerate(zip(gaps, gap_results), 1):
        desc = gap.get("description", f"Gap {i}")
        subq = gap.get("sub_question", "")
        if result is None:
            gap_sections.append(f"### Gap {i}: {desc}\nNo relevant evidence found.")
        else:
            passage_text = "\n\n".join(result["passages"])
            all_passages.extend(result["passages"])
            gap_sections.append(
                f"### Gap {i}: {desc}\n"
                f"Sub-question: {subq}\n"
                f"Retrieved evidence:\n{passage_text}"
            )

    # Build report sections (for subagent modes)
    report_sections = []
    for i, (gap, result) in enumerate(zip(gaps, gap_results), 1):
        desc = gap.get("description", f"Gap {i}")
        subq = gap.get("sub_question", "")
        report = result.get("report", "") if result else ""
        if result is None:
            report_sections.append(f"### Investigation {i}: {desc}\nNo findings.")
        elif report:
            report_sections.append(
                f"### Investigation {i}: {desc}\n"
                f"Sub-question: {subq}\n"
                f"Findings: {report}"
            )
        else:
            report_sections.append(
                f"### Investigation {i}: {desc}\n"
                f"Sub-question: {subq}\n"
                f"(No structured report available)"
            )

    gap_block = "\n\n".join(gap_sections) if gap_sections else "No evidence gaps identified."
    flat_passages = "\n\n".join(all_passages) if all_passages else "No evidence retrieved."
    report_block = "\n\n".join(report_sections) if report_sections else "No investigations completed."

    system = _system_prompt(config, "research" if final_input.startswith("reports") else "rag")

    if final_input == "evidence_only":
        user = f"## Retrieved Passages\n{flat_passages}\n\n## Question\n{question}"
    elif final_input == "no_snap":
        user = (
            f"## Evidence Gathered for Identified Gaps\n{gap_block}\n\n"
            f"## Question\n{question}"
        )
    elif final_input == "snap_and_evidence":
        user = (
            f"## Your Initial Answer\n{snap_answer}\n\n"
            f"## Retrieved Passages\n{flat_passages}\n\n"
            f"## Question\n{question}"
        )
    elif final_input == "reports_nosnap":
        user = (
            f"## Research Findings\n{report_block}\n\n"
            f"## Question\n{question}"
        )
    elif final_input == "reports_and_evidence":
        user = (
            f"## Research Findings\n{report_block}\n\n"
            f"## Supporting Passages\n{flat_passages}\n\n"
            f"## Question\n{question}"
        )
    elif final_input == "reports_and_snap":
        user = (
            f"## Your Initial Answer\n{snap_answer}\n\n"
            f"## Research Findings\n{report_block}\n\n"
            f"## Question\n{question}"
        )
    elif final_input == "reports_snap_evidence":
        user = (
            f"## Your Initial Answer\n{snap_answer}\n\n"
            f"## Research Findings\n{report_block}\n\n"
            f"## Supporting Passages\n{flat_passages}\n\n"
            f"## Question\n{question}"
        )
    else:  # full
        user = (
            f"## Your Initial Answer\n{snap_answer}\n\n"
            f"## Evidence Gathered for Identified Gaps\n{gap_block}\n\n"
            f"## Question\n{question}"
        )

    return _llm_call(system, user, label="gap/final_answer")


def _run_gap(row: pd.Series, config: EvalConfig,
             method: str = "hyde", label: str = "gap_hyde",
             final_input: str = "full") -> dict:
    """Unified gap-informed retrieval: snap → gap analysis → per-gap retrieval → final answer.

    Args:
        method: one of 'hyde', 'rag', 'vectorless', 'subagent_rag', 'subagent_hybrid'
        label: prefix for LLM call labels
        final_input: one of 'full', 'evidence_only', 'no_snap', 'snap_and_evidence',
            'reports_nosnap', or 'reports_and_evidence'
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label=f"{label}/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Gap analysis
    gaps = _gap_analysis(snap_answer, question_intermediate)

    # Step 3: No gaps → use snap directly (unless reports_nosnap, then fresh answer)
    if not gaps:
        if final_input in ("reports_nosnap", "reports_and_evidence", "no_snap", "evidence_only"):
            # Don't leak snap — make a fresh answer call
            fresh_answer = _llm_call(_system_prompt(config, "answer"), question, label=f"{label}/fresh")
            return {
                "final_answer": fresh_answer,
                "snap_answer": snap_answer,
                "snap_letter": snap_letter,
                "gaps": [],
                "gap_results": [],
                "evidence_store": [],
                "retrieved_ids": [],
                "gold_retrieved": False,
            }
        return {
            "final_answer": snap_answer,
            "snap_answer": snap_answer,
            "snap_letter": snap_letter,
            "gaps": [],
            "gap_results": [],
            "evidence_store": [],
            "retrieved_ids": [],
            "gold_retrieved": False,
        }

    # Step 4: Per-gap retrieval
    gap_results = []
    all_evidence = []
    all_ids = []
    for i, gap in enumerate(gaps):
        result = _gap_retrieve(gap, question_intermediate, row, config, i, method=method, snap_answer=snap_answer)
        gap_results.append(result)
        if result:
            all_evidence.extend(result["evidence_store"])
            all_ids.extend([ev["idx"] for ev in result["evidence_store"]])

    # Step 5: Final answer
    gold_idx = str(row.get("gold_idx", ""))
    answer = _gap_final_answer(snap_answer, question, gaps, gap_results, config,
                               final_input=final_input)

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "gaps": gaps,
        "gap_results": [
            {
                "gap": r["gap"],
                "max_ce": r.get("max_ce_score", 0),
                "report": r.get("report", ""),
                "report_raw": r.get("report_raw", ""),
                "report_contains_answer_artifact": r.get("report_contains_answer_artifact", False),
                "hyde_passage": r.get("hyde_passage", ""),
                "hyde_passage_raw": r.get("hyde_passage_raw", ""),
                "hyde_contains_answer_artifact": r.get("hyde_contains_answer_artifact", False),
            } if r else None
            for r in gap_results
        ],
        "evidence_store": all_evidence,
        "retrieved_ids": all_ids,
        "gold_retrieved": gold_idx in all_ids if gold_idx else False,
    }


def run_gap_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed HyDE: snap + gaps + evidence in final call (full context)."""
    return _run_gap(row, config, method="hyde", label="gap_hyde")


def run_gap_hyde_ev(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed HyDE: evidence only in final call (no snap, no gap structure)."""
    return _run_gap(row, config, method="hyde", label="gap_hyde_ev", final_input="evidence_only")


def run_gap_hyde_nosnap(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed HyDE: gaps + evidence but no snap answer in final call."""
    return _run_gap(row, config, method="hyde", label="gap_hyde_ns", final_input="no_snap")


def run_gap_hyde_flat(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed HyDE: snap + flat evidence (no gap structure) in final call."""
    return _run_gap(row, config, method="hyde", label="gap_hyde_flat", final_input="snap_and_evidence")


def run_gap_rag_nosnap(row: pd.Series, config: EvalConfig) -> dict:
    """Gap RAG without snap in final — tests anchoring hypothesis.

    Same retrieval as gap_rag but hides snap answer from the final call.
    If this beats gap_rag (63.5%), anchoring is confirmed as the bottleneck.
    """
    return _run_gap(row, config, method="rag", label="gap_rag_ns", final_input="no_snap")


def run_gap_vectorless(row: pd.Series, config: EvalConfig) -> dict:
    """Gap + vectorless: gap analysis → per-gap LLM knowledge → reports only (no snap).

    Combines gap targeting with vectorless knowledge generation.
    Final call sees subagent reports + question only (no snap, no retrieval).
    """
    return _run_gap(row, config, method="vectorless", label="gap_vless", final_input="reports_nosnap")


def run_subagent_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent HyDE: gap analysis → per-gap HyDE retrieval + LLM summarization → reports (no snap).

    Each subagent generates a HyDE passage for the gap, retrieves with it,
    reads the passages, and writes a focused report.
    Main agent sees subagent reports + question. No snap, no raw passages.
    """
    return _run_gap(row, config, method="subagent_hyde", label="sub_hyde", final_input="reports_nosnap")


def run_subagent_rag(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent RAG: gap analysis → per-gap RAG + LLM summarization → reports only (no snap).

    Each subagent retrieves passages, reads them, and writes a focused report.
    Main agent sees subagent reports + question. No snap, no raw passages.
    """
    return _run_gap(row, config, method="subagent_rag", label="sub_rag", final_input="reports_nosnap")


def run_subagent_hybrid(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent hybrid: gap analysis → per-gap RAG + LLM knowledge → synthesized reports (no snap).

    Each subagent retrieves passages AND generates own knowledge, then writes a combined report.
    Main agent sees subagent reports + question. No snap, no raw passages.
    """
    return _run_gap(row, config, method="subagent_hybrid", label="sub_hyb", final_input="reports_nosnap")


def run_subagent_rag_evidence(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent RAG with evidence: reports + raw passages (no snap).

    Same as subagent_rag but main agent also sees the raw passages alongside reports.
    Tests whether raw evidence adds value on top of subagent summaries.
    """
    return _run_gap(row, config, method="subagent_rag", label="sub_rag_ev", final_input="reports_and_evidence")


def run_subagent_rag_snap(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent RAG + snap visible: reports + snap answer in final call.

    Same retrieval/summarization as subagent_rag, but the final agent also sees
    the snap answer alongside the reports. Tests whether snap anchoring helps or
    hurts when mediated by independently-generated reports.
    """
    return _run_gap(row, config, method="subagent_rag", label="sub_rag_snap", final_input="reports_and_snap")


def run_subagent_rag_full(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent RAG maximum info: reports + snap + raw passages.

    Final agent sees everything: snap answer, subagent reports, AND raw passages.
    Tests whether maximum information helps or creates noise/anchoring.
    """
    return _run_gap(row, config, method="subagent_rag", label="sub_rag_full", final_input="reports_snap_evidence")


def run_snap_hyde_report(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-HyDE + summarization: snap_hyde retrieval pipeline with report denoising.

    Combines snap_hyde's proven retrieval (58.6% at N=1195) with subagent-style
    summarization. Unlike subagent modes, this skips gap analysis — retrieves with
    the full snap HyDE passage, summarizes ALL retrieved passages into a single
    report, then final agent sees report only (no snap, no raw passages).

    Tests whether noise filtering via summarization improves snap_hyde.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="shr/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage from snap (same as snap_hyde)
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="shr/hyde",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve using HyDE passage
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="shr",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Summarize retrieved passages into a focused report
    report_user = (
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Original Question\n{question_intermediate}"
    )
    report = _generate_report(
        _report_prompt(150),
        report_user,
        label="shr/report",
        fallback="Retrieved passages were not helpful.",
    )

    # Step 5: Final answer with report only (no snap, no raw passages)
    user = f"## Research Findings\n{report['text']}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "research"), user, label="shr/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "report": report["text"],
        "report_raw": report["raw"],
        "report_contains_answer_artifact": report["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_snap_hyde_report_snap(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-HyDE + summarization + snap visible: report + snap in final call.

    Same as snap_hyde_report but the final agent also sees the snap answer.
    Tests snap anchoring when combined with a summarized report.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="shrs/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: HyDE from snap
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="shrs/hyde",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="shrs",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Summarize
    report_user = (
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Original Question\n{question_intermediate}"
    )
    report = _generate_report(
        _report_prompt(150),
        report_user,
        label="shrs/report",
        fallback="Retrieved passages were not helpful.",
    )

    # Step 5: Final answer with report + snap
    user = (
        f"## Your Initial Answer\n{snap_answer}\n\n"
        f"## Research Findings\n{report['text']}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_system_prompt(config, "research"), user, label="shrs/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "report": report["text"],
        "report_raw": report["raw"],
        "report_contains_answer_artifact": report["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_gap_rag(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed RAG: snap + gaps + evidence in final call (full context)."""
    return _run_gap(row, config, method="rag", label="gap_rag")


def run_snap_rag(row: pd.Series, config: EvalConfig) -> dict:
    """Snap + simple RAG: answer first, then retrieve with raw question, re-answer with snap + evidence.

    Tests whether snap context improves a simple RAG answer without any gap analysis or HyDE.
    2 LLM calls: snap + final answer with evidence.
    """
    question = _fmt(row, config)
    raw_question = str(row["question"])

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_rag/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Retrieve with raw question, rerank against raw question (same as rag_simple)
    retrieval = _retrieve_and_format(row, [raw_question], k=5, label_prefix="snap_rag",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Final answer with snap context + evidence
    system = _system_prompt(config, "rag")
    user = (
        f"## Your Initial Answer\n{snap_answer}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(system, user, label="snap_rag/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_snap_rag_nosnap(row: pd.Series, config: EvalConfig) -> dict:
    """Snap + simple RAG but final call only sees evidence (no snap). Controls for whether snap helps final answer."""
    question = _fmt(row, config)
    raw_question = str(row["question"])

    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_rag_ns/snap")
    snap_letter = _extract_answer(snap_answer, config)

    retrieval = _retrieve_and_format(row, [raw_question], k=5, label_prefix="snap_rag_ns",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Final answer WITHOUT snap — just evidence + question (same as rag_simple but 2 calls)
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_rag_ns/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


# ---------------------------------------------------------------------------
# Historical "vectorless" family: LLM generates parametric knowledge instead of
# searching the corpus. Same 3-call skeleton as rag_snap_hyde.
# ---------------------------------------------------------------------------

_VECTORLESS_FINAL = (
    "You are a legal expert. Reason through the multiple-choice question independently first. "
    "Generated legal reference notes are provided — use them to verify, refine, or challenge "
    "your reasoning, but do not treat them as automatically correct. "
    "If a note is generic, circular, or contradicted by stronger reasoning, ignore it. "
    "Give your final answer as: Answer: (X)"
)

_VECTORLESS_DIRECT = (
    "You are a legal reference guide. A student answered a legal question. "
    "Write a short doctrinal note to help verify or correct their answer.\n\n"
    "Return ONLY these 4 bullets:\n"
    "- Governing rule:\n"
    "- Key exception or limitation:\n"
    "- Dispositive fact trigger:\n"
    "- What would make a different answer plausible:\n\n"
    "Rules: State black-letter law directly. No answer letters. "
    "No 'the correct answer is'. Keep under 120 words."
)

_VECTORLESS_ROLES = {
    "textbook": (
        "You are a legal textbook author. A student answered a legal question.\n"
        "Return ONLY 3 bullets:\n- Rule:\n- Exception/limitation:\n- Fact that controls:\n\n"
        "No answer letters. State the law directly. Keep under 90 words."
    ),
    "casebook": (
        "You are a casebook editor. A student answered a legal question.\n"
        "Return ONLY 3 bullets:\n- Holding-style rule:\n- Fact pattern that triggers it:\n"
        "- Common overread to avoid:\n\n"
        "No answer letters. Keep under 90 words."
    ),
    "barprep": (
        "You are a bar-prep tutor. A student answered a legal question.\n"
        "Return ONLY 3 bullets:\n- Rule:\n- Trap:\n- Decisive fact:\n\n"
        "No answer letters. Keep under 90 words."
    ),
}

_VECTORLESS_ELEMENTS = (
    "You are a legal issue spotter. A student answered a legal question.\n"
    "Identify the 2-4 dispositive legal elements and assess each.\n\n"
    "For each element, use this format:\n"
    "- [element name]: [rule] | fact=[fact signal] | pressure=[leans_correct/leans_wrong/ambiguous]\n\n"
    "No answer letters. Keep each element to one line."
)

_VECTORLESS_CHOICE_MAP = (
    "You are a bar exam differentiator. A student answered a legal question.\n"
    "Return ONLY 3 bullets:\n"
    "- Governing rule:\n"
    "- Strongest distractor pattern (the most plausible wrong answer and why):\n"
    "- Fact that flips the result:\n\n"
    "No answer letters. Focus on distinguishing the closest wrong answer. Keep under 90 words."
)


def _run_vectorless(row: pd.Series, config: EvalConfig,
                    gen_system: str, label: str = "vdirect",
                    include_snap: bool = False) -> dict:
    """Unified historical 'vectorless' flow: snap → generate parametric knowledge → final answer.

    Args:
        gen_system: system prompt for the knowledge generation step
        label: prefix for LLM call labels
        include_snap: if True, show snap answer in the final call alongside generated note
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label=f"{label}/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate knowledge from parametric memory
    gen_user = f"## Student's Initial Analysis\n{snap_answer}\n\n## Original Question\n{question_intermediate}"
    knowledge_raw = _llm_call(gen_system, gen_user, label=f"{label}/generate")
    knowledge = _sanitize_intermediate_text(knowledge_raw, fallback=knowledge_raw)

    # Step 3: Final answer with generated knowledge
    if include_snap:
        final_user = (
            f"## Your Initial Answer\n{snap_answer}\n\n"
            f"## Generated Legal Reference Note\n{knowledge}\n\n"
            f"## Question\n{question}"
        )
    else:
        final_user = (
            f"## Generated Legal Reference Note\n{knowledge}\n\n"
            f"## Question\n{question}"
        )
    answer = _llm_call(_VECTORLESS_FINAL, final_user, label=f"{label}/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "knowledge_note": knowledge,
        "knowledge_note_raw": knowledge_raw,
        "knowledge_contains_answer_artifact": _contains_answer_artifact(knowledge_raw),
        "evidence_store": [],
        "retrieved_ids": [],
        "gold_retrieved": False,
    }


def run_vectorless_direct(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' parametric reasoning: snap → doctrinal note → answer."""
    return _run_vectorless(row, config, _VECTORLESS_DIRECT, label="vdirect")


def run_vectorless_role(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' reasoning with role-conditioned generation. Use --tag textbook|casebook|barprep."""
    role = (config.tag.split("-")[-1] if config.tag else "barprep").strip().lower()
    system = _VECTORLESS_ROLES.get(role, _VECTORLESS_ROLES["barprep"])
    return _run_vectorless(row, config, system, label=f"vrole/{role}")


def run_vectorless_elements(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' reasoning: snap → identify dispositive legal elements → answer."""
    return _run_vectorless(row, config, _VECTORLESS_ELEMENTS, label="velem")


def run_vectorless_choice_map(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' reasoning: snap → map rule + distractor + decisive fact → answer."""
    return _run_vectorless(row, config, _VECTORLESS_CHOICE_MAP, label="vchoice")


def run_vectorless_nosnap(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' reasoning without snap: question → generate knowledge → answer.

    Control for snap ablation. Compares with vectorless_direct (3 calls, with snap)
    to measure the snap contribution to vectorless knowledge generation.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # No snap — generate knowledge directly from the question
    gen_user = f"## Legal Question\n{question_intermediate}"
    knowledge_raw = _llm_call(_VECTORLESS_DIRECT, gen_user, label="vnosnap/generate")
    knowledge = _sanitize_intermediate_text(knowledge_raw, fallback=knowledge_raw)

    # Answer with generated knowledge
    final_user = (
        f"## Generated Legal Reference Note\n{knowledge}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_VECTORLESS_FINAL, final_user, label="vnosnap/answer")

    return {
        "final_answer": answer,
        "snap_answer": "",
        "snap_letter": None,
        "knowledge_note": knowledge,
        "knowledge_note_raw": knowledge_raw,
        "knowledge_contains_answer_artifact": _contains_answer_artifact(knowledge_raw),
        "evidence_store": [],
        "retrieved_ids": [],
        "gold_retrieved": False,
    }


def run_vectorless_hybrid(row: pd.Series, config: EvalConfig) -> dict:
    """Hybrid: generated parametric knowledge + vector RAG evidence pooled together.

    Tests whether LLM-generated knowledge + retrieved passages > either alone.
    4 LLM calls: snap + generate knowledge + retrieve + answer with both.
    """
    question = _fmt(row, config)
    raw_question = str(row["question"])
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="vhybrid/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate knowledge (vectorless)
    gen_user = f"## Student's Initial Analysis\n{snap_answer}\n\n## Original Question\n{question_intermediate}"
    knowledge_raw = _llm_call(_VECTORLESS_DIRECT, gen_user, label="vhybrid/generate")
    knowledge = _sanitize_intermediate_text(knowledge_raw, fallback=knowledge_raw)

    # Step 3: Also retrieve via snap_hyde path (vector RAG)
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="vhybrid/hyde",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=3, label_prefix="vhybrid",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Answer with both generated knowledge and retrieved evidence
    final_user = (
        f"## Generated Legal Reference Note\n{knowledge}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_VECTORLESS_FINAL, final_user, label="vhybrid/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "knowledge_note": knowledge,
        "knowledge_note_raw": knowledge_raw,
        "knowledge_contains_answer_artifact": _contains_answer_artifact(knowledge_raw),
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_vectorless_keyword(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' keyword baseline: snap → generate search terms → retrieve → answer.

    This variant still searches the corpus. It asks the LLM for targeted keyword-style
    queries, retrieves with those queries, and reranks against the raw question.
    """
    from rag_utils import rerank_with_cross_encoder

    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    raw_question = str(row["question"])

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="vkeyword/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate targeted search terms
    keyword_system = (
        "You are a legal research assistant. Based on a student's answer to a legal question, "
        "generate 3-5 specific search keywords or phrases to find relevant legal authorities.\n\n"
        "Focus on: legal doctrine names, rule names, statute sections, case law concepts, "
        "and specific legal terms that would appear in a legal reference.\n\n"
        "Return one search phrase per line, nothing else."
    )
    keyword_user = f"## Student's Answer\n{snap_answer}\n\n## Question\n{question_intermediate}"
    keywords_raw = _llm_call(keyword_system, keyword_user, label="vkeyword/terms")

    # Parse keywords into search queries
    keywords = [
        _sanitize_intermediate_text(k.strip().lstrip("- •*0123456789."), fallback="")
        for k in keywords_raw.splitlines()
        if k.strip()
    ][:5]
    keywords = [k for k in keywords if k]

    # Step 3: Retrieve using each generated keyword, then rerank against the raw question.
    all_docs = []
    vs = get_vectorstore(_collection_for_config(config))
    for kw in keywords:
        if not kw:
            continue
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(kw)
        all_docs.extend(docs)

    # Dedup and rerank against raw question
    seen = set()
    unique_docs = []
    for doc in all_docs:
        idx = doc.metadata.get("idx", "")
        if idx not in seen:
            seen.add(idx)
            unique_docs.append(doc)

    reranked = rerank_with_cross_encoder(raw_question, unique_docs, top_k=5)

    passages = [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(reranked)]
    evidence_store = [{"idx": doc.metadata.get("idx", ""), "text": doc.page_content,
                       "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0)}
                      for doc in reranked]

    passage_block = "\n\n".join(passages)
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [e["idx"] for e in evidence_store]

    # Step 4: Answer with evidence
    final_user = (
        f"## Retrieved Legal Authorities\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_system_prompt(config, "rag"), final_user, label="vkeyword/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "keywords": keywords,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
    }


# ---------------------------------------------------------------------------
# Entity graph search: real corpus search without embeddings.
# Uses pre-built NLP entity graph + inverted index from build_entity_graph.py.
# Zero LLM calls for preprocessing. 1-2 LLM calls at query time.
# ---------------------------------------------------------------------------

_ENTITY_GRAPH = None  # lazy-loaded singleton
_CORPUS_DF = None  # lazy-loaded corpus for entity search


def _load_entity_graph():
    """Load the entity graph and inverted index. Cached after first load."""
    global _ENTITY_GRAPH
    if _ENTITY_GRAPH is not None:
        return _ENTITY_GRAPH

    import json
    graph_path = os.path.join("datasets", "barexam_qa", "entity_graph", "entity_graph.json")
    if not os.path.exists(graph_path):
        print(f"[entity_graph] WARNING: {graph_path} not found. Run utils/build_entity_graph.py first.")
        return None

    print(f"[entity_graph] Loading graph from {graph_path}...")
    with open(graph_path) as f:
        _ENTITY_GRAPH = json.load(f)
    print(f"[entity_graph] Loaded: {_ENTITY_GRAPH['n_entities']:,} entities, "
          f"{_ENTITY_GRAPH['n_edges']:,} edges, {_ENTITY_GRAPH.get('n_communities', 0)} communities")
    return _ENTITY_GRAPH


def _entity_search(question: str, graph: dict, corpus_df=None, top_k: int = 15) -> list:
    """Search corpus via entity graph inverted index.

    1. Extract entities from question using same regex/spaCy as build time
    2. Look up each entity in inverted index → candidate passage IDs
    3. Score candidates by number of matching entities
    4. Return top_k passage IDs sorted by match count
    """
    import re

    inverted_index = graph['inverted_index']

    # Extract entities from question (simplified — matches build_entity_graph regex patterns)
    q_lower = question.lower()
    q_entities = set()

    # Legal Latin, doctrine names, and common legal concepts
    LEGAL_TERMS = [
        'res ipsa loquitur', 'habeas corpus', 'mens rea', 'actus reus',
        'prima facie', 'bona fide', 'due process', 'equal protection',
        'strict liability', 'negligence per se', 'proximate cause',
        'consideration', 'promissory estoppel', 'specific performance',
        'hearsay', 'adverse possession', 'easement', 'eminent domain',
        'felony murder', 'manslaughter', 'larceny', 'robbery', 'burglary', 'arson',
        'negligence', 'duty of care', 'breach of duty', 'causation', 'damages',
        'offer', 'acceptance', 'mutual assent', 'statute of frauds',
        'parol evidence', 'third party beneficiary', 'assignment',
        'search and seizure', 'miranda', 'exclusionary rule', 'probable cause',
        'reasonable suspicion', 'warrant', 'plain view',
        'double jeopardy', 'self incrimination', 'right to counsel',
        'free speech', 'establishment clause', 'free exercise', 'commerce clause',
        'substantive due process', 'procedural due process', 'rational basis',
        'strict scrutiny', 'intermediate scrutiny',
        'best evidence rule', 'business records', 'dying declaration',
        'excited utterance', 'present sense impression', 'prior inconsistent',
        'character evidence', 'impeachment', 'expert testimony',
        'fee simple', 'life estate', 'future interest', 'remainder',
        'covenant', 'servitude', 'recording act', 'bona fide purchaser',
        'joint tenancy', 'tenancy in common', 'landlord tenant',
        'intentional tort', 'battery', 'assault', 'false imprisonment',
        'trespass', 'conversion', 'defamation', 'libel', 'slander',
        'comparative negligence', 'contributory negligence', 'assumption of risk',
        'vicarious liability', 'respondeat superior', 'joint and several',
        'products liability', 'warranty', 'merchantability',
    ]
    for term in LEGAL_TERMS:
        if term in q_lower:
            q_entities.add(term)

    # Statute references
    for m in re.finditer(r'\b((?:section|rule|article|amendment|clause)\s+\w+)\b', q_lower):
        q_entities.add(m.group(1))

    # Quoted terms
    for m in re.finditer(r'"([^"]{3,50})"', question):
        phrase = m.group(1).lower().strip()
        if phrase in inverted_index:
            q_entities.add(phrase)

    # Multi-word capitalized phrases
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question):
        phrase = m.group(1).lower()
        if phrase in inverted_index:
            q_entities.add(phrase)

    # Try 2-word and 3-word sliding window phrases against the index
    words = q_lower.split()
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if phrase in inverted_index:
                q_entities.add(phrase)

    # Individual words (6+ chars) against the index
    for word in words:
        if len(word) > 5 and word in inverted_index:
            q_entities.add(word)

    if not q_entities:
        return []

    # Look up candidates
    from collections import Counter
    candidate_scores = Counter()
    for entity in q_entities:
        if entity in inverted_index:
            for pid in inverted_index[entity]:
                candidate_scores[pid] += 1

    # Return top_k by match count
    return [pid for pid, score in candidate_scores.most_common(top_k)]


def run_entity_search(row: pd.Series, config: EvalConfig) -> dict:
    """Entity graph search: real corpus search without embeddings.

    Uses pre-built NLP inverted index to find passages containing
    entities from the question. Cross-encoder reranks. 1 LLM call.
    Zero LLM preprocessing. Zero embeddings.
    """
    from rag_utils import rerank_with_cross_encoder
    import pandas as _pd

    question = _fmt(row, config)
    raw_question = str(row["question"])

    graph = _load_entity_graph()
    if graph is None:
        # Fallback to llm_only
        answer = _llm_call(_system_prompt(config, "answer"), question, label="entity/answer")
        return {"final_answer": answer, "snap_answer": "", "snap_letter": None,
                "evidence_store": [], "retrieved_ids": [], "gold_retrieved": False}

    # Search via entity inverted index
    candidate_ids = _entity_search(raw_question, graph, top_k=30)

    if not candidate_ids:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="entity/answer")
        return {"final_answer": answer, "snap_answer": "", "snap_letter": None,
                "evidence_store": [], "retrieved_ids": [], "gold_retrieved": False}

    # Load candidate passages from cached corpus
    global _CORPUS_DF
    if _CORPUS_DF is None:
        print("[entity_graph] Loading corpus CSV (one-time)...")
        _CORPUS_DF = _pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", usecols=['idx', 'text'])
        _CORPUS_DF['idx'] = _CORPUS_DF['idx'].astype(str)
    candidates = _CORPUS_DF[_CORPUS_DF['idx'].isin(set(candidate_ids))]

    if candidates.empty:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="entity/answer")
        return {"final_answer": answer, "snap_answer": "", "snap_letter": None,
                "evidence_store": [], "retrieved_ids": [], "gold_retrieved": False}

    # Build Document objects for cross-encoder reranking
    from langchain_core.documents import Document
    docs = []
    for _, r in candidates.iterrows():
        doc = Document(page_content=str(r['text']), metadata={"idx": str(r['idx'])})
        docs.append(doc)

    reranked = rerank_with_cross_encoder(raw_question, docs, top_k=5)

    passages = [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(reranked)]
    evidence_store = [{"idx": doc.metadata.get("idx", ""), "text": doc.page_content,
                       "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0)}
                      for doc in reranked]
    passage_block = "\n\n".join(passages)
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [e["idx"] for e in evidence_store]

    # Single LLM call — answer with retrieved passages
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="entity/answer")

    return {
        "final_answer": answer,
        "snap_answer": "",
        "snap_letter": None,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
    }


def run_snap_entity_search(row: pd.Series, config: EvalConfig) -> dict:
    """Snap + entity graph search: snap first, then entity search, answer fresh.

    2 LLM calls. Snap steers nothing (entity search uses the raw question),
    but the snap reasoning may still help the final answer indirectly.
    Tests snap contribution to entity-based retrieval.
    """
    from rag_utils import rerank_with_cross_encoder

    question = _fmt(row, config)
    raw_question = str(row["question"])

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_entity/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Entity search (same as run_entity_search but we have snap)
    graph = _load_entity_graph()
    if graph is None:
        # Fallback: answer fresh without snap (don't leak snap)
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_entity/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    candidate_ids = _entity_search(raw_question, graph, top_k=30)

    if not candidate_ids:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_entity/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    import pandas as _pd
    from langchain_core.documents import Document
    global _CORPUS_DF
    if _CORPUS_DF is None:
        print("[entity_graph] Loading corpus CSV (one-time)...")
        _CORPUS_DF = _pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", usecols=['idx', 'text'])
        _CORPUS_DF['idx'] = _CORPUS_DF['idx'].astype(str)
    candidates = _CORPUS_DF[_CORPUS_DF['idx'].isin(set(candidate_ids))]

    if candidates.empty:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_entity/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    docs = [Document(page_content=str(r['text']), metadata={"idx": str(r['idx'])})
            for _, r in candidates.iterrows()]
    reranked = rerank_with_cross_encoder(raw_question, docs, top_k=5)

    passages = [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(reranked)]
    evidence_store = [{"idx": doc.metadata.get("idx", ""), "text": doc.page_content,
                       "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0)}
                      for doc in reranked]
    passage_block = "\n\n".join(passages)
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [e["idx"] for e in evidence_store]

    # Answer fresh — no snap shown (avoids anchoring)
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_entity/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
    }


def run_snap_entity_informed(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-informed entity search: extract entities from snap reasoning + question.

    The snap surfaces legal terms not in the question (e.g., question says
    "garage that encroached" but snap reasons about "adverse possession").
    Entity search equivalent of HyDE — snap reasoning generates better search terms.
    2 LLM calls. Snap hidden from final answer.
    """
    from rag_utils import rerank_with_cross_encoder
    import pandas as _pd
    from langchain_core.documents import Document

    question = _fmt(row, config)
    raw_question = str(row["question"])

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_ent_inf/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Entity search using BOTH question AND snap reasoning
    graph = _load_entity_graph()
    if graph is None:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_ent_inf/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    # Extract entities from combined text — snap reasoning surfaces legal terms
    combined_text = f"{raw_question}\n\n{snap_answer}"
    candidate_ids = _entity_search(combined_text, graph, top_k=30)

    if not candidate_ids:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_ent_inf/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    # Load passages
    global _CORPUS_DF
    if _CORPUS_DF is None:
        print("[entity_graph] Loading corpus CSV (one-time)...")
        _CORPUS_DF = _pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", usecols=['idx', 'text'])
        _CORPUS_DF['idx'] = _CORPUS_DF['idx'].astype(str)
    candidates = _CORPUS_DF[_CORPUS_DF['idx'].isin(set(candidate_ids))]

    if candidates.empty:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_ent_inf/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    docs = [Document(page_content=str(r['text']), metadata={"idx": str(r['idx'])})
            for _, r in candidates.iterrows()]
    reranked = rerank_with_cross_encoder(raw_question, docs, top_k=5)

    passages = [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(reranked)]
    evidence_store = [{"idx": doc.metadata.get("idx", ""), "text": doc.page_content,
                       "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0)}
                      for doc in reranked]
    passage_block = "\n\n".join(passages)
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [e["idx"] for e in evidence_store]

    # Answer fresh — no snap shown
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_ent_inf/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
    }


def run_ce_threshold(row: pd.Series, config: EvalConfig) -> dict:
    """Score-thresholded Snap-HyDE: if best CE score < threshold, discard evidence and use snap answer."""
    CE_THRESHOLD = 4.0  # calibrated from N=200 BarExam analysis: snap=78% below, RAG=78% above
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="ce_thresh/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="ce_thresh/generate",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="ce_thresh",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))

    # Step 4: Check CE threshold — if best passage is below threshold, use snap answer directly
    max_ce = retrieval["max_ce_score"]
    if max_ce < CE_THRESHOLD:
        return {
            "final_answer": snap_answer,
            "snap_answer": snap_answer,
            "snap_letter": snap_letter,
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "evidence_store": retrieval["evidence_store"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "gold_retrieved": retrieval["gold_retrieved"],
            "routed_to": "snap_only",
            "max_ce_score": max_ce,
            "ce_threshold": CE_THRESHOLD,
        }

    # Step 5: Answer with evidence (above threshold — evidence is likely relevant)
    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="ce_thresh/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "routed_to": "rag",
        "max_ce_score": max_ce,
        "ce_threshold": CE_THRESHOLD,
    }


def run_snap_hyde_aspect(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-HyDE + aspect queries: HyDE passage + rule/exception aspect queries for broader retrieval."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="aspect/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="aspect/hyde",
        fallback=question_intermediate,
    )

    # Step 3: Generate aspect queries (rule + exception) based on the snap reasoning
    aspect_prompt = (
        f"Based on this legal question and analysis, generate two short search queries "
        f"targeting different legal dimensions. Return ONLY a JSON object.\n\n"
        f"Question: {question_intermediate}\n\n"
        f"Analysis: {snap_answer}\n\n"
        f'Return: {{"rule": "query targeting the governing rule, statute, or doctrine", '
        f'"exception": "query targeting exceptions, defenses, or limitations"}}'
    )
    aspect_raw = _llm_call("You are a legal search query generator. Return ONLY valid JSON.",
                            aspect_prompt, label="aspect/queries")
    aspect_parsed = _parse_json(aspect_raw)

    # Build query list: HyDE passage (primary for reranking) + aspect queries
    queries = [hyde["text"]]
    if aspect_parsed:
        if "rule" in aspect_parsed:
            rule_query = _sanitize_intermediate_text(str(aspect_parsed["rule"]), fallback="")
            if rule_query:
                queries.append(rule_query)
                aspect_parsed["rule"] = rule_query
        if "exception" in aspect_parsed:
            exception_query = _sanitize_intermediate_text(str(aspect_parsed["exception"]), fallback="")
            if exception_query:
                queries.append(exception_query)
                aspect_parsed["exception"] = exception_query

    # Step 4: Multi-query retrieval (pools candidates from all queries, reranks against primary)
    retrieval = _retrieve_and_format(row, queries, k=5, label_prefix="aspect",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 5: Answer with evidence
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="aspect/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "aspect_queries": aspect_parsed,
        "num_queries": len(queries),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_ce_threshold_k3(row: pd.Series, config: EvalConfig) -> dict:
    """CE-thresholded Snap-HyDE with k=3 instead of k=5. Tests whether fewer, higher-quality passages help."""
    CE_THRESHOLD = 4.0
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="ce_k3/snap")
    snap_letter = _extract_answer(snap_answer, config)

    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="ce_k3/generate",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=3, label_prefix="ce_k3",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))

    max_ce = retrieval["max_ce_score"]
    if max_ce < CE_THRESHOLD:
        return {
            "final_answer": snap_answer,
            "snap_answer": snap_answer,
            "snap_letter": snap_letter,
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "evidence_store": retrieval["evidence_store"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "gold_retrieved": retrieval["gold_retrieved"],
            "routed_to": "snap_only",
            "max_ce_score": max_ce,
            "k": 3,
        }

    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="ce_k3/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "routed_to": "rag",
        "max_ce_score": max_ce,
        "k": 3,
    }


def run_rag_devil_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Devil's advocate HyDE: retrieve for snap answer AND for the opposing answer, present both."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="devil_hyde/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate supporting HyDE passage (same as snap_hyde)
    hyde_user = _snap_hyde_user(question_intermediate, snap_answer)
    support = _generate_hyde(
        config,
        "snap_hyde",
        hyde_user,
        label="devil_hyde/support",
        fallback=question_intermediate,
    )

    # Step 3: Generate devil's advocate HyDE passage (opposing the snap answer)
    devil = _generate_hyde(
        config,
        "devil_hyde",
        hyde_user,
        label="devil_hyde/oppose",
        fallback=question_intermediate,
    )

    # Step 4: Retrieve with BOTH passages pooled
    collection = _collection_for_config(config)
    retrieval = _retrieve_and_format(row, [support["text"], devil["text"]], k=5,
                                     label_prefix="devil_hyde",
                                     where=_where_from_config(config),
                                     collection=collection)
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 5: Answer with evidence (direct — let model weigh both sides)
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="devil_hyde/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "support_passage": support["text"],
        "support_passage_raw": support["raw"],
        "support_contains_answer_artifact": support["contains_answer"],
        "devil_passage": devil["text"],
        "devil_passage_raw": devil["raw"],
        "devil_contains_answer_artifact": devil["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_top2_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Top-2 HyDE: snap answer identifies top 2 choices, generate HyDE for each, pool retrieval."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer — ask for top 2 choices with reasoning
    top2_system = _system_prompt(config, "top2_snap")
    snap_answer = _llm_call(top2_system, question, label="top2_hyde/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE for primary answer
    hyde_user_1 = _snap_hyde_user(question_intermediate, snap_answer)
    hyde_1 = _generate_hyde(
        config,
        "snap_hyde",
        hyde_user_1,
        label="top2_hyde/primary",
        fallback=question_intermediate,
    )

    # Step 3: Generate HyDE for second-choice answer
    hyde_2 = _generate_hyde(
        config,
        "top2_hyde",
        hyde_user_1,
        label="top2_hyde/secondary",
        fallback=question_intermediate,
    )

    # Step 4: Retrieve with both HyDE passages
    collection = _collection_for_config(config)
    retrieval = _retrieve_and_format(row, [hyde_1["text"], hyde_2["text"]], k=5,
                                     label_prefix="top2_hyde",
                                     where=_where_from_config(config),
                                     collection=collection)
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 5: Answer with evidence
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="top2_hyde/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_primary": hyde_1["text"],
        "hyde_primary_raw": hyde_1["raw"],
        "hyde_primary_contains_answer_artifact": hyde_1["contains_answer"],
        "hyde_secondary": hyde_2["text"],
        "hyde_secondary_raw": hyde_2["raw"],
        "hyde_secondary_contains_answer_artifact": hyde_2["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_hyde_arb(row: pd.Series, config: EvalConfig) -> dict:
    """HyDE retrieval + conservative arbitration: snap → HyDE retrieve → review."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="hyde_arb/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: HyDE retrieval
    hyde = _generate_hyde(
        config,
        "hyde",
        _question_only_hyde_user(question_intermediate),
        label="hyde_arb/generate",
        fallback=question_intermediate,
    )
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="hyde_arb",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Arbitrate
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given retrieved legal passages that may be relevant. "
        "Review the passages carefully. If the evidence supports your original answer, keep it. "
        "If the evidence clearly points to a different answer, change it. "
        "Do not change your answer unless the evidence gives you a strong reason to. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    arb_user = (
        f"## Your Previous Answer\n{snap_answer}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    final_answer = _llm_call(arb_system, arb_user, label="hyde_arb/arbitrate")
    final_letter = _extract_answer(final_answer, config)

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "final_letter": final_letter,
        "changed": snap_letter != final_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


_RAG_SYSTEM = (
    "You are a legal expert. Reason through the multiple-choice question "
    "step by step. Retrieved passages are provided — use them to verify or "
    "refine your reasoning, but think through the problem independently first. "
    "Give your final answer as: Answer: (X)"
)


def _where_from_config(config: EvalConfig) -> dict | None:
    """Build ChromaDB where filter from config.source_filter."""
    if config.source_filter:
        return {"source": config.source_filter}
    return None


def run_rag_rewrite(row: pd.Series, config: EvalConfig) -> dict:
    """Query rewrite → retrieval → answer with evidence."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    queries = _rewrite_query(question_intermediate)

    retrieval = _retrieve_and_format(row, queries, k=5, label_prefix="rewrite",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="rag_rewrite/answer")

    return {
        "final_answer": answer,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "rewrite_queries": queries,
    }


def run_rag_simple(row: pd.Series, config: EvalConfig) -> dict:
    """Raw question → retrieval → answer with evidence (no rewrite)."""
    question = _fmt(row, config)
    raw_question = str(row["question"])

    retrieval = _retrieve_and_format(row, [raw_question], k=5, label_prefix="simple",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="rag_simple/answer")

    return {
        "final_answer": answer,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_arbitration(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then reviews retrieved passages (conservative framing)."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    queries = _rewrite_query(question_intermediate, label="rag_arb/rewrite")

    # Step 1: Snap answer (no evidence)
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="rag_arb/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Retrieve
    retrieval = _retrieve_and_format(row, queries, k=5, label_prefix="rag_arb",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Arbitrate with conservative framing
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given retrieved legal passages that may be relevant. "
        "Review the passages carefully. If the evidence supports your original answer, keep it. "
        "If the evidence clearly points to a different answer, change it. "
        "Do not change your answer unless the evidence gives you a strong reason to. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    arb_user = (
        f"## Your Previous Answer\n{snap_answer}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    final_answer = _llm_call(arb_system, arb_user, label="rag_arb/arbitrate")
    final_letter = _extract_answer(final_answer, config)

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "final_letter": final_letter,
        "changed": snap_letter != final_letter,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "rewrite_queries": queries,
        "max_ce_score": retrieval["max_ce_score"],
    }


def run_decompose(row: pd.Series, config: EvalConfig) -> dict:
    """Decompose-then-answer: break question into sub-questions, answer each, synthesize."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Determine which decomposition variant to use (controlled by tag)
    variant = "structured" if "structured" in (config.tag or "") else "natural"

    if variant == "structured":
        # Variant A: IRAC-structured decomposition
        decompose_system = (
            "You are a legal analyst. Given a legal question, identify the 2-3 key sub-issues "
            "that must be resolved to answer it. Structure them as:\n"
            "1. RULE: What is the governing legal rule or doctrine?\n"
            "2. APPLICATION: How do the specific facts interact with the rule?\n"
            "3. EXCEPTION: Are there any exceptions, defenses, or limitations that apply?\n\n"
            "Output ONLY a JSON list of sub-questions, e.g.:\n"
            '[\"What is the rule for...\", \"How do the facts...\", \"Are there exceptions...\"]'
        )
    else:
        # Variant B: Natural decomposition — let model decide what matters
        decompose_system = (
            "You are a legal analyst. Given a legal question, identify the 2-3 key issues "
            "you need to resolve to answer it correctly. Think about what makes this question "
            "hard and what you'd need to figure out.\n\n"
            "Output ONLY a JSON list of sub-questions, e.g.:\n"
            '[\"Does X apply here?\", \"What is the standard for...\", \"Is there an exception when...\"]'
        )

    # Step 1: Decompose
    raw_decomp = _llm_call(decompose_system, question_intermediate, label="decompose/split")
    sub_questions = _parse_json(raw_decomp)
    if not isinstance(sub_questions, list) or not sub_questions:
        # Fallback: if decomposition fails, just answer directly
        sub_questions = [question_intermediate]

    # Cap at 3 sub-questions
    sub_questions = sub_questions[:3]

    # Step 2: Answer each sub-question independently
    sub_answers = []
    answer_system = _system_prompt(config, "answer")
    for i, sq in enumerate(sub_questions):
        # Give the sub-question in context of the original
        sub_prompt = f"In the context of this question:\n{question_intermediate}\n\nAddress this specific issue:\n{sq}"
        sub_ans = _llm_call(answer_system, sub_prompt, label=f"decompose/sub_{i}")
        sub_answers.append({"question": sq, "answer": sub_ans})

    # Step 3: Synthesize sub-answers into final answer
    synth_parts = []
    for sa in sub_answers:
        synth_parts.append(f"Issue: {sa['question']}\nAnalysis: {sa['answer']}")
    synth_block = "\n\n".join(synth_parts)

    synth_system = _system_prompt(config, "answer")
    synth_user = (
        f"You previously analyzed a legal question by breaking it into sub-issues. "
        f"Now synthesize your analysis into a final answer.\n\n"
        f"## Sub-Issue Analyses\n{synth_block}\n\n"
        f"## Original Question\n{question}"
    )
    final_answer = _llm_call(synth_system, synth_user, label="decompose/synthesize")

    return {
        "final_answer": final_answer,
        "variant": variant,
        "sub_questions": sub_questions,
        "sub_answers": [sa["answer"] for sa in sub_answers],
        "num_sub_questions": len(sub_questions),
    }


def run_decompose_rag(row: pd.Series, config: EvalConfig) -> dict:
    """Decompose + Snap-HyDE RAG: break into sub-questions, RAG each, synthesize with evidence."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Determine decomposition variant (controlled by tag)
    variant = "structured" if "structured" in (config.tag or "") else "natural"

    if variant == "structured":
        decompose_system = (
            "You are a legal analyst. Given a legal question, identify the 2-3 key sub-issues "
            "that must be resolved to answer it. Structure them as:\n"
            "1. RULE: What is the governing legal rule or doctrine?\n"
            "2. APPLICATION: How do the specific facts interact with the rule?\n"
            "3. EXCEPTION: Are there any exceptions, defenses, or limitations that apply?\n\n"
            "Output ONLY a JSON list of sub-questions, e.g.:\n"
            '[\"What is the rule for...\", \"How do the facts...\", \"Are there exceptions...\"]'
        )
    else:
        decompose_system = (
            "You are a legal analyst. Given a legal question, identify the 2-3 key issues "
            "you need to resolve to answer it correctly. Think about what makes this question "
            "hard and what you'd need to figure out.\n\n"
            "Output ONLY a JSON list of sub-questions, e.g.:\n"
            '[\"Does X apply here?\", \"What is the standard for...\", \"Is there an exception when...\"]'
        )

    # Step 1: Decompose
    raw_decomp = _llm_call(decompose_system, question_intermediate, label="decomp_rag/split")
    sub_questions = _parse_json(raw_decomp)
    if not isinstance(sub_questions, list) or not sub_questions:
        sub_questions = [question_intermediate]
    sub_questions = sub_questions[:3]

    # Step 2: For each sub-question — snap answer → HyDE → retrieve
    sub_results = []
    all_evidence = []
    all_retrieved_ids = []
    any_gold = False

    for i, sq in enumerate(sub_questions):
        sub_prompt = f"In the context of this question:\n{question_intermediate}\n\nAddress this specific issue:\n{sq}"

        # Snap answer this sub-question
        sub_snap = _llm_call(_system_prompt(config, "answer"), sub_prompt, label=f"decomp_rag/snap_{i}")

        # Generate HyDE passage from the sub-answer
        hyde = _generate_hyde(
            config,
            "snap_hyde",
            _snap_hyde_user(sq, sub_snap),
            label=f"decomp_rag/hyde_{i}",
            fallback=sq,
        )

        # Retrieve evidence for this sub-question
        retrieval = _retrieve_and_format(row, [hyde["text"]], k=3, label_prefix=f"decomp_rag_{i}",
                                         where=_where_from_config(config),
                                         collection=_collection_for_config(config))

        sub_results.append({
            "sub_question": sq,
            "snap_answer": sub_snap,
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "passages": retrieval["passages"],
        })
        all_evidence.extend(retrieval["evidence_store"])
        all_retrieved_ids.extend(retrieval["retrieved_ids"])
        if retrieval["gold_retrieved"]:
            any_gold = True

    # Step 3: Synthesize all sub-answers + evidence into final answer
    synth_parts = []
    for sr in sub_results:
        evidence_block = "\n".join(sr["passages"]) if sr["passages"] else "(no evidence retrieved)"
        synth_parts.append(
            f"Issue: {sr['sub_question']}\n"
            f"Analysis: {sr['snap_answer']}\n"
            f"Supporting Evidence:\n{evidence_block}"
        )
    synth_block = "\n\n---\n\n".join(synth_parts)

    synth_system = _system_prompt(config, "rag")
    synth_user = (
        f"You previously analyzed a legal question by breaking it into sub-issues. "
        f"Each sub-issue has been analyzed and supporting evidence has been retrieved. "
        f"Now synthesize everything into a final answer.\n\n"
        f"## Sub-Issue Analyses with Evidence\n{synth_block}\n\n"
        f"## Original Question\n{question}"
    )
    final_answer = _llm_call(synth_system, synth_user, label="decomp_rag/synthesize")

    return {
        "final_answer": final_answer,
        "variant": variant,
        "sub_questions": sub_questions,
        "sub_answers": [sr["snap_answer"] for sr in sub_results],
        "hyde_passages": [sr["hyde_passage"] for sr in sub_results],
        "hyde_passages_raw": [sr["hyde_passage_raw"] for sr in sub_results],
        "hyde_contains_answer_artifact": any(sr["hyde_contains_answer_artifact"] for sr in sub_results),
        "num_sub_questions": len(sub_questions),
        "evidence_store": all_evidence,
        "retrieved_ids": all_retrieved_ids,
        "gold_retrieved": any_gold,
    }


def run_conf_ce_threshold(row: pd.Series, config: EvalConfig) -> dict:
    """Combined: confidence gating (3 snap votes) + CE threshold on the RAG path."""
    CE_THRESHOLD = 4.0
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Take 3 snap answers
    snaps = []
    snap_letters = []
    for k in range(3):
        answer = _llm_call(_system_prompt(config, "answer"), question, label=f"conf_ce/snap_{k}")
        letter = _extract_answer(answer, config)
        snaps.append(answer)
        snap_letters.append(letter)

    # Step 2: Check consensus
    from collections import Counter
    vote_counts = Counter(snap_letters)
    majority_answer, majority_count = vote_counts.most_common(1)[0]
    unanimous = majority_count == 3
    majority_idx = snap_letters.index(majority_answer)

    if unanimous:
        return {
            "final_answer": snaps[majority_idx],
            "snap_answers": snaps,
            "snap_letters": snap_letters,
            "routed_to": "skip_rag",
            "consensus": "unanimous",
            "majority_answer": majority_answer,
        }

    # Step 3: Low confidence — Snap-HyDE with CE threshold
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snaps[majority_idx]),
        label="conf_ce/hyde",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="conf_ce",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))

    # Step 4: CE threshold — if evidence is low quality, use majority snap answer
    max_ce = retrieval["max_ce_score"]
    if max_ce < CE_THRESHOLD:
        return {
            "final_answer": snaps[majority_idx],
            "snap_answers": snaps,
            "snap_letters": snap_letters,
            "routed_to": "snap_ce_fallback",
            "consensus": f"{majority_count}/3",
            "majority_answer": majority_answer,
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "max_ce_score": max_ce,
            "ce_threshold": CE_THRESHOLD,
            "evidence_store": retrieval["evidence_store"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "gold_retrieved": retrieval["gold_retrieved"],
        }

    # Step 5: Good evidence — answer with RAG
    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="conf_ce/answer")

    return {
        "final_answer": answer,
        "snap_answers": snaps,
        "snap_letters": snap_letters,
        "routed_to": "rag",
        "consensus": f"{majority_count}/3",
        "majority_answer": majority_answer,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "max_ce_score": max_ce,
        "ce_threshold": CE_THRESHOLD,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_confidence_gated(row: pd.Series, config: EvalConfig) -> dict:
    """Confidence-gated RAG: 3 snap answers vote; unanimous = skip RAG, disagreement = Snap-HyDE."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Take 3 snap answers
    snaps = []
    snap_letters = []
    for k in range(3):
        answer = _llm_call(_system_prompt(config, "answer"), question, label=f"conf_gate/snap_{k}")
        letter = _extract_answer(answer, config)
        snaps.append(answer)
        snap_letters.append(letter)

    # Step 2: Check consensus
    from collections import Counter
    vote_counts = Counter(snap_letters)
    majority_answer, majority_count = vote_counts.most_common(1)[0]
    unanimous = majority_count == 3
    majority_idx = snap_letters.index(majority_answer)  # use this snap's reasoning

    if unanimous:
        # High confidence — skip RAG, return majority snap answer
        return {
            "final_answer": snaps[majority_idx],
            "snap_answers": snaps,
            "snap_letters": snap_letters,
            "routed_to": "skip_rag",
            "consensus": "unanimous",
            "majority_answer": majority_answer,
        }

    # Step 3: Low confidence — apply Snap-HyDE using majority snap's reasoning
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snaps[majority_idx]),
        label="conf_gate/hyde",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="conf_gate",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="conf_gate/answer")

    return {
        "final_answer": answer,
        "snap_answers": snaps,
        "snap_letters": snap_letters,
        "routed_to": "snap_hyde",
        "consensus": f"{majority_count}/3",
        "majority_answer": majority_answer,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


# ---------------------------------------------------------------------------
# Atomic Blocks — testing individual reasoning strategies
# ---------------------------------------------------------------------------

def run_self_verify(row: pd.Series, config: EvalConfig) -> dict:
    """Self-verification: snap answer, then ask the model to review and correct it.
    Tests whether a second pass catches errors. 2 LLM calls, no retrieval."""
    question = _fmt(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="verify/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Self-review
    review_prompt = (
        f"You previously answered a legal question. Review your answer carefully for errors "
        f"in legal reasoning, missed elements, or incorrect conclusions. If you find an error, "
        f"provide the corrected answer. If your answer is correct, restate it.\n\n"
        f"## Your Previous Answer\n{snap_answer}\n\n"
        f"## Original Question\n{question}"
    )
    verified = _llm_call(
        "You are a careful legal reviewer. Check the answer for errors and correct if needed.",
        review_prompt, label="verify/review"
    )

    return {
        "final_answer": verified,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
    }


def run_double_snap(row: pd.Series, config: EvalConfig) -> dict:
    """Double-snap: two independent answers. If same → use it. If different → CE-threshold RAG.
    Tests the cheapest confidence signal (2 calls when confident). 2-4 LLM calls."""
    CE_THRESHOLD = 4.0
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Two independent snap answers
    snap1 = _llm_call(_system_prompt(config, "answer"), question, label="dsnap/snap1")
    letter1 = _extract_answer(snap1, config)
    snap2 = _llm_call(_system_prompt(config, "answer"), question, label="dsnap/snap2")
    letter2 = _extract_answer(snap2, config)

    if letter1 == letter2:
        # Agreement — high confidence, skip RAG
        return {
            "final_answer": snap1,
            "snap1": snap1, "snap2": snap2,
            "letter1": letter1, "letter2": letter2,
            "routed_to": "snap_agree",
        }

    # Step 2: Disagreement — CE-threshold RAG using snap1's reasoning
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap1),
        label="dsnap/hyde",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=5, label_prefix="dsnap",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))

    max_ce = retrieval["max_ce_score"]
    if max_ce < CE_THRESHOLD:
        return {
            "final_answer": snap1,
            "snap1": snap1, "snap2": snap2,
            "letter1": letter1, "letter2": letter2,
            "routed_to": "snap_ce_fallback",
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "max_ce_score": max_ce,
            "evidence_store": retrieval["evidence_store"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "gold_retrieved": retrieval["gold_retrieved"],
        }

    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="dsnap/answer")

    return {
        "final_answer": answer,
        "snap1": snap1, "snap2": snap2,
        "letter1": letter1, "letter2": letter2,
        "routed_to": "rag",
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "max_ce_score": max_ce,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_snap_debate(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-debate: snap answer, then a second call sees the first and critiques it.
    Tests whether adversarial self-review improves over simple self-verification. 2 LLM calls."""
    question = _fmt(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="debate/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Adversarial review — explicitly look for errors
    debate_prompt = (
        f"A student answered a legal question. Your job is to find flaws in their reasoning. "
        f"Look for: incorrect legal rules, missing elements, wrong conclusions, or misapplied "
        f"standards. If you find errors, provide the correct answer with your reasoning. "
        f"If the answer is genuinely correct, confirm it and explain why.\n\n"
        f"## Student's Answer\n{snap_answer}\n\n"
        f"## Original Question\n{question}"
    )
    debated = _llm_call(
        "You are a law professor grading an exam. Be critical and precise.",
        debate_prompt, label="debate/critique"
    )

    return {
        "final_answer": debated,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
    }


MODE_RUNNERS = {
    "full_pipeline": run_full_pipeline,
    "llm_only": run_llm_only,
    "rag_rewrite": run_rag_rewrite,
    "rag_simple": run_rag_simple,
    "golden_passage": run_golden_passage,
    "golden_arbitration": run_golden_arbitration,
    "golden_arb_conservative": run_golden_arb_conservative,
    "rag_arbitration": run_rag_arbitration,
    "rag_hyde": run_rag_hyde,
    "rag_hyde_arb": run_rag_hyde_arb,
    "rag_multi_hyde": run_rag_multi_hyde,
    "rag_snap_hyde": run_rag_snap_hyde,
    "snap_hyde_aligned": run_snap_hyde_aligned,
    "gap_hyde": run_gap_hyde,
    "gap_hyde_ev": run_gap_hyde_ev,
    "gap_hyde_nosnap": run_gap_hyde_nosnap,
    "gap_hyde_flat": run_gap_hyde_flat,
    "gap_rag": run_gap_rag,
    "gap_rag_nosnap": run_gap_rag_nosnap,
    "gap_vectorless": run_gap_vectorless,
    "subagent_hyde": run_subagent_hyde,
    "subagent_rag": run_subagent_rag,
    "subagent_hybrid": run_subagent_hybrid,
    "subagent_rag_evidence": run_subagent_rag_evidence,
    "subagent_rag_snap": run_subagent_rag_snap,
    "subagent_rag_full": run_subagent_rag_full,
    "snap_hyde_report": run_snap_hyde_report,
    "snap_hyde_report_snap": run_snap_hyde_report_snap,
    "snap_rag": run_snap_rag,
    "snap_rag_nosnap": run_snap_rag_nosnap,
    "vectorless_direct": run_vectorless_direct,
    "vectorless_role": run_vectorless_role,
    "vectorless_elements": run_vectorless_elements,
    "vectorless_choice_map": run_vectorless_choice_map,
    "vectorless_nosnap": run_vectorless_nosnap,
    "vectorless_hybrid": run_vectorless_hybrid,
    "vectorless_keyword": run_vectorless_keyword,
    "entity_search": run_entity_search,
    "snap_entity_search": run_snap_entity_search,
    "snap_entity_informed": run_snap_entity_informed,
    "rag_devil_hyde": run_rag_devil_hyde,
    "rag_top2_hyde": run_rag_top2_hyde,
    "confidence_gated": run_confidence_gated,
    "decompose": run_decompose,
    "decompose_rag": run_decompose_rag,
    "ce_threshold": run_ce_threshold,
    "conf_ce_threshold": run_conf_ce_threshold,
    "snap_hyde_aspect": run_snap_hyde_aspect,
    "ce_threshold_k3": run_ce_threshold_k3,
    "self_verify": run_self_verify,
    "double_snap": run_double_snap,
    "snap_debate": run_snap_debate,
}


# ---------------------------------------------------------------------------
# Harness Core
# ---------------------------------------------------------------------------

def _setup_provider(config: EvalConfig):
    """Set env vars and clear caches for provider/skill switching."""
    os.environ["LLM_PROVIDER"] = config.provider
    _get_llm_cached.cache_clear()

    if config.skill_dir != "skills":
        os.environ["SKILL_DIR"] = config.skill_dir
        load_skill.cache_clear()


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _serialize_result(result: dict) -> dict:
    """Ensure all values are JSON-serializable."""
    out = {}
    for k, v in result.items():
        if isinstance(v, float) and (v != v):  # NaN check
            out[k] = None
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def run_eval(config: EvalConfig):
    """Run evaluation with the given config."""
    if config.mode not in MODE_RUNNERS:
        print(f"Unknown mode '{config.mode}'. Available: {', '.join(MODE_RUNNERS)}")
        sys.exit(1)

    _setup_provider(config)
    runner = MODE_RUNNERS[config.mode]
    provider_info = get_provider_info()
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None

    qa = load_questions(config)
    n = len(qa)

    print(f"\n{'=' * 70}")
    filter_str = f" | filter={config.source_filter}" if config.source_filter else ""
    dataset_str = f" | dataset={config.dataset}" if config.dataset != "barexam" else ""
    print(f"EVAL: {config.mode} | {provider_info['provider']} ({provider_info['model']}) | {n} questions{dataset_str}{filter_str}")
    if config.skill_dir != "skills":
        print(f"Skills: {config.skill_dir}")
    if config.tag:
        print(f"Tag: {config.tag}")
    print(f"{'=' * 70}\n")

    results = []
    correct = 0
    total_start = time.time()

    is_open_ended = config.dataset in ("legal_rag", "australian")

    for i, row in qa.iterrows():
        _reset_llm_call_counter()
        q_start = time.time()

        # Dataset-specific labeling
        if config.dataset == "housing":
            subject = str(row.get("state", "unknown"))
            label = f"hqa_{subject}_{row.get('idx', i)}"
        elif config.dataset == "casehold":
            subject = "casehold"
            label = f"ch_{row.get('idx', i)}"
        elif config.dataset == "legal_rag":
            subject = "crim_law"
            label = f"lrq_{row.get('idx', i)}"
        elif config.dataset == "australian":
            subject = str(row.get("jurisdiction", "unknown"))
            label = f"aus_{subject}_{row.get('idx', i)}"
        else:
            subject = str(row.get("subject", "unknown"))
            label = f"qa_{subject}_{row.get('idx', i)}"
        idx = str(row.get("idx", i))

        # Gold answer formatting
        gold = str(row["answer"]).strip()
        if config.dataset == "housing":
            gold = gold.capitalize()
        elif config.dataset in ("barexam", "casehold"):
            gold = gold.upper()
        # open-ended: gold stays as-is

        try:
            result = runner(row, config)
            answer_text = result.get("final_answer", "")
            predicted = _extract_answer(answer_text, config)

            if is_open_ended:
                is_correct = _judge_open_answer(row["question"], gold, answer_text, config)
                result["judge_score"] = is_correct  # store for analysis
            else:
                is_correct = predicted == gold
            error = None
        except Exception as e:
            result = {}
            answer_text = ""
            predicted = None
            is_correct = False
            error = str(e)

        elapsed = time.time() - q_start
        metrics = _get_metrics()

        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"
        if is_open_ended:
            print(
                f"[{i+1}/{n}] {label:<35} {status:<6} "
                f"({elapsed:.1f}s, {metrics['count']} calls)",
                flush=True,
            )
        else:
            print(
                f"[{i+1}/{n}] {label:<35} {status:<6} "
                f"gold={gold} pred={predicted} "
                f"({elapsed:.1f}s, {metrics['count']} calls)",
                flush=True,
            )

        # Build per-question record
        record = {
            "label": label,
            "subject": subject,
            "idx": idx,
            "question": str(row["question"])[:500],
            "correct_answer": gold[:500] if is_open_ended else gold,
            "predicted_answer": str(predicted)[:500] if is_open_ended else predicted,
            "is_correct": is_correct,
            "error": error,
            "elapsed_sec": round(elapsed, 1),
            "llm_calls": metrics["count"],
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "gold_idx": str(row.get("gold_idx", "")),
            "final_answer": answer_text[:500] if is_open_ended else answer_text,
            "mode": config.mode,
            "provider": config.provider,
            "dataset": config.dataset,
            "embedding_model": embedding_model,
        }
        if config.dataset == "housing":
            record["state"] = str(row.get("state", ""))
        elif config.dataset == "casehold":
            record["choices"] = {
                letter: str(row.get(f"choice_{letter.lower()}", ""))[:200]
                for letter in "ABCDE"
            }
        elif config.dataset == "australian":
            record["jurisdiction"] = str(row.get("jurisdiction", ""))
        elif config.dataset == "legal_rag":
            record["relevant_passages"] = str(row.get("relevant_passages", ""))
        else:
            record["choices"] = {
                letter: str(row.get(f"choice_{letter.lower()}", ""))
                for letter in "ABCD"
            }
            record["gold_passage"] = str(row.get("gold_passage", ""))[:500]
        # Merge mode-specific fields (evidence_store, audit_log, etc.)
        for k, v in result.items():
            if k != "final_answer" and k not in record:
                record[k] = v

        results.append(_serialize_result(record))

    total_time = time.time() - total_start
    accuracy = correct / n if n > 0 else 0

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{n} ({accuracy*100:.1f}%)")
    print(f"Total time: {total_time:.0f}s ({total_time/n:.1f}s/query)")

    # By-subject breakdown
    by_subject = {}
    for r in results:
        subj = r.get("subject", "unknown")
        if subj not in by_subject:
            by_subject[subj] = [0, 0]
        by_subject[subj][1] += 1
        if r["is_correct"]:
            by_subject[subj][0] += 1

    if len(by_subject) > 1:
        print("\nBy subject:")
        for subj, (c, t) in sorted(by_subject.items()):
            print(f"  {subj:<15} {c}/{t} ({c/t*100:.0f}%)")

    print(f"{'=' * 70}")

    # --- Save detail log ---
    ts = time.strftime("%Y%m%d_%H%M")
    question_set = config.questions if config.questions in ("curated", "full") else f"n{config.questions}"
    detail_filename = f"eval_{config.mode}_{config.provider}_{ts}_detail.jsonl"
    detail_path = os.path.join("logs", detail_filename)
    os.makedirs("logs", exist_ok=True)

    with open(detail_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetail log: {detail_path}")

    # --- Append to experiments.jsonl ---
    run_id = f"{ts}_{config.mode}_{config.provider}"
    if config.tag:
        run_id += f"_{config.tag}"

    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": config.mode,
        "dataset": config.dataset,
        "provider": provider_info["provider"],
        "model": provider_info["model"],
        "embedding_model": embedding_model,
        "question_set": question_set,
        "n_questions": n,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": n,
        "by_subject": by_subject,
        "avg_latency_sec": round(total_time / n, 1) if n else 0,
        "avg_llm_calls": round(sum(r["llm_calls"] for r in results) / n, 1) if n else 0,
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "skill_dir": config.skill_dir,
        "tag": config.tag,
        "source_filter": config.source_filter,
        "detail_log": detail_path,
        "git_commit": _git_commit_short(),
    }

    experiments_path = os.path.join("logs", "experiments.jsonl")
    with open(experiments_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Run summary appended to: {experiments_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model evaluation harness for Legal RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<20} {v}" for k, v in EVAL_MODES.items()),
    )
    parser.add_argument("--mode", default="full_pipeline", choices=EVAL_MODES.keys(),
                        help="Evaluation mode (default: full_pipeline)")
    parser.add_argument("--provider", default="deepseek",
                        help="LLM provider key from llm_config.py (default: deepseek)")
    parser.add_argument("--questions", default="30",
                        help="'curated', 'full', or integer N (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for question sampling (default: 42)")
    parser.add_argument("--skill-dir", default="skills",
                        help="Directory containing skill prompts (default: skills)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--tag", default="",
                        help="Optional label for this run")
    parser.add_argument("--source-filter", default="",
                        help="Metadata source filter for retrieval, e.g. 'mbe' (default: none)")
    parser.add_argument("--dataset", default="barexam",
                        choices=["barexam", "housing", "legal_rag", "australian", "casehold"],
                        help="Dataset to evaluate on (default: barexam)")

    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "1"

    config = EvalConfig(
        mode=args.mode,
        provider=args.provider,
        questions=args.questions,
        seed=args.seed,
        skill_dir=args.skill_dir,
        verbose=args.verbose,
        tag=args.tag,
        source_filter=args.source_filter,
        dataset=args.dataset,
    )

    run_eval(config)


if __name__ == "__main__":
    main()
