"""Profile-driven baseline runners used by runtime and evals."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from llm_config import get_provider_info
from rag_utils import retrieve_documents, retrieve_documents_multi_query

from .core import _llm_call, _parse_json, load_skill
from .models import ExecutionResult, ExperimentProfile

LLM_ONLY_PROMPT = (
    "Answer the following legal question. Reason through it step by step, "
    "then give your final answer as **Answer: (X)**"
)

RAG_ANSWER_PROMPT = """Answer the following legal question using the retrieved passages as your primary source. You may also apply established legal doctrine where the passages are insufficient.

Reason through it step by step, then give your final answer as **Answer: (X)**"""


def _base_metadata(profile: ExperimentProfile) -> Dict[str, Any]:
    provider = get_provider_info()
    return {
        "provider": provider.get("provider", "unknown"),
        "model": provider.get("model", "unknown"),
        "profile_name": profile.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _docs_to_evidence(docs) -> List[Dict[str, Any]]:
    evidence = []
    for index, doc in enumerate(docs, 1):
        evidence.append(
            {
                "idx": str(doc.metadata.get("idx", f"doc_{index}")),
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "step_ids": [1],
                "cross_encoder_score": float(doc.metadata.get("cross_encoder_score", 0.0)),
            }
        )
    return evidence


def run_llm_only(question: str, profile: ExperimentProfile, raw_question: str | None = None) -> ExecutionResult:
    answer = _llm_call(LLM_ONLY_PROMPT, question, label=profile.name).strip()
    return ExecutionResult(
        profile=profile,
        final_answer=answer,
        agent_metadata=_base_metadata(profile),
        extra={"raw_question": raw_question or question},
    )


def _rewrite_queries(raw_question: str) -> List[str]:
    rewrite_prompt = (
        f"Original legal research question: {raw_question}\n\n"
        f"Sub-question: {raw_question}\n"
        f"Authority target: legal doctrine\n"
        f"Retrieval hints: none"
    )
    raw = _llm_call(load_skill("query_rewriter"), rewrite_prompt, label="rewrite")
    parsed = _parse_json(raw)
    if parsed and "primary" in parsed:
        return [parsed["primary"]] + parsed.get("alternatives", [])
    return [raw_question]


def run_rag_baseline(
    question: str,
    profile: ExperimentProfile,
    raw_question: str | None = None,
) -> ExecutionResult:
    retrieval_question = raw_question or question
    started = time.perf_counter()

    if profile.use_query_rewrite:
        queries = _rewrite_queries(retrieval_question)
        docs = retrieve_documents_multi_query(queries, k=5, use_bm25=profile.use_bm25)
    else:
        queries = [retrieval_question]
        docs = retrieve_documents(retrieval_question, k=5, use_bm25=profile.use_bm25)

    passages = "\n\n".join(
        f"[Passage {index + 1}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
        for index, doc in enumerate(docs)
    )
    answer = _llm_call(RAG_ANSWER_PROMPT, f"RETRIEVED PASSAGES:\n{passages}\n\nQUESTION:\n{question}", label=profile.name).strip()

    return ExecutionResult(
        profile=profile,
        final_answer=answer,
        evidence_store=_docs_to_evidence(docs),
        agent_metadata=_base_metadata(profile),
        extra={
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "retrieved_ids": [str(doc.metadata.get("idx", "")) for doc in docs],
            "rewrite_queries": queries,
            "raw_question": retrieval_question,
        },
    )
