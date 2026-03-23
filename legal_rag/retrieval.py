"""Retrieval and per-step answer-generation helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

from langchain_core.documents import Document

from rag_utils import (
    compute_confidence,
    get_vectorstore,
    rerank_with_cross_encoder,
    retrieve_documents,
    retrieve_documents_multi_query,
)
from web_scraper import scrape_urls

from .core import _llm_call, _parse_json, load_skill
from .models import ExperimentProfile, LegalAgentState, PlanningStep, RAG_STRATEGY_ASPECT
from .state_utils import research_question_from_state


def web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo text search."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=k))

        results = []
        for item in raw:
            title = (item.get("title") or "").strip()
            body = (item.get("body") or "").strip()
            href = (item.get("href") or "").strip()
            if title or body:
                results.append({"title": title, "body": body, "href": href})
        return results
    except Exception as exc:
        print(f"    [web_search] Error: {exc}")
        return []


def _enrich_with_scraper(search_results: List[Dict[str, str]], max_scrape: int = 2) -> List[Dict[str, str]]:
    urls = [item["href"] for item in search_results if item.get("href")]
    if not urls:
        return search_results

    scraped = scrape_urls(urls, max_results=max_scrape, max_chars=6000)
    print(f"    Scraped {len(scraped)}/{len(urls)} URLs successfully")

    enriched = list(search_results)
    for item in scraped:
        enriched.append(
            {
                "title": item["title"],
                "body": item["text"],
                "href": item["url"],
                "source": "scraped",
            }
        )
    return enriched


def _build_standard_queries(step: PlanningStep, state: LegalAgentState, profile: ExperimentProfile) -> Dict[str, Any]:
    question = research_question_from_state(state)
    if not profile.use_query_rewrite:
        return {
            "mode": "raw",
            "primary": step.sub_question,
            "alternatives": [],
            "all_queries": [step.sub_question],
        }

    rewrite_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {step.sub_question}\n"
        f"Authority target: {step.authority_target}\n"
        f"Retrieval hints: {', '.join(step.retrieval_hints) if step.retrieval_hints else 'none'}"
    )
    raw = _llm_call(load_skill("query_rewriter"), rewrite_prompt, label="executor/rewrite")
    parsed = _parse_json(raw)
    if parsed and "primary" in parsed:
        primary = parsed["primary"]
        alternatives = parsed.get("alternatives", [])
        return {
            "mode": "rewrite",
            "primary": primary,
            "alternatives": alternatives,
            "all_queries": [primary] + alternatives,
        }
    return {
        "mode": "raw_fallback",
        "primary": step.sub_question,
        "alternatives": [],
        "all_queries": [step.sub_question],
    }


def _build_aspect_queries(step: PlanningStep, state: LegalAgentState) -> Dict[str, Any]:
    question = research_question_from_state(state)
    prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {step.sub_question}\n"
        f"Authority target: {step.authority_target}\n"
        f"Retrieval hints: {', '.join(step.retrieval_hints) if step.retrieval_hints else 'none'}"
    )
    raw = _llm_call(load_skill("aspect_query_rewriter"), prompt, label="executor/aspect")
    parsed = _parse_json(raw)
    aspects = {}
    if parsed:
        for key in ("rule", "exception", "application"):
            value = parsed.get(key)
            if value:
                aspects[key] = value
    if not aspects:
        aspects = {
            "rule": step.sub_question,
            "exception": step.sub_question,
            "application": step.sub_question,
        }
    return aspects


def _evidence_from_docs(step: PlanningStep, docs: Iterable[Document]) -> List[Dict[str, Any]]:
    evidence = []
    for index, doc in enumerate(docs, 1):
        metadata = dict(doc.metadata)
        evidence.append(
            {
                "idx": str(metadata.get("idx", f"step{step.step_id}_{index}")),
                "text": doc.page_content,
                "source": metadata.get("source", "unknown"),
                "step_id": step.step_id,
                "step_ids": [step.step_id],
                "cross_encoder_score": float(metadata.get("cross_encoder_score", 0.0)),
                "retrieval_aspect": metadata.get("retrieval_aspect"),
                "retrieval_query": metadata.get("retrieval_query"),
            }
        )
    return evidence


def _evidence_from_web(step: PlanningStep, enriched: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    evidence = []
    for index, item in enumerate(enriched):
        evidence.append(
            {
                "idx": f"web_{step.step_id}_{index}",
                "text": item.get("body", ""),
                "source": "web_scraped" if item.get("source") == "scraped" else "web_snippet",
                "step_id": step.step_id,
                "step_ids": [step.step_id],
                "cross_encoder_score": 0.0,
                "url": item.get("href", ""),
            }
        )
    return evidence


def _build_passages_for_synthesis(docs: Iterable[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    passages = []
    docs_list = list(docs)
    evidence = []
    for index, doc in enumerate(docs_list, 1):
        metadata = dict(doc.metadata)
        header = f"[Source {index}] ({metadata.get('source', 'unknown')})"
        if metadata.get("retrieval_aspect"):
            header += f" aspect={metadata['retrieval_aspect']}"
        passages.append(f"{header}\n{doc.page_content}")
        evidence.append(metadata)
    return "\n\n".join(passages) if passages else "[No passages retrieved]", evidence


def _clone_doc(doc: Document, **extra_metadata: Any) -> Document:
    metadata = dict(doc.metadata)
    metadata.update(extra_metadata)
    return Document(page_content=doc.page_content, metadata=metadata)


def _retrieve_standard_docs(
    step: PlanningStep,
    state: LegalAgentState,
    table_snapshot: List[PlanningStep],
    evidence_snapshot: List[Dict[str, Any]],
    profile: ExperimentProfile,
) -> Tuple[List[Document], Dict[str, Any]]:
    queries_info = _build_standard_queries(step, state, profile)
    queries = queries_info["all_queries"]

    print(f"    Primary query: {queries_info['primary']}")
    for index, alt in enumerate(queries_info.get("alternatives", []), 1):
        print(f"    Alt query {index}: {alt}")

    prior_ids = set()
    for table_step in table_snapshot:
        prior_ids.update(table_step.evidence_ids)
    for evidence in evidence_snapshot:
        prior_ids.add(str(evidence.get("idx", "")))

    collections = state.get("collections", ["legal_passages"])
    all_docs: List[Document] = []
    for collection_name in collections:
        vectorstore = get_vectorstore(collection_name=collection_name)
        docs = (
            retrieve_documents_multi_query(
                queries=queries,
                k=5,
                exclude_ids=prior_ids or None,
                vectorstore=vectorstore,
                use_bm25=profile.use_bm25,
            )
            if len(queries) > 1
            else retrieve_documents(
                queries[0],
                k=5,
                exclude_ids=prior_ids or None,
                vectorstore=vectorstore,
                use_bm25=profile.use_bm25,
            )
        )
        all_docs.extend(docs)

    if len(collections) > 1 and len(all_docs) > 5:
        all_docs = rerank_with_cross_encoder(queries_info["primary"], all_docs, top_k=5)

    return all_docs[:5], queries_info


def _retrieve_aspect_docs(
    step: PlanningStep,
    state: LegalAgentState,
    table_snapshot: List[PlanningStep],
    evidence_snapshot: List[Dict[str, Any]],
    profile: ExperimentProfile,
) -> Tuple[List[Document], Dict[str, Any]]:
    aspects = _build_aspect_queries(step, state)
    for aspect, query in aspects.items():
        print(f"    {aspect.title()} query: {query}")

    prior_ids = set()
    for table_step in table_snapshot:
        prior_ids.update(table_step.evidence_ids)
    for evidence in evidence_snapshot:
        prior_ids.add(str(evidence.get("idx", "")))

    collections = state.get("collections", ["legal_passages"])
    pooled_docs: List[Document] = []
    for collection_name in collections:
        vectorstore = get_vectorstore(collection_name=collection_name)
        for aspect, query in aspects.items():
            docs = retrieve_documents(
                query,
                k=3,
                exclude_ids=prior_ids or None,
                vectorstore=vectorstore,
                use_bm25=profile.use_bm25,
            )
            pooled_docs.extend(
                _clone_doc(
                    doc,
                    retrieval_aspect=aspect,
                    retrieval_query=query,
                    source=doc.metadata.get("source", "unknown"),
                )
                for doc in docs
            )

    deduped = []
    seen = set()
    for doc in pooled_docs:
        key = str(doc.metadata.get("idx", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    reranked = rerank_with_cross_encoder(step.sub_question, deduped, top_k=6)
    return reranked, {"mode": "aspect", "aspects": aspects}


def _execute_direct_answer(step: PlanningStep, state: LegalAgentState) -> Tuple[str, List[Dict[str, Any]], float, Dict[str, Any]]:
    question = research_question_from_state(state)
    prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question to answer from established legal doctrine:\n{step.sub_question}\n\n"
        f"Authority target: {step.authority_target}\n\n"
        f"Provide a clear, well-grounded answer based on established legal doctrine. "
        f"Flag any uncertainty or contested areas explicitly."
    )
    result = _llm_call(load_skill("synthesize_and_cite"), prompt, label="executor/direct")
    return result, [], 0.0, {"mode": "direct"}


def _execute_rag_search(
    step: PlanningStep,
    state: LegalAgentState,
    table_snapshot: List[PlanningStep],
    evidence_snapshot: List[Dict[str, Any]],
    profile: ExperimentProfile,
) -> Tuple[str, List[Dict[str, Any]], float, Dict[str, Any]]:
    question = research_question_from_state(state)
    if profile.rag_strategy == RAG_STRATEGY_ASPECT:
        docs, query_trace = _retrieve_aspect_docs(step, state, table_snapshot, evidence_snapshot, profile)
    else:
        docs, query_trace = _retrieve_standard_docs(step, state, table_snapshot, evidence_snapshot, profile)

    print(f"    Retrieved {len(docs)} passage(s) from {state.get('collections', ['legal_passages'])}")
    passages, _ = _build_passages_for_synthesis(docs)
    synth_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Research sub-question: {step.sub_question}\n\n"
        f"Evidence passages:\n{passages}"
    )
    result = _llm_call(load_skill("synthesize_and_cite"), synth_prompt, label="executor/synth")
    raw_logit = compute_confidence(step.sub_question, docs) if docs else 0.0
    return result, _evidence_from_docs(step, docs), raw_logit, query_trace


def _execute_web_search(step: PlanningStep, state: LegalAgentState) -> Tuple[str, List[Dict[str, Any]], float, Dict[str, Any]]:
    question = research_question_from_state(state)
    search_results = web_search(step.sub_question, k=5)
    print(f"    Web results: {len(search_results)} result(s)")
    for index, item in enumerate(search_results, 1):
        print(f"      [{index}] {item.get('title', '(no title)')}")
        if item.get("href"):
            print(f"          {item['href']}")

    if not search_results:
        return "[No web results found]", [], 0.0, {"mode": "web", "urls": []}

    enriched = _enrich_with_scraper(search_results, max_scrape=2)
    passages = []
    for index, item in enumerate(enriched, 1):
        source_tag = "Scraped" if item.get("source") == "scraped" else "Snippet"
        header = f"[WebResult {index}] ({source_tag}) {item.get('title', '')}"
        if item.get("href"):
            header += f"\nURL: {item['href']}"
        passages.append(f"{header}\n{item.get('body', '')}")

    synth_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Research sub-question: {step.sub_question}\n\n"
        f"Web search results:\n{os.linesep.join(passages)}"
    )
    result = _llm_call(load_skill("synthesize_and_cite"), synth_prompt, label="executor/web")
    return result, _evidence_from_web(step, enriched), 0.0, {
        "mode": "web",
        "urls": [item.get("href") for item in enriched if item.get("href")],
    }


__all__ = [
    "_build_passages_for_synthesis",
    "_execute_direct_answer",
    "_execute_rag_search",
    "_execute_web_search",
    "web_search",
]
