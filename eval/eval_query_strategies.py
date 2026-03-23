"""Compare query rewriting strategies for retrieval quality.

Tests multiple query generation approaches on the same questions
and compares: what passages does each strategy find? How relevant
are they? Does the gold passage appear?

No full pipeline — just query → retrieve → inspect.

Usage:
  uv run python eval/eval_query_strategies.py             # Default 5 questions
  uv run python eval/eval_query_strategies.py 10           # 10 questions
  uv run python eval/eval_query_strategies.py --verbose    # Show passage text
"""

import os
import sys
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from main import _llm_call, _parse_json, load_skill
from rag_utils import retrieve_documents_multi_query, rerank_with_cross_encoder
from eval.eval_utils import select_qa_queries


# ---------------------------------------------------------------------------
# Strategy definitions — each returns a list of query strings
# ---------------------------------------------------------------------------

def strategy_raw(question: str, **kw) -> list[str]:
    """Strategy 0: Raw question, no rewriting at all."""
    return [question]


def strategy_current(question: str, **kw) -> list[str]:
    """Strategy 1: Current query_rewriter.md — synonym alternatives."""
    prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {question}\n"
        f"Authority target: legal doctrine\n"
        f"Retrieval hints: none"
    )
    raw = _llm_call(load_skill("query_rewriter"), prompt, label="rewrite/current")
    parsed = _parse_json(raw)
    if parsed and "primary" in parsed:
        return [parsed["primary"]] + parsed.get("alternatives", [])
    return [question]


def strategy_aspect(question: str, **kw) -> list[str]:
    """Strategy 2: Aspect-based — target different legal dimensions."""
    prompt = f"""Given this legal question, generate THREE retrieval queries that each target a DIFFERENT legal aspect:

1. **Rule query**: Target the governing rule, doctrine, or elements (textbook-level)
2. **Exception query**: Target exceptions, defenses, or limitations to the rule
3. **Application query**: Target how courts apply this rule to specific facts

Return ONLY valid JSON:
{{"rule": "...", "exception": "...", "application": "..."}}

Question: {question}"""

    raw = _llm_call(prompt, question, label="rewrite/aspect")
    parsed = _parse_json(raw)
    if parsed:
        queries = []
        for key in ["rule", "exception", "application"]:
            if key in parsed:
                queries.append(parsed[key])
        if queries:
            return queries
    return [question]


def strategy_decompose(question: str, **kw) -> list[str]:
    """Strategy 3: Decompose into sub-concepts, one query per concept."""
    prompt = f"""Given this legal question, identify the 2-3 KEY legal concepts it tests and generate a focused retrieval query for each.

Each query should be 10-20 words of dense legal terminology targeting ONE specific concept.

Return ONLY valid JSON:
{{"queries": ["query1", "query2", "query3"]}}

Question: {question}"""

    raw = _llm_call(prompt, question, label="rewrite/decompose")
    parsed = _parse_json(raw)
    if parsed and "queries" in parsed:
        return parsed["queries"][:3]
    return [question]


def strategy_abstract(question: str, **kw) -> list[str]:
    """Strategy 4: Abstract the question to pure doctrine, stripping all facts."""
    prompt = f"""Extract the pure legal doctrine being tested by this question. Strip ALL fact-specific details (names, places, specific actions) and rewrite as a textbook-style query.

Return ONLY valid JSON:
{{"abstract": "...", "keywords": "..."}}

- abstract: The question rephrased as a pure doctrine query (10-20 words)
- keywords: Dense legal keywords for retrieval (10-15 words)

Question: {question}"""

    raw = _llm_call(prompt, question, label="rewrite/abstract")
    parsed = _parse_json(raw)
    if parsed:
        queries = []
        if "abstract" in parsed:
            queries.append(parsed["abstract"])
        if "keywords" in parsed:
            queries.append(parsed["keywords"])
        if queries:
            return queries
    return [question]


STRATEGIES = {
    "raw": strategy_raw,
    "current": strategy_current,
    "aspect": strategy_aspect,
    "decompose": strategy_decompose,
    "abstract": strategy_abstract,
}


def evaluate_strategy(name: str, queries: list[str], gold_idx: str,
                      k: int = 5, verbose: bool = False) -> dict:
    """Run retrieval with given queries and check gold passage recall."""
    docs = retrieve_documents_multi_query(queries, k=k)

    retrieved_ids = [doc.metadata.get("idx", "") for doc in docs]
    scores = [doc.metadata.get("cross_encoder_score", 0) for doc in docs]
    gold_found = gold_idx in retrieved_ids
    gold_rank = retrieved_ids.index(gold_idx) + 1 if gold_found else None

    result = {
        "strategy": name,
        "queries": queries,
        "gold_found": gold_found,
        "gold_rank": gold_rank,
        "top_score": max(scores) if scores else 0,
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "retrieved_ids": retrieved_ids,
        "sources": [doc.metadata.get("source", "?") for doc in docs],
    }

    if verbose:
        for i, doc in enumerate(docs):
            idx = doc.metadata.get("idx", "")
            score = doc.metadata.get("cross_encoder_score", 0)
            gold_tag = " *** GOLD ***" if idx == gold_idx else ""
            print(f"      [{i+1}] {idx} score={score:.3f}{gold_tag}")
            print(f"          {doc.page_content[:150]}...")

    return result


def main():
    args = sys.argv[1:]
    verbose = "--verbose" in args
    if verbose:
        args.remove("--verbose")
    n = int(args[0]) if args else 5

    print(f"\n{'='*80}")
    print(f"QUERY STRATEGY COMPARISON ({n} questions, {len(STRATEGIES)} strategies)")
    print(f"{'='*80}\n")

    queries = select_qa_queries(n)

    # Track per-strategy stats
    stats = {name: {"gold_hits": 0, "total_score": 0, "count": 0}
             for name in STRATEGIES}

    all_results = []

    for i, q in enumerate(queries):
        question = q["question"]
        gold_idx = str(q.get("gold_idx", ""))

        print(f"\n{'─'*80}")
        print(f"[{i+1}/{n}] {q['label']}")
        print(f"  Q: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"  Gold: {gold_idx}")

        query_results = {}
        for name, strategy_fn in STRATEGIES.items():
            t0 = time.time()
            try:
                rewritten = strategy_fn(question)
            except Exception as e:
                print(f"  {name}: ERROR — {e}")
                rewritten = [question]
            elapsed = time.time() - t0

            result = evaluate_strategy(
                name, rewritten, gold_idx, k=5, verbose=verbose
            )
            result["rewrite_time"] = round(elapsed, 1)
            query_results[name] = result

            gold_tag = f"GOLD@{result['gold_rank']}" if result['gold_found'] else "no gold"
            src_summary = ", ".join(f"{result['sources'].count(s)} {s}"
                                    for s in set(result['sources']))
            print(f"  {name:<12} top={result['top_score']:>6.3f} "
                  f"mean={result['mean_score']:>6.3f} {gold_tag:<10} "
                  f"[{src_summary}] ({elapsed:.1f}s)")
            if verbose or name != "raw":
                for j, rq in enumerate(rewritten[:2]):
                    print(f"    q{j+1}: {rq[:80]}")

            stats[name]["count"] += 1
            stats[name]["total_score"] += result["mean_score"]
            if result["gold_found"]:
                stats[name]["gold_hits"] += 1

        all_results.append({"label": q["label"], "gold_idx": gold_idx,
                            "strategies": query_results})

    # Summary
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<14} {'Gold Recall':>12} {'Avg Top Score':>14} {'Avg Mean Score':>15}")
    print(f"{'─'*58}")
    for name in STRATEGIES:
        s = stats[name]
        gold_rate = s["gold_hits"] / s["count"] * 100 if s["count"] else 0
        avg_score = s["total_score"] / s["count"] if s["count"] else 0
        print(f"{name:<14} {s['gold_hits']:>3}/{s['count']:<3} ({gold_rate:>4.0f}%) "
              f"{'':>8} {avg_score:>14.3f}")
    print(f"{'='*80}\n")

    # Save detailed results
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H")
    detail_file = f"logs/eval_query_strategies_{timestamp}.jsonl"
    with open(detail_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Detail log saved to {detail_file}")


if __name__ == "__main__":
    main()
